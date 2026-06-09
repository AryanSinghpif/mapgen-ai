"""
data_analyzer.py — Data profiling, cleaning, and format detection
=================================================================
Runs before column selection. Handles:
  - Wide format  : one row per geography, one column per variable  (most common)
  - Long format  : geo | variable | value  (tidy/melted data)
  - Messy headers: extra blank rows, merged-cell artifacts, numeric first row
  - Mixed numerics: values with %, commas, currency symbols
  - State vs district level detection
  - Multi-sheet Excel (picks the most data-rich sheet)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Known geography keywords ──────────────────────────────────────────────────

_GEO_KEYWORDS = [
    "district", "state", "region", "zone", "division", "taluk", "tehsil",
    "mandal", "block", "geography", "location", "place", "area", "name",
    "unit", "entity", "ut", "uts", "state_ut", "dist", "dname", "sname",
]

_INDIA_STATES_NORM = {
    # 28 states
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal",
    # 8 UTs
    "delhi", "jammu and kashmir", "ladakh", "puducherry",
    "andaman and nicobar islands", "chandigarh", "lakshadweep",
    "dadra and nagar haveli and daman and diu",
    # common alternate spellings (post-_norm, & already replaced with ' and ')
    "jammu and kashmir", "jammu kashmir",
    "andaman and nicobar", "a and n islands",
    "daman and diu", "dadra and nagar haveli",
    # abbreviations
    "up", "mp", "ap", "tn", "wb", "hp", "uk", "j and k", "jk",
    # total / all india rows (so they don't lower match rate unfairly)
    "india", "all india", "total",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s*&\s*", " and ", s)          # "J&K" → "j and k", "Jammu & Kashmir" → "jammu and kashmir"
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip().lower()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DataProfile:
    fmt: str                        # "wide" | "long" | "unknown"
    level: str                      # "district" | "state" | "unknown"
    geo_col: Optional[str]          # detected geography column
    value_cols: list[str]           # candidate numeric columns to map
    variable_col: Optional[str]     # long-format: column holding variable names
    long_value_col: Optional[str]   # long-format: column holding numeric values
    n_rows: int
    n_cols: int
    issues: list[str] = field(default_factory=list)
    notes: list[str]  = field(default_factory=list)


@dataclass
class CleanResult:
    df: pd.DataFrame
    changes: list[str] = field(default_factory=list)


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> CleanResult:
    """
    Systematic cleaning pass:
      1. Drop entirely empty rows/cols
      2. Fix numeric columns that contain %, commas, currency symbols
      3. Strip whitespace from all string cells
      4. Detect and skip extra header rows (all-NaN or repeated header)
      5. Reset index
    """
    changes: list[str] = []
    df = df.copy()

    # 1 — Drop fully empty rows and columns
    before = df.shape
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    if df.shape != before:
        changes.append(f"Dropped {before[0]-df.shape[0]} empty rows, "
                       f"{before[1]-df.shape[1]} empty columns.")

    # 2 — Detect stray header rows (first N rows where >half cells are non-numeric strings)
    df = _fix_header_rows(df, changes)

    # 3 — Strip whitespace from object columns
    for col in df.select_dtypes("object").columns:
        cleaned = df[col].astype(str).str.strip()
        if (cleaned != df[col].astype(str)).any():
            df[col] = cleaned
    changes.append("Stripped leading/trailing whitespace from all text cells.")

    # 4 — Clean numeric columns (remove %, commas, currency)
    for col in df.columns:
        if df[col].dtype == object:
            coerced = _try_numeric(df[col])
            if coerced is not None:
                df[col] = coerced
                changes.append(f"Converted '{col}' to numeric (removed %, commas, symbols).")

    df.reset_index(drop=True, inplace=True)
    return CleanResult(df=df, changes=changes)


def _fix_header_rows(df: pd.DataFrame, changes: list[str]) -> pd.DataFrame:
    """
    If the first real data row looks like a repeated header (all strings,
    matches column names), drop it.
    """
    if len(df) < 2:
        return df
    first = df.iloc[0]
    if first.astype(str).str.lower().tolist() == [c.lower() for c in df.columns]:
        df = df.iloc[1:].reset_index(drop=True)
        changes.append("Removed duplicate header row detected as first data row.")
    return df


def _try_numeric(series: pd.Series) -> Optional[pd.Series]:
    """
    Try to coerce a string series to numeric by stripping common non-numeric
    characters. Returns None if the result is mostly NaN (not worth converting).
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[%,₹$€£\s]", "", regex=True)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    coerced = pd.to_numeric(cleaned, errors="coerce")
    non_null_original = series.notna().sum()
    non_null_coerced  = coerced.notna().sum()
    if non_null_original == 0:
        return None
    if non_null_coerced / non_null_original >= 0.75:
        return coerced
    return None


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(df: pd.DataFrame) -> tuple[str, Optional[str], Optional[str]]:
    """
    Returns (format, variable_col, value_col).
    format = "wide" | "long"

    Long format heuristic:
      - Has a column where all values are strings and the unique count is LOW
        (looks like a variable name column, e.g. "Indicator", "Metric")
      - Has exactly one numeric column
      - OR: has a column named 'variable'/'indicator'/'metric'/'measure'
    """
    obj_cols = list(df.select_dtypes("object").columns)
    num_cols = list(df.select_dtypes("number").columns)

    # Explicit name signal
    long_signals = {"variable", "indicator", "metric", "measure", "category",
                    "parameter", "attribute", "type", "year", "period"}

    for col in obj_cols:
        col_n = _norm(col)
        if any(sig in col_n for sig in long_signals):
            # Candidate variable column — check it has few unique values relative to rows
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio < 0.5 and len(num_cols) == 1:
                return "long", col, num_cols[0]

    # Structural: one numeric col, one geo col, one other string col with low cardinality
    if len(num_cols) == 1 and len(obj_cols) >= 2:
        # Find the low-cardinality string col (variable) vs high-cardinality (geo)
        cardinalities = {col: df[col].nunique() for col in obj_cols}
        sorted_cols = sorted(cardinalities, key=lambda c: cardinalities[c])
        low_card = sorted_cols[0]
        high_card = sorted_cols[-1]
        if (
            cardinalities[low_card] <= 20          # ≤20 unique variable names
            and cardinalities[high_card] >= 10     # geo column has ≥10 unique places
            and cardinalities[low_card] < cardinalities[high_card]
        ):
            return "long", low_card, num_cols[0]

    return "wide", None, None


# ── Geography column detection ────────────────────────────────────────────────

def detect_geo_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically identify the geography (district/state name) column.
    Scoring: column name keywords + value pattern matching.
    """
    obj_cols = list(df.select_dtypes("object").columns)
    if not obj_cols:
        return None

    scores: dict[str, float] = {}

    for col in obj_cols:
        score = 0.0
        col_n = _norm(col)

        # Name-based signal
        for kw in _GEO_KEYWORDS:
            if kw in col_n:
                score += 3.0
                break

        # Value-based: check how many values look like Indian place names
        sample = df[col].dropna().astype(str).head(30)
        state_hits = sum(1 for v in sample if _norm(v) in _INDIA_STATES_NORM)
        score += state_hits * 0.5

        # High unique count relative to rows = likely a name column
        unique_ratio = df[col].nunique() / max(len(df), 1)
        if 0.3 < unique_ratio <= 1.05:
            score += 1.5

        # Avoid columns that look like free text (avg word count > 4)
        avg_words = sample.str.split().str.len().mean() if len(sample) else 0
        if avg_words and avg_words > 4:
            score -= 2.0

        scores[col] = score

    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else obj_cols[0]


# ── Level detection ───────────────────────────────────────────────────────────

def detect_level(df: pd.DataFrame, geo_col: str) -> str:
    """
    'state' if ≥50% of values in geo_col match known Indian state names,
    else 'district'.
    """
    vals = df[geo_col].dropna().astype(str)
    hits = sum(1 for v in vals if _norm(v) in _INDIA_STATES_NORM)
    return "state" if hits / max(len(vals), 1) >= 0.50 else "district"


# ── Master profile ────────────────────────────────────────────────────────────

def profile_data(df: pd.DataFrame) -> DataProfile:
    """
    Full analysis: clean → detect format → detect geo col → detect level.
    Returns DataProfile. df is NOT mutated; caller should use CleanResult.df.
    """
    issues: list[str] = []
    notes:  list[str] = []

    # Format detection
    fmt, variable_col, long_value_col = detect_format(df)
    if fmt == "long":
        notes.append(
            f"Long (tidy) format detected — '{variable_col}' holds variable names, "
            f"'{long_value_col}' holds values. Will pivot to wide before mapping."
        )

    # Working frame: if long, pivot to wide first for geo/level detection
    if fmt == "long" and variable_col and long_value_col:
        geo_col_candidate = detect_geo_column(df)
        try:
            wide = df.pivot_table(
                index=geo_col_candidate,
                columns=variable_col,
                values=long_value_col,
                aggfunc="mean",
            ).reset_index()
            wide.columns.name = None
            work_df = wide
        except Exception as e:
            notes.append(f"Could not auto-pivot: {e}. Treating as wide.")
            fmt = "wide"
            work_df = df
    else:
        work_df = df

    geo_col   = detect_geo_column(work_df)
    level     = detect_level(work_df, geo_col) if geo_col else "unknown"
    num_cols  = list(work_df.select_dtypes("number").columns)
    value_cols = [c for c in num_cols if c != geo_col]

    if not geo_col:
        issues.append("Could not detect a geography (district/state name) column.")
    if not value_cols:
        issues.append("No numeric columns found — nothing to map.")
    if len(value_cols) > 10:
        notes.append(f"{len(value_cols)} numeric columns found. Pick the one to map.")

    return DataProfile(
        fmt=fmt,
        level=level,
        geo_col=geo_col,
        value_cols=value_cols,
        variable_col=variable_col,
        long_value_col=long_value_col,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        issues=issues,
        notes=notes,
    )


# ── Emoji suggester ───────────────────────────────────────────────────────────

_EMOJI_MAP: list[tuple[list[str], str, str]] = [
    # (keywords, emoji, label)
    (["gdp", "gva", "gross value", "income", "revenue", "economy", "economic",
      "fiscal", "tax", "expenditure", "budget", "finance", "monetary"],           "💰", "Economy"),
    (["population", "census", "household", "hh", "hhsize", "residents",
      "demographic", "density", "persons", "people"],                             "👥", "Population"),
    (["literacy", "education", "school", "enroll", "dropout", "learning",
      "teacher", "college", "university", "student"],                             "📚", "Education"),
    (["health", "hospital", "mortality", "disease", "death", "birth",
      "medical", "medicine", "clinic", "doctor", "nurse", "mmr", "imr",
      "malaria", "tb", "hiv", "covid", "nutrition", "malnutrition"],              "🏥", "Health"),
    (["power", "electricity", "energy", "plant", "solar", "wind",
      "renewable", "electrification", "kwh", "megawatt", "grid"],                "⚡", "Energy"),
    (["water", "sanitation", "toilet", "drainage", "sewage", "hygiene",
      "swachh", "clean water", "drinking", "handwash", "wash"],                  "💧", "Water & Sanitation"),
    (["road", "transport", "highway", "railway", "rail", "bus", "vehicle",
      "accident", "traffic", "connectivity", "aviation", "port"],                "🛣️", "Transport"),
    (["agriculture", "agri", "crop", "farm", "yield", "harvest", "kharif",
      "rabi", "irrigation", "soil", "fertilizer", "pesticide", "msp"],           "🌾", "Agriculture"),
    (["forest", "tree", "green", "environment", "ecology", "pollution",
      "emission", "carbon", "biodiversity", "wildlife", "air quality"],          "🌳", "Environment"),
    (["crime", "violence", "safety", "police", "fir", "assault", "theft",
      "murder", "rape", "atrocity", "law", "order"],                             "🔒", "Crime & Safety"),
    (["poverty", "hunger", "bpl", "below poverty", "deprivation",
      "multidimensional", "mpi", "destitute"],                                   "📉", "Poverty"),
    (["employment", "jobs", "labour", "labor", "wage", "salary", "nrega",
      "mgnrega", "unemployment", "workforce", "worker", "earning"],              "👷", "Employment"),
    (["gender", "women", "woman", "female", "sex ratio", "girl",
      "maternal", "mahila", "beti", "widow", "dowry"],                           "♀️", "Gender"),
    (["child", "infant", "birth", "fertility", "under5", "stunting",
      "wasting", "anemia", "anganwadi", "icds", "poshan"],                       "👶", "Child & Infant"),
    (["rank", "index", "score", "rating", "performance", "composite",
      "indicator", "percentile"],                                                 "🏆", "Index / Rank"),
    (["housing", "house", "dwelling", "pucca", "kutcha", "homeless",
      "slum", "shelter", "roof"],                                                 "🏠", "Housing"),
    (["industry", "manufacturing", "factory", "msme", "industrial",
      "production", "output", "iip"],                                             "🏭", "Industry"),
    (["migration", "migrant", "refugee", "displacement", "urban",
      "urbanisation", "urbanization", "city", "town"],                           "🚶", "Migration / Urban"),
    (["internet", "digital", "mobile", "telecom", "broadband", "wifi",
      "tech", "e-governance"],                                                   "📶", "Digital"),
    (["disaster", "flood", "drought", "cyclone", "earthquake", "relief",
      "ndrf", "sdrf", "calamity"],                                               "🌊", "Disaster"),
]

_DEFAULT_EMOJI = "📊"


def suggest_emoji(prompt: str = "", col_name: str = "") -> list[tuple[str, str]]:
    """
    Match prompt + col_name against keyword categories.
    Returns list of (emoji, label) tuples, best matches first (up to 3).
    Falls back to [("📊", "Data")] if nothing matches.
    """
    text = _norm(f"{prompt} {col_name}").lower()
    scores: list[tuple[int, str, str]] = []

    for keywords, emoji, label in _EMOJI_MAP:
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scores.append((hits, emoji, label))

    scores.sort(reverse=True)
    result = [(e, l) for _, e, l in scores[:3]]
    return result if result else [(_DEFAULT_EMOJI, "Data")]


def pivot_long_to_wide(
    df: pd.DataFrame,
    geo_col: str,
    variable_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Pivot long→wide and return clean wide DataFrame."""
    wide = df.pivot_table(
        index=geo_col,
        columns=variable_col,
        values=value_col,
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide
