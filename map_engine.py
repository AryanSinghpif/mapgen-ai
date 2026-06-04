"""
map_engine.py — Deterministic core for District Map Generator
=============================================================
All logic here is rule-based and reproducible. No LLM calls.
LLM fallback (Groq) lives in groq_matcher.py and is called only for names
that survive all four tiers below.

Tier chain (in order):
  1. Exact string match
  2. Normalized match  (lowercase, no accents/punctuation)
  3. Alias lookup      (curated India-specific rename dictionary)
  4. Fuzzy match       (rapidfuzz SequenceMatcher)
  5. → groq_matcher.py (external, optional)
  6. → Human confirmation gate (handled by app.py / flows.py)
"""

from __future__ import annotations

import io
import json
import os
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
import mapclassify
import folium
from folium.features import GeoJsonTooltip
from rapidfuzz import fuzz, process as fuzz_process

from aliases import ALIASES


# ── Constants ────────────────────────────────────────────────────────────────

HIGH_CONF_THRESHOLD = 0.88   # auto-accept above this
FUZZY_CUTOFF       = 0.70   # fuzzy matches below this are sent to Groq/human
MISSING_COLOR      = "#c8c8c8"
MISSING_HATCH      = "///"
NO_DATA_LABEL      = "No data"

CLASSIFICATION_SCHEMES = {
    "quantiles":      "Quantiles — equal number of districts per bin. "
                      "Good for ranked comparisons; hides within-group spread.",
    "equal_interval": "Equal interval — bins of equal width. "
                      "Intuitive but skewed by outliers.",
    "fisher_jenks":   "Natural Breaks (Fisher-Jenks) — minimises within-class variance. "
                      "Best general-purpose choice for skewed distributions.",
}

COLOR_RAMPS = {
    "Blues":     "Sequential blue — good for positive indicators (population, income).",
    "Reds":      "Sequential red — good for risk/stress indicators.",
    "Greens":    "Sequential green — good for vegetation, growth, literacy.",
    "Purples":   "Sequential purple — neutral alternative to blue.",
    "Oranges":   "Sequential orange — warm, suitable for economic activity.",
    "RdYlGn":    "Diverging red-yellow-green — use when data has a meaningful midpoint.",
    "RdBu":      "Diverging red-blue — good for above/below average comparisons.",
    "YlOrRd":    "Multi-hue yellow-orange-red — high contrast for print.",
    "viridis":   "Perceptually uniform, colourblind-safe.",
}

ICON_MAP = {
    "population":   "👥",
    "income":       "💰",
    "literacy":     "📚",
    "industry":     "🏭",
    "agriculture":  "🌾",
    "location":     "📍",
    "health":       "🏥",
    "education":    "🎓",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DistrictMatch:
    """One resolved district name."""
    data_name:  str
    shp_name:   Optional[str]       # None if completely unresolved
    tier:       str                 # "exact" | "normalized" | "alias" | "fuzzy" | "groq" | "manual"
    confidence: float               # 0–1
    note:       str = ""            # human-readable explanation


@dataclass
class MatchResult:
    """Full output of the matching pipeline."""
    high_confidence: list[DistrictMatch] = field(default_factory=list)
    low_confidence:  list[DistrictMatch] = field(default_factory=list)   # needs human review
    unmatched:       list[str]           = field(default_factory=list)   # no candidate found

    def all_confirmed(self) -> list[DistrictMatch]:
        """Return all matches that have a shp_name (exclude unmatched)."""
        return self.high_confidence + [m for m in self.low_confidence if m.shp_name]

    def as_dict(self) -> dict[str, str | None]:
        """data_name → shp_name mapping for all confirmed matches."""
        return {m.data_name: m.shp_name for m in self.all_confirmed()}

    def summary(self) -> str:
        total = len(self.high_confidence) + len(self.low_confidence) + len(self.unmatched)
        return (
            f"{len(self.high_confidence)}/{total} auto-matched | "
            f"{len(self.low_confidence)} need review | "
            f"{len(self.unmatched)} unmatched"
        )


# ── Name normalization ────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """
    Canonical form used for all matching tiers.
    Steps: strip accents → lowercase → remove punctuation → collapse whitespace.
    """
    if not isinstance(name, str):
        return ""
    # NFKD decompose → drop non-ASCII (strips diacritics like ā, ū, ñ)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Replace punctuation/special chars with space
    name = re.sub(r"[^\w\s]", " ", name)
    # Collapse internal whitespace
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


# ── Alias loading ─────────────────────────────────────────────────────────────

def _build_alias_map() -> dict[str, str]:
    """Return {normalized_alt → normalized_canonical} from aliases.py."""
    return {normalize(k): normalize(v) for k, v in ALIASES.items()}


_ALIAS_MAP: dict[str, str] = _build_alias_map()


# ── Tiered district matching ──────────────────────────────────────────────────

def match_districts(
    data_names: list[str],
    shp_names:  list[str],
    fuzzy_cutoff: float = FUZZY_CUTOFF,
) -> MatchResult:
    """
    Runs tiers 1–4. Returns MatchResult.
    Tier 5 (Groq) and Tier 6 (human) are handled externally.
    """
    # Build fast lookup: normalized_shp → original_shp
    shp_norm_to_orig: dict[str, str] = {}
    for s in shp_names:
        n = normalize(s)
        if n not in shp_norm_to_orig:  # first-seen wins (avoids collision overwrite)
            shp_norm_to_orig[n] = s
    shp_norm_list = list(shp_norm_to_orig.keys())

    result = MatchResult()

    for name in data_names:
        norm_name = normalize(name)

        # ── Tier 1: Exact string match ────────────────────────────────────
        if name in shp_names:
            result.high_confidence.append(
                DistrictMatch(name, name, "exact", 1.0, "Exact string match")
            )
            continue

        # ── Tier 2: Normalized match ──────────────────────────────────────
        if norm_name in shp_norm_to_orig:
            result.high_confidence.append(
                DistrictMatch(name, shp_norm_to_orig[norm_name], "normalized", 0.99,
                              "Matched after normalization (case/accent/punctuation)")
            )
            continue

        # ── Tier 3: Alias lookup ──────────────────────────────────────────
        aliased_norm = _ALIAS_MAP.get(norm_name)
        if aliased_norm and aliased_norm in shp_norm_to_orig:
            result.high_confidence.append(
                DistrictMatch(name, shp_norm_to_orig[aliased_norm], "alias", 0.97,
                              f"Known alias → {shp_norm_to_orig[aliased_norm]}")
            )
            continue

        # ── Tier 4: Fuzzy match (rapidfuzz) ──────────────────────────────
        fuzzy_hit = fuzz_process.extractOne(
            norm_name,
            shp_norm_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=fuzzy_cutoff * 100,   # rapidfuzz uses 0–100
        )
        if fuzzy_hit:
            matched_norm, raw_score, _ = fuzzy_hit
            confidence = raw_score / 100.0
            match = DistrictMatch(
                name,
                shp_norm_to_orig[matched_norm],
                "fuzzy",
                confidence,
                f"Fuzzy match (score {raw_score:.0f}/100)",
            )
            if confidence >= HIGH_CONF_THRESHOLD:
                result.high_confidence.append(match)
            else:
                result.low_confidence.append(match)
            continue

        # ── No match found — will go to Groq / human ─────────────────────
        result.unmatched.append(name)

    return result


def apply_manual_corrections(
    result: MatchResult,
    corrections: dict[str, str | None],
) -> MatchResult:
    """
    Apply human corrections from the confirm gate.
    corrections: {data_name → shp_name or None (explicitly unmatched)}
    Moves items from low_confidence/unmatched into high_confidence (manual tier).
    Returns a new MatchResult.
    """
    corrected_names = set(corrections.keys())
    new_high = list(result.high_confidence)  # keep auto-accepted as-is
    new_low:  list[DistrictMatch] = []
    new_unmatched: list[str] = []

    for m in result.low_confidence:
        if m.data_name in corrected_names:
            chosen = corrections[m.data_name]
            if chosen:
                new_high.append(DistrictMatch(m.data_name, chosen, "manual", 1.0,
                                              "Confirmed by user"))
            # else: user said "skip this" → stays unmatched
        else:
            new_low.append(m)

    for name in result.unmatched:
        if name in corrected_names:
            chosen = corrections[name]
            if chosen:
                new_high.append(DistrictMatch(name, chosen, "manual", 1.0,
                                              "Assigned by user"))
        else:
            new_unmatched.append(name)

    return MatchResult(
        high_confidence=new_high,
        low_confidence=new_low,
        unmatched=new_unmatched,
    )


# ── File loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load .csv or .xlsx into a DataFrame."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx.")


def load_shapefile(path: str | Path) -> gpd.GeoDataFrame:
    """Load shapefile, GeoJSON, or zipped shapefile."""
    import zipfile, tempfile, os as _os
    path = Path(path)
    if path.suffix.lower() == ".zip":
        # Extract zip to a temp dir and find the .shp inside
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        shp_files = [f for f in Path(tmp).rglob("*.shp")]
        if not shp_files:
            # Try GeoJSON inside
            geojson_files = [f for f in Path(tmp).rglob("*.geojson")]
            if not geojson_files:
                raise ValueError("No .shp or .geojson found inside the zip.")
            read_path = geojson_files[0]
        else:
            read_path = shp_files[0]
        gdf = gpd.read_file(read_path)
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS. Set it to EPSG:4326 before using.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def detect_district_column(gdf: gpd.GeoDataFrame) -> str:
    """
    Heuristically identify the district name column in a shapefile.
    Checks a priority list of common column names, then falls back to the
    first object-typed column with highest cardinality.
    """
    priority = [
        "district", "DISTRICT", "dist_name", "DIST_NAME", "dtname", "DTNAME",
        "NAME_2", "NAME_3", "NAME_3_EN", "GEO_NAME", "admin2Name", "shapeName",
    ]
    for col in priority:
        if col in gdf.columns:
            return col

    # Fallback: pick the object column with the most unique values (likely district names)
    str_cols = [c for c in gdf.columns if gdf[c].dtype == object and c != "geometry"]
    if not str_cols:
        raise ValueError(
            "Could not auto-detect district name column. "
            "Please specify it manually."
        )
    return max(str_cols, key=lambda c: gdf[c].nunique())


# ── Geo join ──────────────────────────────────────────────────────────────────

def join_to_geo(
    gdf:          gpd.GeoDataFrame,
    data_df:      pd.DataFrame,
    data_name_col: str,
    shp_name_col:  str,
    name_map:      dict[str, str | None],  # data_name → shp_name
    value_col:     str,
) -> gpd.GeoDataFrame:
    """
    Left-join user data onto GeoDataFrame so every shapefile polygon
    is preserved. Districts with no data get NaN (rendered distinctly).

    Returns a GeoDataFrame with an added column '_value'.
    """
    # Build shp_name → value lookup from user data
    df = data_df.copy()
    df["__shp_name"] = df[data_name_col].map(name_map)

    # Keep only rows with a valid mapping
    df_matched = df.dropna(subset=["__shp_name"])

    # Aggregate duplicates (same shapefile district) — take mean
    agg = df_matched.groupby("__shp_name")[value_col].mean().reset_index()
    agg.columns = ["__shp_name", "_value"]

    # Left join: all geo rows kept, unmatched get NaN _value
    merged = gdf.merge(agg, left_on=shp_name_col, right_on="__shp_name", how="left")
    merged.drop(columns=["__shp_name"], errors="ignore", inplace=True)

    return merged


# ── Static map rendering (matplotlib) ────────────────────────────────────────

def render_static(
    gdf:            gpd.GeoDataFrame,
    value_col:      str          = "_value",
    scheme:         str          = "quantiles",
    cmap_name:      str          = "Blues",
    n_classes:      int          = 5,
    title:          str          = "",
    source:         str          = "",
    boundary_year:  str          = "",
    figsize:        tuple        = (14, 11),
) -> plt.Figure:
    """
    Render a choropleth map.

    Missing-data districts are shown in MISSING_COLOR with hatching,
    never as zero. Legend, title, and source footnote are mandatory.
    """
    scheme_map = {
        "quantiles":      "quantiles",
        "equal_interval": "equal_interval",
        "fisher_jenks":   "fisher_jenks",
    }
    mpl_scheme = scheme_map.get(scheme, "quantiles")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("white")

    has_data   = gdf[value_col].notna()
    gdf_data   = gdf[has_data]
    gdf_nodata = gdf[~has_data]

    if gdf_data.empty:
        ax.text(0.5, 0.5, "No data to display",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        return fig

    # ── Plot districts with data ──────────────────────────────────────────
    gdf_data.plot(
        column=value_col,
        scheme=mpl_scheme,
        k=n_classes,
        cmap=cmap_name,
        linewidth=0.4,
        edgecolor="#555555",
        ax=ax,
        legend=True,
        legend_kwds={
            "loc":      "lower right",
            "fontsize": 9,
            "title":    value_col.replace("_", " ").title(),
            "title_fontsize": 9,
        },
    )

    # ── Plot no-data districts with hatching ──────────────────────────────
    if not gdf_nodata.empty:
        gdf_nodata.plot(
            ax=ax,
            color=MISSING_COLOR,
            linewidth=0.4,
            edgecolor="#888888",
            hatch=MISSING_HATCH,
            zorder=2,
        )
        no_data_patch = mpatches.Patch(
            facecolor=MISSING_COLOR,
            hatch=MISSING_HATCH,
            edgecolor="#888888",
            label=NO_DATA_LABEL,
        )
        existing_handles, existing_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=existing_handles + [no_data_patch],
            labels=existing_labels + [NO_DATA_LABEL],
            loc="lower left",
            fontsize=8,
        )

    # ── Titles and footnote ───────────────────────────────────────────────
    ax.set_title(title or value_col.replace("_", " ").title(),
                 fontsize=15, fontweight="bold", pad=14)
    ax.axis("off")

    footnote_parts = []
    if source:
        footnote_parts.append(f"Source: {source}")
    if boundary_year:
        footnote_parts.append(f"Boundary vintage: {boundary_year}")
    footnote_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    footnote = "  |  ".join(footnote_parts)
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom",
             fontsize=8, color="#666666", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── Export: static formats ────────────────────────────────────────────────────

def export_png(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format="png")
    return path


def export_jpg(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format="jpeg",
                pil_kwargs={"quality": 92})
    return path


def export_svg(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    fig.savefig(path, bbox_inches="tight", format="svg")
    return path


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 200) -> bytes:
    """Return figure as bytes without touching the filesystem (for Streamlit)."""
    buf = io.BytesIO()
    if fmt == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
    elif fmt in ("jpg", "jpeg"):
        fig.savefig(buf, format="jpeg", dpi=dpi, bbox_inches="tight",
                    pil_kwargs={"quality": 92})
    else:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ── Interactive map rendering (Folium) ────────────────────────────────────────

def render_interactive(
    gdf:           gpd.GeoDataFrame,
    value_col:     str  = "_value",
    label_col:     str  = "district",
    title:         str  = "",
    cmap_name:     str  = "Blues",
    n_classes:     int  = 5,
    scheme:        str  = "quantiles",
    icon_label:    str  = "",
) -> folium.Map:
    """
    Build a Folium choropleth with hover tooltips and zoom.
    Missing-data districts are styled separately (gray, no choropleth class).
    """
    # Compute map center
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    # ── Classify values ───────────────────────────────────────────────────
    has_data   = gdf[value_col].notna()
    gdf_data   = gdf[has_data].copy()
    gdf_nodata = gdf[~has_data].copy()

    if not gdf_data.empty:
        classifier_cls = {
            "quantiles":      mapclassify.Quantiles,
            "equal_interval": mapclassify.EqualInterval,
            "fisher_jenks":   mapclassify.FisherJenks,
        }.get(scheme, mapclassify.Quantiles)

        k = min(n_classes, len(gdf_data[value_col].unique()))
        classifier = classifier_cls(gdf_data[value_col].dropna(), k=k)
        bins = [gdf_data[value_col].min()] + list(classifier.bins)

        # Build colormap
        cmap_obj = plt.get_cmap(cmap_name)
        colors = [mcolors.to_hex(cmap_obj(i / (len(bins) - 1)))
                  for i in range(len(bins))]

        # ── Choropleth layer ──────────────────────────────────────────────
        gdf_data["__id"] = gdf_data.index.astype(str)
        gdf_data["__val"] = gdf_data[value_col]
        geo_json_data = json.loads(gdf_data.to_json())

        choropleth = folium.Choropleth(
            geo_data=geo_json_data,
            data=gdf_data.set_index("__id")["__val"],
            columns=["__id", "__val"],
            key_on="feature.id",
            fill_color=cmap_name,
            fill_opacity=0.75,
            line_opacity=0.4,
            line_color="#555",
            nan_fill_color=MISSING_COLOR,
            bins=bins,
            legend_name=f"{icon_label} {title or value_col}".strip(),
            name="Choropleth",
        ).add_to(m)

        # ── Hover tooltips on data districts ──────────────────────────────
        tooltip_cols = [c for c in [label_col, value_col] if c in gdf_data.columns]
        tooltip_aliases = {
            label_col:  "District",
            value_col:  icon_label + " " + (title or value_col),
        }
        folium.GeoJson(
            geo_json_data,
            style_function=lambda f: {
                "fillOpacity": 0,
                "weight": 0,
            },
            tooltip=GeoJsonTooltip(
                fields=tooltip_cols,
                aliases=[tooltip_aliases.get(c, c) for c in tooltip_cols],
                localize=True,
                sticky=True,
            ),
            name="Tooltips",
        ).add_to(m)

    # ── No-data layer ─────────────────────────────────────────────────────
    if not gdf_nodata.empty:
        gdf_nodata["__label"] = gdf_nodata.get(label_col, "Unknown")
        folium.GeoJson(
            json.loads(gdf_nodata.to_json()),
            style_function=lambda f: {
                "fillColor": MISSING_COLOR,
                "fillOpacity": 0.6,
                "color": "#888",
                "weight": 0.5,
                "dashArray": "4 4",
            },
            tooltip=GeoJsonTooltip(
                fields=[label_col] if label_col in gdf_nodata.columns else [],
                aliases=["District"],
                localize=True,
                sticky=True,
            ),
            name="No data",
        ).add_to(m)

    # ── Title overlay ─────────────────────────────────────────────────────
    if title:
        title_html = f"""
        <div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
                    background:rgba(255,255,255,0.9);padding:8px 18px;
                    border-radius:6px;font-size:15px;font-weight:bold;
                    box-shadow:0 2px 6px rgba(0,0,0,0.2);z-index:9999;">
            {title}
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl().add_to(m)
    return m


def export_html(m: folium.Map, path: str | Path) -> Path:
    path = Path(path)
    m.save(str(path))
    return path


def map_to_html_bytes(m: folium.Map) -> bytes:
    """Return folium map as HTML bytes (for Streamlit download button)."""
    buf = io.BytesIO()
    html_str = m._repr_html_()
    # get_root().render() gives a full self-contained page
    html_full = m.get_root().render()
    buf.write(html_full.encode("utf-8"))
    buf.seek(0)
    return buf.read()


# ── GeoJSON export ────────────────────────────────────────────────────────────

def export_geojson(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Export the joined GeoDataFrame (with _value) as GeoJSON."""
    path = Path(path)
    gdf.to_file(path, driver="GeoJSON")
    return path


def geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    buf = io.BytesIO()
    buf.write(gdf.to_json().encode("utf-8"))
    buf.seek(0)
    return buf.read()


# ── Standalone script generator ───────────────────────────────────────────────

def generate_script(
    shapefile_path:   str,
    shp_name_col:     str,
    data_records:     dict[str, float | None],  # {district_name → value or None}
    value_col_label:  str,
    scheme:           str,
    cmap_name:        str,
    n_classes:        int,
    title:            str,
    source:           str,
    boundary_year:    str,
) -> str:
    """
    Generate a standalone Python script that reproduces the exact map.
    Data is embedded inline — no external data file dependency.
    User only needs to supply the shapefile at the same path.
    """
    # json.dumps produces multi-line output whose continuation lines have 0
    # leading spaces.  textwrap.dedent can't strip the 8-space common prefix of
    # the f-string template if any line has fewer spaces.  Re-indent continuation
    # lines so they share the same 8-space prefix as the rest of the template.
    _raw_json   = json.dumps(data_records, indent=4, ensure_ascii=False)
    _json_lines = _raw_json.split("\n")
    data_repr   = _json_lines[0] + "\n" + "\n".join(
        "        " + ln for ln in _json_lines[1:]
    )
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-
        \"\"\"
        Auto-generated by District Map Generator
        =========================================
        Title          : {title}
        Data indicator : {value_col_label}
        Source         : {source}
        Boundary year  : {boundary_year}
        Generated      : {timestamp}

        To reproduce this map:
          1. Place this script in the same directory as your shapefile.
          2. Install dependencies:  pip install geopandas matplotlib mapclassify
          3. Run:  python reproduce_map.py
        \"\"\"

        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import geopandas as gpd
        import mapclassify
        import numpy as np
        from pathlib import Path
        from datetime import datetime

        # ── Configuration (baked in at export time) ────────────────────────
        SHAPEFILE_PATH  = {json.dumps(shapefile_path)}
        SHP_NAME_COL    = {json.dumps(shp_name_col)}
        VALUE_COL_LABEL = {json.dumps(value_col_label)}
        TITLE           = {json.dumps(title)}
        SOURCE          = {json.dumps(source)}
        BOUNDARY_YEAR   = {json.dumps(boundary_year)}
        SCHEME          = {json.dumps(scheme)}
        CMAP            = {json.dumps(cmap_name)}
        N_CLASSES       = {n_classes}
        MISSING_COLOR   = "#c8c8c8"
        MISSING_HATCH   = "///"

        # ── Embedded data (district → value, None = no data) ──────────────
        DATA = {data_repr}

        # ── Load shapefile ─────────────────────────────────────────────────
        gdf = gpd.read_file(SHAPEFILE_PATH)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # ── Join data ──────────────────────────────────────────────────────
        gdf["_value"] = gdf[SHP_NAME_COL].map(DATA)

        # ── Render ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))
        fig.patch.set_facecolor("white")

        has_data   = gdf["_value"].notna()
        gdf_data   = gdf[has_data]
        gdf_nodata = gdf[~has_data]

        gdf_data.plot(
            column="_value",
            scheme=SCHEME,
            k=N_CLASSES,
            cmap=CMAP,
            linewidth=0.4,
            edgecolor="#555555",
            ax=ax,
            legend=True,
            legend_kwds={{
                "loc": "lower right",
                "fontsize": 9,
                "title": VALUE_COL_LABEL,
                "title_fontsize": 9,
            }},
        )

        if not gdf_nodata.empty:
            gdf_nodata.plot(
                ax=ax,
                color=MISSING_COLOR,
                linewidth=0.4,
                edgecolor="#888888",
                hatch=MISSING_HATCH,
                zorder=2,
            )
            no_data_patch = mpatches.Patch(
                facecolor=MISSING_COLOR,
                hatch=MISSING_HATCH,
                edgecolor="#888888",
                label="No data",
            )
            ax.legend(handles=[no_data_patch], loc="lower left", fontsize=8)

        ax.set_title(TITLE, fontsize=15, fontweight="bold", pad=14)
        ax.axis("off")

        footnote = f"Source: {{SOURCE}}  |  Boundary vintage: {{BOUNDARY_YEAR}}  |  Generated: {{datetime.now().strftime('%Y-%m-%d')}}"
        fig.text(0.5, 0.01, footnote, ha="center", va="bottom",
                 fontsize=8, color="#666666", style="italic")

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig("map_output.png", dpi=200, bbox_inches="tight")
        print("Saved map_output.png")
    """)
    return script
