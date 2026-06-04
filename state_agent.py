"""
state_agent.py — Agent 2: Level Detection + State Identification
================================================================
PRIMARY job: answer two questions about the user's data:

  Q1. What LEVEL is this data?
        "state"    — one row per Indian state/UT  (all-India state map)
        "district" — one row per district         (state or all-India district map)

  Q2. Which STATE(S) does the data cover?  (only relevant for district-level data)
        single     — crop the shapefile to that state's districts
        multi      — keep full all-India district shapefile
        all_india  — data explicitly covers all states

Detection strategy
------------------
  1. Try matching geo names against STATE_UT column  → state-level score
  2. Try matching geo names against DISTRICT column  → district-level score
  3. Whichever column yields higher match rate wins  → sets level
  4. For district-level: majority-vote STATE_UT of matched rows → which state(s)
  5. Groq fallback only when confidence is below threshold

Public API
----------
  run(geo_names, gdf) → StateAgentResult
  filter_by_state(gdf, state, state_col) → gpd.GeoDataFrame
  resolve_ambiguous_state(geo_names, candidates, api_key) → str | None
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process as fuzz_process

from map_engine import normalize, _ALIAS_MAP, FUZZY_CUTOFF

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

LEVEL_CONFIDENCE_MIN = 0.40   # need at least 40% match rate to trust level detection
AMBIGUITY_THRESHOLD  = 0.60   # if top state < 60% of district matches → multi-state
MIN_STATE_MATCH_FRAC = 0.50   # need 50% of names to match something for reliable result


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StateAgentResult:
    # Level
    level: str                      # "state" | "district" | "unknown"
    level_confidence: float         # 0–1, how sure we are about the level
    level_reasoning: str            # human-readable explanation

    # State identification (district-level only)
    state: Optional[str]            # dominant state (None if multi-state or unknown)
    state_confidence: float         # fraction of matched rows from top state
    state_counts: dict[str, int]    # {state_name: hit_count}
    matched: int                    # names that matched something in shapefile
    total: int                      # total input names
    multi_state: bool               # True if data spans multiple states
    all_states: list[str]           # all detected states, sorted by hit count

    # Coverage
    coverage: str                   # "single_state" | "multi_state" | "all_india" | "unknown"

    def summary(self) -> str:
        lines = [f"Level: {self.level} ({self.level_confidence:.0%} confidence)"]
        if self.level == "district":
            if self.multi_state:
                lines.append(f"Coverage: multi-state — {', '.join(self.all_states)}")
            elif self.state:
                lines.append(
                    f"State: {self.state} "
                    f"({self.matched}/{self.total} names matched, "
                    f"{self.state_confidence:.0%} within-state confidence)"
                )
            else:
                lines.append("State: could not determine.")
        elif self.level == "state":
            lines.append(f"Coverage: all-India state map ({self.matched}/{self.total} state names matched)")
        return " | ".join(lines)


# ── Internal matching helpers ─────────────────────────────────────────────────

def _match_rate(geo_names: list[str], targets: list[str], cutoff: float) -> tuple[int, dict[str, int]]:
    """
    Match geo_names against targets using tiers 1-3 (exact, alias, fuzzy).
    Returns (n_matched, {target: count}).
    """
    norm_targets = {normalize(t): t for t in targets}
    target_counts: dict[str, int] = {}
    matched = 0

    for name in geo_names:
        name = str(name).strip()
        n = normalize(name)
        hit: Optional[str] = None

        # Tier 1 — exact (normalized)
        if n in norm_targets:
            hit = norm_targets[n]

        # Tier 2 — alias
        if hit is None:
            aliased = _ALIAS_MAP.get(n)
            if aliased and aliased in norm_targets:
                hit = norm_targets[aliased]

        # Tier 3 — fuzzy
        if hit is None:
            best = fuzz_process.extractOne(
                n, list(norm_targets.keys()),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=cutoff * 100,
            )
            if best:
                hit = norm_targets[best[0]]

        if hit:
            target_counts[hit] = target_counts.get(hit, 0) + 1
            matched += 1

    return matched, target_counts


def _match_with_state(
    geo_names: list[str],
    gdf,
    district_col: str,
    state_col: str,
    cutoff: float,
) -> tuple[int, dict[str, int]]:
    """
    Match geo_names against district column, tally STATE_UT for each hit.
    Returns (n_matched, {state_name: hit_count}).
    """
    # Build norm_district → state lookup
    norm_to_state: dict[str, str] = {}
    norm_list: list[str] = []
    for _, row in gdf.iterrows():
        n = normalize(str(row[district_col]))
        if n not in norm_to_state:
            norm_to_state[n] = str(row[state_col])
            norm_list.append(n)

    state_counts: dict[str, int] = {}
    matched = 0

    for name in geo_names:
        name = str(name).strip()
        n = normalize(name)
        hit_state: Optional[str] = None

        if n in norm_to_state:
            hit_state = norm_to_state[n]
        else:
            aliased = _ALIAS_MAP.get(n)
            if aliased and aliased in norm_to_state:
                hit_state = norm_to_state[aliased]
        if hit_state is None:
            best = fuzz_process.extractOne(
                n, norm_list,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=cutoff * 100,
            )
            if best:
                hit_state = norm_to_state[best[0]]

        if hit_state:
            state_counts[hit_state] = state_counts.get(hit_state, 0) + 1
            matched += 1

    return matched, state_counts


# ── Main agent entry point ────────────────────────────────────────────────────

def run(
    geo_names:    list[str],
    gdf,                          # gpd.GeoDataFrame — full all-India shapefile
    district_col: str  = "DISTRICT",
    state_col:    str  = "STATE_UT",
    fuzzy_cutoff: float = FUZZY_CUTOFF,
) -> StateAgentResult:
    """
    Agent 2 — authoritative level + state detection.

    Step 1: try to match geo_names against STATE_UT  → state_match_rate
    Step 2: try to match geo_names against DISTRICT   → district_match_rate
    Step 3: whichever is higher → level
    Step 4: if district-level, majority-vote STATE_UT → coverage/crop info
    """
    total = len(geo_names)
    if total == 0:
        return _unknown(total)

    state_vals    = gdf[state_col].dropna().unique().tolist()    if state_col    in gdf.columns else []
    district_vals = gdf[district_col].dropna().unique().tolist() if district_col in gdf.columns else []

    # ── Step 1: state-level match rate ───────────────────────────────────
    state_matched, state_hit_counts = _match_rate(geo_names, state_vals, fuzzy_cutoff)
    state_rate = state_matched / total

    # ── Step 2: district-level match rate ────────────────────────────────
    district_matched, district_state_counts = _match_with_state(
        geo_names, gdf, district_col, state_col, fuzzy_cutoff
    )
    district_rate = district_matched / total

    # ── Step 3: level decision ────────────────────────────────────────────
    both_low = state_rate < LEVEL_CONFIDENCE_MIN and district_rate < LEVEL_CONFIDENCE_MIN

    if both_low:
        # Neither column matched well — unknown
        return _unknown(total)

    if state_rate >= district_rate:
        level = "state"
        level_confidence = state_rate
        level_reasoning = (
            f"{state_matched}/{total} names matched state/UT names "
            f"({state_rate:.0%}) vs {district_matched}/{total} district names "
            f"({district_rate:.0%}). Treating as state-level data."
        )
        # For state-level: coverage = all_india, no cropping needed
        all_states = sorted(state_hit_counts.keys(), key=lambda s: -state_hit_counts[s])
        return StateAgentResult(
            level=level,
            level_confidence=level_confidence,
            level_reasoning=level_reasoning,
            state=None,
            state_confidence=1.0,
            state_counts=state_hit_counts,
            matched=state_matched,
            total=total,
            multi_state=False,
            all_states=all_states,
            coverage="all_india",
        )

    else:
        level = "district"
        level_confidence = district_rate
        level_reasoning = (
            f"{district_matched}/{total} names matched district names "
            f"({district_rate:.0%}) vs {state_matched}/{total} state names "
            f"({state_rate:.0%}). Treating as district-level data."
        )

    # ── Step 4: which state(s)? ───────────────────────────────────────────
    if not district_state_counts:
        return StateAgentResult(
            level=level, level_confidence=level_confidence,
            level_reasoning=level_reasoning,
            state=None, state_confidence=0.0,
            state_counts={}, matched=district_matched,
            total=total, multi_state=False, all_states=[],
            coverage="unknown",
        )

    top_state  = max(district_state_counts, key=district_state_counts.get)
    top_count  = district_state_counts[top_state]
    all_states = sorted(district_state_counts.keys(), key=lambda s: -district_state_counts[s])
    state_confidence = top_count / district_matched if district_matched else 0.0

    second_count = sorted(district_state_counts.values(), reverse=True)[1] \
                   if len(district_state_counts) > 1 else 0
    multi_state = (second_count / district_matched) > (1 - AMBIGUITY_THRESHOLD) \
                  if district_matched else False

    # Coverage
    n_states = len([s for s in district_state_counts if district_state_counts[s] >= 2])
    if multi_state and n_states >= 5:
        coverage = "all_india"
    elif multi_state:
        coverage = "multi_state"
    else:
        coverage = "single_state"

    return StateAgentResult(
        level=level,
        level_confidence=level_confidence,
        level_reasoning=level_reasoning,
        state=top_state if not multi_state else None,
        state_confidence=state_confidence,
        state_counts=district_state_counts,
        matched=district_matched,
        total=total,
        multi_state=multi_state,
        all_states=all_states,
        coverage=coverage,
    )


def _unknown(total: int) -> StateAgentResult:
    return StateAgentResult(
        level="unknown", level_confidence=0.0,
        level_reasoning="Could not match geo names against either state or district columns.",
        state=None, state_confidence=0.0, state_counts={},
        matched=0, total=total, multi_state=False, all_states=[],
        coverage="unknown",
    )


# ── Crop helper ───────────────────────────────────────────────────────────────

def filter_by_state(gdf, state: str, state_col: str = "STATE_UT"):
    """Return GDF rows where state_col == state."""
    if state_col not in gdf.columns:
        raise ValueError(f"Column '{state_col}' not found in shapefile.")
    filtered = gdf[gdf[state_col] == state].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for '{state}' in column '{state_col}'.")
    return filtered


# ── Groq fallback ─────────────────────────────────────────────────────────────

def resolve_ambiguous_state(
    geo_names:  list[str],
    candidates: list[str],
    api_key:    Optional[str] = None,
    model:      str = "llama3-8b-8192",
) -> Optional[str]:
    """LLM fallback when state detection is ambiguous."""
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        from groq import Groq
    except ImportError:
        return None

    names_block = ", ".join(f'"{n}"' for n in geo_names[:30])
    cands_block = ", ".join(f'"{c}"' for c in candidates)
    prompt = (
        f"These district names are from an Indian researcher's data file:\n[{names_block}]\n\n"
        f"Which Indian state do they most likely belong to?\n"
        f"Choose ONLY from: [{cands_block}]\n"
        f"Respond with ONLY the state name, nothing else."
    )
    try:
        client = Groq(api_key=key)
        resp   = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise Indian geography expert."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0, max_tokens=20,
        )
        answer = resp.choices[0].message.content.strip().strip('"')
        if answer in candidates:
            return answer
        hit = fuzz_process.extractOne(answer, candidates, scorer=fuzz.ratio, score_cutoff=80)
        return hit[0] if hit else None
    except Exception as exc:
        logger.error(f"Groq state resolution failed: {exc}")
        return None


# ── Backwards-compat shim (old detect_state() callers in app.py) ──────────────

@dataclass
class StateDetectionResult:
    """Thin wrapper kept for backward compatibility with app.py step 3."""
    state:        Optional[str]
    confidence:   float
    state_counts: dict[str, int]
    matched:      int
    total:        int
    multi_state:  bool
    all_states:   list[str]

    def summary(self) -> str:
        if self.multi_state:
            return f"Multi-state: {', '.join(self.all_states)}"
        if self.state:
            return f"{self.state} ({self.matched}/{self.total}, {self.confidence:.0%})"
        return "State unknown."


def detect_state(
    data_names:   list[str],
    gdf,
    shp_name_col: str,
    state_col:    str = "STATE_UT",
    fuzzy_cutoff: float = FUZZY_CUTOFF,
) -> StateDetectionResult:
    """Backward-compat wrapper — calls run() and converts to StateDetectionResult."""
    result = run(
        geo_names=data_names,
        gdf=gdf,
        district_col=shp_name_col,
        state_col=state_col,
        fuzzy_cutoff=fuzzy_cutoff,
    )
    return StateDetectionResult(
        state=result.state,
        confidence=result.state_confidence,
        state_counts=result.state_counts,
        matched=result.matched,
        total=result.total,
        multi_state=result.multi_state,
        all_states=result.all_states,
    )
