"""
state_agent.py — State detection + shapefile crop agent
========================================================
Detects which Indian state(s) a user's district data covers, then
filters the all-India GeoDataFrame to only those districts.

Detection is deterministic (majority vote across tiers 1-4 matching).
Groq is called only when the top state has < AMBIGUITY_THRESHOLD of
matches AND at least 2 states have meaningful representation.

Public API
----------
detect_state(data_names, gdf, shp_name_col, state_col)  → StateDetectionResult
filter_by_state(gdf, state, state_col)                  → gpd.GeoDataFrame
resolve_ambiguous_state(data_names, candidates, api_key) → str | None
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process as fuzz_process

from map_engine import normalize, _ALIAS_MAP, FUZZY_CUTOFF

logger = logging.getLogger(__name__)

AMBIGUITY_THRESHOLD = 0.60   # below this → call Groq to resolve
MIN_MATCH_FRACTION  = 0.50   # need at least 50% of names matched to trust result


@dataclass
class StateDetectionResult:
    state:        Optional[str]        # dominant state name (None if undetectable)
    confidence:   float                # fraction of matched districts from top state
    state_counts: dict[str, int]       # {state_name: district_count}
    matched:      int                  # districts matched to any state
    total:        int                  # total input district names
    multi_state:  bool                 # True if data clearly spans multiple states
    all_states:   list[str]            # sorted list of detected states (for UI picker)

    def summary(self) -> str:
        if not self.state:
            return "Could not detect state."
        pct = f"{self.confidence:.0%}"
        if self.multi_state:
            return f"Multi-state data: {', '.join(self.all_states)}"
        return f"Detected: {self.state} ({self.matched}/{self.total} districts matched, {pct} confidence)"


def detect_state(
    data_names:   list[str],
    gdf,                          # gpd.GeoDataFrame
    shp_name_col: str,
    state_col:    str = "STATE_UT",
    fuzzy_cutoff: float = FUZZY_CUTOFF,
) -> StateDetectionResult:
    """
    Identify which state(s) the user's district names belong to.

    Algorithm:
      For each data district name, run the same tier 1-4 matching against
      the shapefile district names. For each hit, record the STATE_UT of
      the matched shapefile row. The state with the most hits wins.
    """
    if state_col not in gdf.columns:
        logger.warning(f"State column '{state_col}' not in shapefile. Cannot detect state.")
        return StateDetectionResult(None, 0.0, {}, 0, len(data_names), False, [])

    # Build: norm_shp_name → (orig_shp_name, state)
    norm_to_state: dict[str, str] = {}
    norm_to_orig:  dict[str, str] = {}
    shp_norm_list: list[str] = []

    for _, row in gdf.iterrows():
        orig  = str(row[shp_name_col])
        state = str(row[state_col])
        n     = normalize(orig)
        if n not in norm_to_state:
            norm_to_state[n] = state
            norm_to_orig[n]  = orig
            shp_norm_list.append(n)

    state_counts: dict[str, int] = {}
    matched = 0

    for name in data_names:
        norm_name = normalize(name)
        hit_state: Optional[str] = None

        # Tier 1: exact
        if norm_name in norm_to_state:
            hit_state = norm_to_state[norm_name]

        # Tier 2: alias
        if hit_state is None:
            aliased = _ALIAS_MAP.get(norm_name)
            if aliased and aliased in norm_to_state:
                hit_state = norm_to_state[aliased]

        # Tier 3: fuzzy
        if hit_state is None:
            fuzzy_hit = fuzz_process.extractOne(
                norm_name,
                shp_norm_list,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=fuzzy_cutoff * 100,
            )
            if fuzzy_hit:
                matched_norm = fuzzy_hit[0]
                hit_state = norm_to_state[matched_norm]

        if hit_state:
            state_counts[hit_state] = state_counts.get(hit_state, 0) + 1
            matched += 1

    if not state_counts:
        return StateDetectionResult(None, 0.0, {}, 0, len(data_names), False, [])

    total       = len(data_names)
    top_state   = max(state_counts, key=state_counts.get)
    top_count   = state_counts[top_state]
    confidence  = top_count / matched if matched else 0.0
    all_states  = sorted(state_counts.keys(), key=lambda s: -state_counts[s])

    # Multi-state: second state has > 25% of matches
    second_count = sorted(state_counts.values(), reverse=True)[1] if len(state_counts) > 1 else 0
    multi_state  = second_count / matched > 0.25 if matched else False

    # Low match rate → unreliable
    if matched / total < MIN_MATCH_FRACTION:
        logger.warning(f"Only {matched}/{total} districts matched — state detection unreliable.")
        return StateDetectionResult(None, confidence, state_counts, matched, total, False, all_states)

    return StateDetectionResult(
        state       = top_state if not multi_state else None,
        confidence  = confidence,
        state_counts= state_counts,
        matched     = matched,
        total       = total,
        multi_state = multi_state,
        all_states  = all_states,
    )


def filter_by_state(
    gdf,
    state:     str,
    state_col: str = "STATE_UT",
):
    """Return GDF rows where state_col == state. Preserves CRS and index."""
    if state_col not in gdf.columns:
        raise ValueError(f"Column '{state_col}' not found in shapefile.")
    filtered = gdf[gdf[state_col] == state].copy()
    if filtered.empty:
        raise ValueError(f"No districts found for state '{state}' in column '{state_col}'.")
    return filtered


def resolve_ambiguous_state(
    data_names:  list[str],
    candidates:  list[str],          # state names to choose from
    api_key:     Optional[str] = None,
    model:       str = "llama3-8b-8192",
) -> Optional[str]:
    """
    Groq fallback: given ambiguous state detection, ask the LLM which
    Indian state these district names most likely belong to.
    Returns a state name from `candidates`, or None on failure.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    names_block = ", ".join(f'"{n}"' for n in data_names[:30])
    cands_block = ", ".join(f'"{c}"' for c in candidates)

    prompt = f"""\
These district names are from an Indian researcher's data file:
[{names_block}]

Based on these names, which Indian state do they most likely belong to?
Choose ONLY from this list: [{cands_block}]

Respond with ONLY the state name, nothing else. No explanation.
"""
    try:
        client   = Groq(api_key=key)
        response = client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": "You are a precise Indian geography expert."},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.0,
            max_tokens  = 20,
        )
        answer = response.choices[0].message.content.strip().strip('"')
        if answer in candidates:
            return answer
        # Fuzzy match answer against candidates (LLM may add punctuation)
        hit = fuzz_process.extractOne(answer, candidates, scorer=fuzz.ratio, score_cutoff=80)
        return hit[0] if hit else None
    except Exception as exc:
        logger.error(f"Groq state resolution failed: {exc}")
        return None
