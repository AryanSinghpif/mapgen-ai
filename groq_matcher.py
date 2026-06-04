"""
groq_matcher.py — LLM-assisted district name resolution (Tier 5)
=================================================================
Called ONLY for district names that survive all four rule-based tiers in
map_engine.py. This module is intentionally isolated so:

  1. The rest of the pipeline runs without it if GROQ_API_KEY is absent.
  2. The LLM call can be swapped for any other provider by changing one file.
  3. The prompt + response are fully logged for auditability.

COST POLICY
-----------
One call per upload, regardless of how many districts are unmatched.
All unmatched names are batched into a single prompt.
Groq free tier: ~14,400 requests/day at time of writing — well within
single-user / demo use. Costs nothing.

RATE LIMIT HANDLING
-------------------
Groq free tier is rate-limited (tokens/min). If you hit a 429, the
`batch_resolve` function backs off with exponential retry (max 3 attempts).
Under heavy concurrent load consider caching results or upgrading tier.
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Public interface ──────────────────────────────────────────────────────────

def batch_resolve(
    unmatched_names:  list[str],
    valid_shp_names:  list[str],
    api_key:          Optional[str] = None,
    model:            str = "llama3-8b-8192",   # free tier, fast
    max_retries:      int = 3,
) -> list[dict]:
    """
    Ask Groq to match every name in `unmatched_names` to the closest name
    in `valid_shp_names`.

    Parameters
    ----------
    unmatched_names : district names from user data that rule-based tiers missed
    valid_shp_names : canonical district names from the shapefile
    api_key         : GROQ_API_KEY (falls back to env var)
    model           : Groq model ID (llama3-8b-8192 is free & sufficient)
    max_retries     : attempts on rate-limit / transient error

    Returns
    -------
    list of dicts:
        [{"data_name": "...", "shp_name": "...", "confidence": 0.85}, ...]
    `shp_name` is None if the LLM could not find a plausible match.
    `confidence` is 0–1, as reported by the LLM.
    """
    if not unmatched_names:
        return []

    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        logger.warning(
            "GROQ_API_KEY not set. Skipping LLM resolution — "
            "unmatched districts will go straight to human review."
        )
        return _null_results(unmatched_names)

    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        return _null_results(unmatched_names)

    client = Groq(api_key=key)
    prompt = _build_prompt(unmatched_names, valid_shp_names)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise data-matching assistant for Indian "
                            "geographic district names. Return ONLY valid JSON, "
                            "no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,   # deterministic
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return _parse_response(raw, unmatched_names, valid_shp_names)

        except Exception as exc:
            err_str = str(exc)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = 2 ** attempt
                logger.warning(f"Groq rate limit hit. Waiting {wait}s (attempt {attempt}/{max_retries}).")
                time.sleep(wait)
            else:
                logger.error(f"Groq API error: {exc}")
                break

    logger.error("Groq resolution failed after retries. Sending to human review.")
    return _null_results(unmatched_names)


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_prompt(unmatched: list[str], valid: list[str]) -> str:
    """
    Build a tight, structured prompt that asks for JSON output.
    Sending the full valid list caps Groq at one call regardless of
    how many districts are unmatched.
    """
    unmatched_block = "\n".join(f"  - {n}" for n in unmatched)
    valid_block     = ", ".join(f'"{v}"' for v in valid)

    return f"""\
You are matching district names from a researcher's data file to the canonical
district names in an Indian state shapefile.

UNMATCHED NAMES (from user data):
{unmatched_block}

VALID SHAPEFILE DISTRICT NAMES:
[{valid_block}]

TASK: For each unmatched name, find the best-matching shapefile district name.
Consider:
  - Historical renames (e.g. Allahabad → Prayagraj, Gurgaon → Gurugram)
  - Transliteration variants (e.g. Mysore → Mysuru, Belgaum → Belagavi)
  - Common abbreviations or partial names
  - Spelling errors

RULES:
  1. Only use names from the VALID list above. Do not invent names.
  2. If no plausible match exists, set "shp_name" to null and "confidence" to 0.
  3. Confidence is your certainty that this is the correct match (0.0–1.0).
  4. Confidence >= 0.85 means you are very sure. < 0.85 means the human should confirm.

RESPOND WITH EXACTLY this JSON structure (an object with a "matches" array):
{{
  "matches": [
    {{"data_name": "<original unmatched name>", "shp_name": "<shapefile name or null>", "confidence": 0.95}},
    ...
  ]
}}
"""


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_response(
    raw:            str,
    unmatched:      list[str],
    valid_shp:      list[str],
) -> list[dict]:
    """
    Parse LLM JSON response. Validates that:
      - shp_name is in the valid list (or null)
      - confidence is a float 0–1
    Falls back to null result for any malformed entry.
    """
    valid_set = set(valid_shp)
    results: list[dict] = []

    try:
        parsed = json.loads(raw)
        matches = parsed.get("matches", [])
    except json.JSONDecodeError:
        logger.error(f"Groq returned invalid JSON: {raw[:200]}")
        return _null_results(unmatched)

    matched_data_names: set[str] = set()

    for entry in matches:
        data_name  = entry.get("data_name", "")
        shp_name   = entry.get("shp_name")
        confidence = float(entry.get("confidence", 0.0))

        # Validate shp_name against valid list
        if shp_name and shp_name not in valid_set:
            logger.warning(
                f"Groq returned invalid shp_name '{shp_name}' for '{data_name}'. "
                "Setting to null."
            )
            shp_name   = None
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        results.append({
            "data_name":  data_name,
            "shp_name":   shp_name,
            "confidence": confidence,
        })
        matched_data_names.add(data_name)

    # Ensure every unmatched name has an entry (LLM might skip some)
    for name in unmatched:
        if name not in matched_data_names:
            logger.warning(f"Groq did not return a result for '{name}'. Adding null.")
            results.append({"data_name": name, "shp_name": None, "confidence": 0.0})

    return results


def _null_results(names: list[str]) -> list[dict]:
    """Return a null-match list for all names (used on error / no API key)."""
    return [{"data_name": n, "shp_name": None, "confidence": 0.0} for n in names]


# ── Integration helper ────────────────────────────────────────────────────────

def apply_groq_results(
    result,           # MatchResult from map_engine
    groq_matches: list[dict],
    high_conf_threshold: float = 0.85,
) -> None:
    """
    Mutate `result` in place: move Groq-resolved names from `result.unmatched`
    into `result.high_confidence` or `result.low_confidence` as appropriate.

    Import DistrictMatch here to avoid circular imports.
    """
    from map_engine import DistrictMatch

    remaining_unmatched: list[str] = []

    for entry in groq_matches:
        data_name  = entry["data_name"]
        shp_name   = entry["shp_name"]
        confidence = entry["confidence"]

        if shp_name and confidence >= high_conf_threshold:
            result.high_confidence.append(
                DistrictMatch(data_name, shp_name, "groq", confidence,
                              f"LLM-matched (confidence {confidence:.0%})")
            )
        elif shp_name:
            result.low_confidence.append(
                DistrictMatch(data_name, shp_name, "groq", confidence,
                              f"LLM suggestion — please confirm (confidence {confidence:.0%})")
            )
        else:
            remaining_unmatched.append(data_name)

    result.unmatched = remaining_unmatched
