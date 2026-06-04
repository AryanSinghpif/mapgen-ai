"""
flows.py — CrewAI Flow orchestration (headless / CLI mode)
===========================================================
This module implements the full pipeline as a CrewAI Flow so the tool can be
run non-interactively (scripts, cron jobs, CI pipelines) without the Streamlit UI.

The human-confirmation gate uses CLI prompts when run directly.
The Streamlit app (app.py) implements the same logical steps via session state.

USAGE (CLI)
-----------
    python flows.py \\
        --data     sample_data/karnataka_sample.csv \\
        --shapefile shapefiles/karnataka.geojson \\
        --state     Karnataka \\
        --col       literacy_rate_2011 \\
        --title    "Literacy Rate by District, Karnataka (2011)" \\
        --source   "Census of India 2011" \\
        --year      2011 \\
        --scheme    quantiles \\
        --cmap      Blues \\
        --out       output/

FLOW STEPS
----------
  1. inspect_file      — load data, detect columns, validate inputs
  2. match_districts   — tiered matching (tiers 1-4 via map_engine)
  3. groq_resolve      — Groq API for leftovers (tier 5)
  4. human_confirm     — CLI gate for low-confidence + unmatched (tier 6)
  5. render_map        — generate static + interactive outputs
  6. export_all        — write all formats + standalone .py script
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Attempt CrewAI import (optional — falls back to plain class if absent) ────
try:
    from crewai.flow.flow import Flow, listen, start
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False
    logger.warning(
        "crewai not installed — running in plain-Python fallback mode. "
        "Install with:  pip install crewai"
    )
    # Minimal stubs so the rest of the file parses without crewai
    def start():
        return lambda f: f
    def listen(*args):
        return lambda f: f
    class Flow:
        def kickoff(self):
            pass

from map_engine import (
    MatchResult,
    DistrictMatch,
    load_data,
    load_shapefile,
    detect_district_column,
    match_districts,
    apply_manual_corrections,
    join_to_geo,
    render_static,
    render_interactive,
    export_png,
    export_jpg,
    export_svg,
    export_html,
    export_geojson,
    generate_script,
    HIGH_CONF_THRESHOLD,
)
from groq_matcher import batch_resolve, apply_groq_results


# ── Flow state ────────────────────────────────────────────────────────────────

@dataclass
class MapFlowState:
    # Inputs
    data_path:      str = ""
    shapefile_path: str = ""
    data_name_col:  str = ""   # district column in user data
    value_col:      str = ""
    shp_name_col:   str = ""   # district column in shapefile
    title:          str = ""
    source:         str = ""
    boundary_year:  str = ""
    scheme:         str = "quantiles"
    cmap_name:      str = "Blues"
    n_classes:      int = 5
    output_dir:     str = "output"
    groq_api_key:   str = ""

    # Computed mid-flow
    df:             object = None   # pd.DataFrame
    gdf:            object = None   # gpd.GeoDataFrame
    match_result:   object = None   # MatchResult
    merged_gdf:     object = None   # gpd.GeoDataFrame with _value
    fig:            object = None   # matplotlib Figure
    folium_map:     object = None   # folium.Map
    exports:        dict   = field(default_factory=dict)  # format → Path


# ── The Flow ──────────────────────────────────────────────────────────────────

class MapGeneratorFlow(Flow if _CREWAI_AVAILABLE else object):
    """
    CrewAI Flow implementing the District Map Generator pipeline.
    Each @listen step receives the output of its predecessor.
    Human-confirm gate (step 4) pauses and prompts the user via stdin.
    """

    def __init__(self, state: MapFlowState):
        if _CREWAI_AVAILABLE:
            super().__init__()
        self.state = state

    # ── Step 1 ────────────────────────────────────────────────────────────────

    @start()
    def inspect_file(self) -> str:
        logger.info("─── Step 1: Inspecting data file ───")
        s = self.state

        s.df  = load_data(s.data_path)
        s.gdf = load_shapefile(s.shapefile_path)

        if not s.shp_name_col:
            s.shp_name_col = detect_district_column(s.gdf)
            logger.info(f"  Shapefile district column auto-detected: '{s.shp_name_col}'")

        logger.info(f"  Data file:     {s.data_path} ({len(s.df)} rows, {len(s.df.columns)} cols)")
        logger.info(f"  Shapefile:     {s.shapefile_path} ({len(s.gdf)} districts)")
        logger.info(f"  Shapefile CRS: {s.gdf.crs}")
        logger.info(f"  Value column:  '{s.value_col}'")

        # Validate value column exists
        if s.value_col not in s.df.columns:
            available = list(s.df.select_dtypes("number").columns)
            raise ValueError(
                f"Column '{s.value_col}' not found in data file.\n"
                f"Available numeric columns: {available}"
            )
        return "inspect_done"

    # ── Step 2 ────────────────────────────────────────────────────────────────

    @listen(inspect_file)
    def match_districts_step(self, _) -> str:
        logger.info("─── Step 2: Matching district names (tiers 1–4) ───")
        s = self.state

        data_names = s.df[s.data_name_col].astype(str).tolist()
        shp_names  = s.gdf[s.shp_name_col].astype(str).tolist()

        s.match_result = match_districts(data_names, shp_names)
        logger.info(f"  {s.match_result.summary()}")

        for m in s.match_result.high_confidence:
            logger.info(f"  ✓ [{m.tier:10s}] '{m.data_name}' → '{m.shp_name}'  ({m.confidence:.0%})")

        return "match_done"

    # ── Step 3 ────────────────────────────────────────────────────────────────

    @listen(match_districts_step)
    def groq_resolve_step(self, _) -> str:
        logger.info("─── Step 3: Groq resolution for leftovers (tier 5) ───")
        s = self.state
        result = s.match_result

        needs_groq = result.unmatched + [m.data_name for m in result.low_confidence]

        if not needs_groq:
            logger.info("  Nothing left for Groq — all districts matched by rules.")
            return "groq_done"

        shp_names = s.gdf[s.shp_name_col].astype(str).tolist()
        groq_matches = batch_resolve(
            needs_groq,
            shp_names,
            api_key=s.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
        )

        apply_groq_results(result, groq_matches, high_conf_threshold=HIGH_CONF_THRESHOLD)
        logger.info(f"  After Groq: {result.summary()}")
        return "groq_done"

    # ── Step 4: Human-confirm gate ────────────────────────────────────────────

    @listen(groq_resolve_step)
    def human_confirm_step(self, _) -> str:
        logger.info("─── Step 4: Human confirmation gate ───")
        s      = self.state
        result = s.match_result

        needs_review = result.low_confidence
        unmatched    = result.unmatched
        shp_names    = s.gdf[s.shp_name_col].astype(str).tolist()

        if not needs_review and not unmatched:
            logger.info("  All matches high-confidence. No human review needed.")
            return "confirm_done"

        print("\n" + "="*60)
        print("HUMAN REVIEW REQUIRED")
        print("="*60)

        corrections: dict[str, str | None] = {}

        # ── Low-confidence matches ────────────────────────────────────────
        if needs_review:
            print(f"\n{len(needs_review)} low-confidence match(es) to review:")
            print("  Press ENTER to accept the suggestion, or type a corrected name.\n")

            for m in needs_review:
                print(f"  Data name   : {m.data_name}")
                print(f"  Suggestion  : {m.shp_name}  (confidence {m.confidence:.0%}, tier: {m.tier})")
                print(f"  Note        : {m.note}")
                user_input = input("  Accept? [ENTER=yes / type correction / 'skip']: ").strip()

                if user_input == "" :
                    corrections[m.data_name] = m.shp_name  # accept suggestion
                elif user_input.lower() == "skip":
                    corrections[m.data_name] = None        # explicitly unmatched
                else:
                    # Validate against shp names
                    if user_input in shp_names:
                        corrections[m.data_name] = user_input
                    else:
                        print(f"  ⚠ '{user_input}' not in shapefile. Skipping.")
                        corrections[m.data_name] = None
                print()

        # ── Fully unmatched ───────────────────────────────────────────────
        if unmatched:
            print(f"\n{len(unmatched)} district(s) with NO match found:")
            print("  Type the correct shapefile district name, or 'skip' to omit.\n")

            for name in unmatched:
                print(f"  Data name: {name}")
                user_input = input("  Map to shapefile district [type name / 'skip']: ").strip()

                if user_input.lower() == "skip" or user_input == "":
                    corrections[name] = None
                elif user_input in shp_names:
                    corrections[name] = user_input
                else:
                    print(f"  ⚠ '{user_input}' not in shapefile. Omitting.")
                    corrections[name] = None
                print()

        s.match_result = apply_manual_corrections(result, corrections)

        still_unmatched = s.match_result.unmatched
        if still_unmatched:
            logger.warning(
                f"  {len(still_unmatched)} district(s) will be omitted from the map "
                f"(rendered as 'No data'): {still_unmatched}"
            )

        return "confirm_done"

    # ── Step 5 ────────────────────────────────────────────────────────────────

    @listen(human_confirm_step)
    def render_map_step(self, _) -> str:
        logger.info("─── Step 5: Rendering map ───")
        s = self.state

        name_map = s.match_result.as_dict()

        s.merged_gdf = join_to_geo(
            gdf           = s.gdf,
            data_df       = s.df,
            data_name_col = s.data_name_col,
            shp_name_col  = s.shp_name_col,
            name_map      = name_map,
            value_col     = s.value_col,
        )

        s.fig = render_static(
            gdf           = s.merged_gdf,
            value_col     = "_value",
            scheme        = s.scheme,
            cmap_name     = s.cmap_name,
            n_classes     = s.n_classes,
            title         = s.title,
            source        = s.source,
            boundary_year = s.boundary_year,
        )

        s.folium_map = render_interactive(
            gdf         = s.merged_gdf,
            value_col   = "_value",
            label_col   = s.shp_name_col,
            title       = s.title,
            cmap_name   = s.cmap_name,
            n_classes   = s.n_classes,
            scheme      = s.scheme,
        )

        logger.info("  Rendering complete.")
        return "render_done"

    # ── Step 6 ────────────────────────────────────────────────────────────────

    @listen(render_map_step)
    def export_all_step(self, _) -> str:
        logger.info("─── Step 6: Exporting all formats ───")
        s = self.state

        out = Path(s.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = "district_map"

        s.exports["png"]  = export_png(s.fig,        out / f"{stem}.png")
        s.exports["jpg"]  = export_jpg(s.fig,        out / f"{stem}.jpg")
        s.exports["svg"]  = export_svg(s.fig,        out / f"{stem}.svg")
        s.exports["html"] = export_html(s.folium_map, out / f"{stem}.html")
        s.exports["geojson"] = export_geojson(s.merged_gdf, out / f"{stem}.geojson")

        # Standalone reproducible Python script
        name_map  = s.match_result.as_dict()
        data_dict = {
            shp: float(
                s.df.loc[s.df[s.data_name_col] == data, s.value_col].values[0]
            ) if shp else None
            for data, shp in name_map.items()
        }
        script_code = generate_script(
            shapefile_path  = str(Path(s.shapefile_path).resolve()),
            shp_name_col    = s.shp_name_col,
            data_records    = data_dict,
            value_col_label = s.value_col,
            scheme          = s.scheme,
            cmap_name       = s.cmap_name,
            n_classes       = s.n_classes,
            title           = s.title,
            source          = s.source,
            boundary_year   = s.boundary_year,
        )
        script_path = out / f"{stem}_reproduce.py"
        script_path.write_text(script_code, encoding="utf-8")
        s.exports["py_script"] = script_path

        logger.info("  Exports saved:")
        for fmt, path in s.exports.items():
            logger.info(f"    {fmt:10s} → {path}")

        return "export_done"


# ── Convenience runner ────────────────────────────────────────────────────────

def run_flow(state: MapFlowState) -> MapFlowState:
    """
    Execute the full pipeline. Returns the state with all outputs populated.
    Works with or without CrewAI installed.
    """
    flow = MapGeneratorFlow(state)

    if _CREWAI_AVAILABLE:
        flow.kickoff()
    else:
        # Plain-Python fallback: call steps manually in order
        logger.info("Running in plain-Python mode (crewai not installed).")
        flow.inspect_file()
        flow.match_districts_step(None)
        flow.groq_resolve_step(None)
        flow.human_confirm_step(None)
        flow.render_map_step(None)
        flow.export_all_step(None)

    return state


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="District Map Generator — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data",      required=True,  help="Path to .csv or .xlsx data file")
    p.add_argument("--shapefile", required=True,  help="Path to district shapefile or GeoJSON")
    p.add_argument("--col",       required=True,  help="Column name to map (value column)")
    p.add_argument("--name-col",  default="",     help="District name column in data (auto-detected if blank)")
    p.add_argument("--shp-col",   default="",     help="District name column in shapefile (auto-detected if blank)")
    p.add_argument("--title",     default="",     help="Map title")
    p.add_argument("--source",    default="",     help="Data source (for footnote)")
    p.add_argument("--year",      default="",     help="Boundary vintage year (for footnote)")
    p.add_argument("--scheme",    default="quantiles",
                   choices=["quantiles", "equal_interval", "fisher_jenks"],
                   help="Classification scheme")
    p.add_argument("--cmap",      default="Blues",
                   help="Matplotlib colormap (e.g. Blues, Reds, viridis)")
    p.add_argument("--classes",   default=5, type=int, help="Number of colour classes")
    p.add_argument("--out",       default="output/",   help="Output directory")
    p.add_argument("--groq-key",  default="",          help="Groq API key (or set GROQ_API_KEY env var)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Auto-detect district column in data if not given
    import pandas as pd
    df_preview = pd.read_csv(args.data) if args.data.endswith(".csv") \
                 else pd.read_excel(args.data, nrows=5)
    name_col = args.name_col
    if not name_col:
        obj_cols = list(df_preview.select_dtypes("object").columns)
        if not obj_cols:
            print("ERROR: Could not auto-detect district name column. Use --name-col.")
            sys.exit(1)
        name_col = obj_cols[0]
        print(f"Auto-detected district column in data: '{name_col}'")

    state = MapFlowState(
        data_path      = args.data,
        shapefile_path = args.shapefile,
        data_name_col  = name_col,
        value_col      = args.col,
        shp_name_col   = args.shp_col,
        title          = args.title,
        source         = args.source,
        boundary_year  = args.year,
        scheme         = args.scheme,
        cmap_name      = args.cmap,
        n_classes      = args.classes,
        output_dir     = args.out,
        groq_api_key   = args.groq_key,
    )

    final_state = run_flow(state)
    print(f"\nDone. Outputs in: {Path(args.out).resolve()}")
