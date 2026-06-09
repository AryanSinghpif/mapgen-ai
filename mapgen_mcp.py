"""
mapgen_mcp.py — MCP Server for mapgen(ai)
==========================================
Exposes mapgen as MCP tools so Claude Desktop / Claude Code can call them.

Tools exposed:
  profile_data     — load + clean + profile a CSV/Excel file
  detect_level     — state vs district, which state(s)
  match_names      — run tiered name matching against India shapefile
  render_map       — render choropleth → save HTML + PNG, open in browser

Usage (Claude Desktop):
  Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
  {
    "mcpServers": {
      "mapgen": {
        "command": "/path/to/.venv/bin/python",
        "args": ["/path/to/mapgen_mcp.py"]
      }
    }
  }
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ── MCP SDK ───────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── mapgen modules ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from data_analyzer import clean_dataframe, profile_data, suggest_emoji, pivot_long_to_wide
from state_agent   import run as agent2_run
from map_engine    import (
    load_shapefile, match_districts, match_states,
    join_to_geo, join_state_to_geo,
    render_static, render_interactive,
    map_to_html_bytes, fig_to_bytes,
    detect_district_column,
)

# ── Bundled shapefile ─────────────────────────────────────────────────────────
_BUNDLED_SHP = _HERE / "shapefiles" / "india_districts.zip"

# ── File resolver: path on disk OR inline CSV text ────────────────────────────
def _resolve_df(arguments: dict) -> tuple[pd.DataFrame, str]:
    """
    Return (DataFrame, resolved_file_path).
    Accepts either:
      - file_path  — absolute path on local disk
      - csv_content — raw CSV text pasted inline
    """
    if "csv_content" in arguments and arguments["csv_content"]:
        text = arguments["csv_content"]
        tmp  = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="mapgen_upload_"
        )
        tmp.write(text); tmp.close()
        return pd.read_csv(tmp.name), tmp.name

    file_path = arguments.get("file_path", "")
    p = Path(file_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            "Tip: provide the full absolute path, e.g. /Users/yourname/Downloads/data.xlsx"
        )
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p), str(p)
    else:
        xl = pd.ExcelFile(p, engine="openpyxl")
        best_df, best_size = None, 0
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
                if df.size > best_size:
                    best_df, best_size = df, df.size
            except Exception:
                continue
        return best_df, str(p)


# ── Server ────────────────────────────────────────────────────────────────────
server = Server("mapgen")


def _load_gdf() -> gpd.GeoDataFrame:
    if not _BUNDLED_SHP.exists():
        raise FileNotFoundError(f"Bundled shapefile not found at {_BUNDLED_SHP}")
    return load_shapefile(_BUNDLED_SHP)


def _read_file(file_path: str) -> pd.DataFrame:
    p = Path(file_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        xl = pd.ExcelFile(p, engine="openpyxl")
        best_df, best_size = None, 0
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
                if df.size > best_size:
                    best_df, best_size = df, df.size
            except Exception:
                continue
        return best_df


# ── Tool: profile_data ────────────────────────────────────────────────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    _file_schema = {
        "file_path": {
            "type": "string",
            "description": (
                "Absolute path to the file on the user's Mac, e.g. "
                "/Users/thesinghaa/Downloads/data.xlsx. "
                "Ask the user for the full path if not provided. "
                "Alternatively use csv_content to pass data inline."
            ),
        },
        "csv_content": {
            "type": "string",
            "description": (
                "Raw CSV text of the data (use when the file was uploaded as an "
                "attachment and no local path is available). Convert the attachment "
                "to CSV text and pass it here."
            ),
        },
    }

    return [
        types.Tool(
            name="start_mapping",
            description=(
                "ALWAYS call this first when a user wants to create a map or says things like "
                "'map this', 'create a map', 'visualise this data', 'choropleth', etc. "
                "Profiles the data and returns a conversational question set so you can ask "
                "the user what they want to map — do NOT proceed to render without asking. "
                "Needs file_path (absolute Mac path) OR csv_content (raw CSV text). "
                "If the user only provides a filename without a full path, ask: "
                "'What is the full path to your file? e.g. /Users/yourname/Downloads/file.xlsx'"
            ),
            inputSchema={
                "type": "object",
                "properties": _file_schema,
            },
        ),
        types.Tool(
            name="profile_data",
            description=(
                "Low-level profile tool — use start_mapping instead for conversational flows. "
                "Load and analyse an India geo-data file (CSV or Excel). "
                "Returns: geo column, value columns, level (state/district), data quality."
            ),
            inputSchema={
                "type": "object",
                "properties": _file_schema,
            },
        ),
        types.Tool(
            name="detect_level",
            description=(
                "STEP 2 — Detect whether data is state-level or district-level. "
                "Call after profile_data. Pass the same file_path or csv_content used in step 1."
            ),
            inputSchema={
                "type": "object",
                "properties": {**_file_schema, "geo_col": {"type": "string"}},
                "required": ["geo_col"],
            },
        ),
        types.Tool(
            name="match_names",
            description=(
                "STEP 3 — Match geography names against India shapefile. "
                "Call after detect_level. Pass the same file_path or csv_content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    **_file_schema,
                    "geo_col": {"type": "string"},
                    "level":   {"type": "string", "enum": ["state", "district"]},
                },
                "required": ["geo_col", "level"],
            },
        ),
        types.Tool(
            name="render_map",
            description=(
                "STEP 4 — Render choropleth map. Returns PNG shown inline in chat + "
                "saves interactive HTML to Desktop. "
                "Call after match_names. Pass the same file_path or csv_content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    **_file_schema,
                    "geo_col":    {"type": "string"},
                    "value_col":  {"type": "string", "description": "Numeric column to map."},
                    "level":      {"type": "string", "enum": ["state", "district"]},
                    "title":      {"type": "string"},
                    "cmap_name":  {"type": "string", "description": "Matplotlib colormap (default: YlOrRd)."},
                    "n_classes":  {"type": "integer"},
                    "output_dir": {"type": "string"},
                },
                "required": ["geo_col", "value_col", "level"],
            },
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── start_mapping ─────────────────────────────────────────────────────
    if name == "start_mapping":
        try:
            raw_df, file_path = _resolve_df(arguments)
            clean_result = clean_dataframe(raw_df)
            df           = clean_result.df
            prof         = profile_data(df)
            geo_names    = df[prof.geo_col].astype(str).tolist() if prof.geo_col else []

            # Run level detection
            level_info = {}
            if geo_names:
                try:
                    gdf    = _load_gdf()
                    result = agent2_run(geo_names=geo_names, gdf=gdf)
                    level_info = {
                        "level":      result.level,
                        "confidence": f"{result.level_confidence:.0%}",
                        "coverage":   result.coverage,
                        "state":      result.state,
                    }
                except Exception:
                    level_info = {"level": prof.level or "unknown"}

            # Build human-readable column list
            val_cols = prof.value_cols or []
            col_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(val_cols))

            # Emoji suggestions
            emojis = suggest_emoji(col_name=prof.geo_col or "")
            emoji_str = "  ".join(f"{e}" for e, _ in emojis[:6]) if emojis else "📊"

            out = {
                "status":        "ready_to_ask",
                "file":          Path(file_path).name,
                "rows":          prof.n_rows,
                "geo_col":       prof.geo_col,
                "level":         level_info.get("level", "unknown"),
                "level_confidence": level_info.get("confidence", ""),
                "coverage":      level_info.get("coverage", ""),
                "state":         level_info.get("state"),
                "value_columns": val_cols,
                "cleaning_done": clean_result.changes,
                "issues":        prof.issues,

                # ── Prompt Claude to ask these questions ──────────────────
                "INSTRUCTIONS_FOR_CLAUDE": (
                    "Present this to the user conversationally. Say something like:\n\n"
                    f"'Got it! I've analysed **{Path(file_path).name}** — "
                    f"it has {prof.n_rows} rows of **{level_info.get('level','?')}-level** "
                    f"India data ({level_info.get('coverage','')}).\n\n"
                    f"Here are the columns I can map:\n{col_list}\n\n"
                    "A few quick questions before I render:\n"
                    "1. **Which column** do you want to map? (pick a number or name)\n"
                    "2. **Map title** — what should it say? (or I'll auto-generate one)\n"
                    "3. **Colour scheme** — warm 🔴 (YlOrRd), cool 🔵 (Blues), green 🟢 (YlGn), or diverging ↔️ (RdYlGn)?\n"
                    "4. **Number of colour classes** — 4, 5 (default), or 6?\n\n"
                    f"Suggested emojis for the legend: {emoji_str}'"
                ),
            }
            return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

        except Exception as exc:
            return [types.TextContent(type="text",
                    text=json.dumps({"status": "error", "message": str(exc)}))]

    # ── profile_data ──────────────────────────────────────────────────────
    elif name == "profile_data":
        try:
            raw_df, file_path = _resolve_df(arguments)
            clean_result = clean_dataframe(raw_df)
            df           = clean_result.df
            profile      = profile_data(df)
            emojis       = suggest_emoji(col_name=profile.geo_col or "")

            result = {
                "status":       "ok",
                "file":         Path(file_path).name,
                "rows":         profile.n_rows,
                "cols":         profile.n_cols,
                "format":       profile.fmt,
                "level":        profile.level,
                "geo_col":      profile.geo_col,
                "value_cols":   profile.value_cols,
                "variable_col": profile.variable_col,
                "issues":       profile.issues,
                "notes":        profile.notes,
                "cleaning":     clean_result.changes,
                "emoji_suggestions": [f"{e} {l}" for e, l in emojis],
                "sample_geo_values": (
                    df[profile.geo_col].dropna().astype(str).head(8).tolist()
                    if profile.geo_col else []
                ),
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as exc:
            return [types.TextContent(type="text",
                    text=json.dumps({"status": "error", "message": str(exc)}))]

    # ── detect_level ──────────────────────────────────────────────────────
    elif name == "detect_level":
        geo_col   = arguments["geo_col"]
        try:
            raw_df, file_path = _resolve_df(arguments)
            df        = clean_dataframe(raw_df).df
            gdf       = _load_gdf()
            geo_names = df[geo_col].astype(str).tolist()

            result    = agent2_run(geo_names=geo_names, gdf=gdf)
            out = {
                "status":           "ok",
                "level":            result.level,
                "confidence":       f"{result.level_confidence:.0%}",
                "reasoning":        result.level_reasoning,
                "matched":          result.matched,
                "total":            result.total,
                "coverage":         result.coverage,
                "state":            result.state,
                "all_states":       result.all_states[:5],
                "multi_state":      result.multi_state,
            }
            return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

        except Exception as exc:
            return [types.TextContent(type="text",
                    text=json.dumps({"status": "error", "message": str(exc)}))]

    # ── match_names ───────────────────────────────────────────────────────
    elif name == "match_names":
        geo_col   = arguments["geo_col"]
        level     = arguments["level"]
        try:
            raw_df, file_path = _resolve_df(arguments)
            df        = clean_dataframe(raw_df).df
            gdf       = _load_gdf()
            geo_names = df[geo_col].astype(str).tolist()

            if level == "state":
                state_col  = "STATE_UT" if "STATE_UT" in gdf.columns else gdf.columns[0]
                shp_vals   = gdf[state_col].dropna().unique().tolist()
                match_res  = match_states(geo_names, shp_vals)
            else:
                dist_col   = detect_district_column(gdf)
                shp_names  = gdf[dist_col].astype(str).tolist()
                match_res  = match_districts(geo_names, shp_names)

            out = {
                "status":          "ok",
                "level":           level,
                "auto_matched":    len(match_res.high_confidence),
                "needs_review":    len(match_res.low_confidence),
                "unmatched":       len(match_res.unmatched),
                "total":           len(geo_names),
                "unmatched_names": match_res.unmatched[:10],
                "sample_matches":  [
                    {"data": m.data_name, "shapefile": m.shp_name,
                     "confidence": f"{m.confidence:.0%}"}
                    for m in match_res.high_confidence[:5]
                ],
            }
            return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

        except Exception as exc:
            return [types.TextContent(type="text",
                    text=json.dumps({"status": "error", "message": str(exc)}))]

    # ── render_map ────────────────────────────────────────────────────────
    elif name == "render_map":
        geo_col    = arguments["geo_col"]
        value_col  = arguments["value_col"]
        level      = arguments["level"]
        title      = arguments.get("title", value_col.replace("_", " ").title())
        cmap_name  = arguments.get("cmap_name", "YlOrRd")
        n_classes  = int(arguments.get("n_classes", 5))
        output_dir = Path(arguments.get("output_dir",
                          Path.home() / "Desktop")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ── Render inline (synchronous) so PNG can be returned in chat ──
            raw_df, file_path = _resolve_df(arguments)

            stem     = Path(file_path).stem.replace(" ", "_")
            html_out = output_dir / f"{stem}_map.html"
            png_out  = output_dir / f"{stem}_map.png"

            from data_analyzer import clean_dataframe
            from map_engine import (
                load_shapefile, match_districts, match_states,
                join_to_geo, join_state_to_geo,
                render_static, render_interactive,
                map_to_html_bytes, fig_to_bytes,
                detect_district_column,
            )

            df  = clean_dataframe(raw_df).df
            gdf = _load_gdf()

            if level == "state":
                state_col = "STATE_UT" if "STATE_UT" in gdf.columns else gdf.columns[0]
                match_res = match_states(
                    df[geo_col].astype(str).tolist(),
                    gdf[state_col].dropna().unique().tolist()
                )
                merged    = join_state_to_geo(
                    gdf=gdf, data_df=df, data_name_col=geo_col,
                    state_col=state_col, name_map=match_res.as_dict(),
                    value_col=value_col
                )
                label_col = state_col
            else:
                dist_col  = detect_district_column(gdf)
                match_res = match_districts(
                    df[geo_col].astype(str).tolist(),
                    gdf[dist_col].astype(str).tolist()
                )
                merged    = join_to_geo(
                    gdf=gdf, data_df=df, data_name_col=geo_col,
                    shp_name_col=dist_col, name_map=match_res.as_dict(),
                    value_col=value_col
                )
                label_col = dist_col

            # Interactive HTML
            fm = render_interactive(
                gdf=merged, value_col="_value", label_col=label_col,
                title=title, cmap_name=cmap_name, n_classes=n_classes
            )
            html_bytes = map_to_html_bytes(fm)
            html_out.write_bytes(html_bytes)

            # Static PNG (shown inline in chat)
            fig      = render_static(
                merged, value_col="_value", scheme="quantiles",
                cmap_name=cmap_name, n_classes=n_classes, title=title
            )
            png_bytes = fig_to_bytes(fig, "png")
            png_out.write_bytes(png_bytes)

            summary = {
                "status":   "done",
                "title":    title,
                "level":    level,
                "matched":  len(match_res.high_confidence),
                "html_map": str(html_out),
                "png_map":  str(png_out),
                "exports":  "Open the HTML file for interactive map + GeoJSON/CSV/Print export panel.",
            }

            return [
                # PNG shown inline in Claude Desktop chat
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(png_bytes).decode(),
                    mimeType="image/png",
                ),
                # Summary text
                types.TextContent(type="text", text=json.dumps(summary, indent=2)),
            ]

        except Exception as exc:
            import traceback
            return [types.TextContent(type="text",
                    text=json.dumps({"status": "error", "message": str(exc),
                                     "trace": traceback.format_exc()[-800:]}))]

    return [types.TextContent(type="text",
            text=json.dumps({"status": "error", "message": f"Unknown tool: {name}"}))]


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
