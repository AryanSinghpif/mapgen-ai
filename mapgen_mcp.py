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

import io
import json
import os
import subprocess
import sys
import tempfile
import webbrowser
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
    return [
        types.Tool(
            name="profile_data",
            description=(
                "Load and analyse a CSV or Excel data file. "
                "Returns: detected format (wide/long), level (state/district), "
                "geo column, available value columns, and data quality notes. "
                "Always call this first before any other mapgen tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the CSV or Excel file to analyse.",
                    }
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="detect_level",
            description=(
                "Run Agent 2 to authoritatively determine whether data is "
                "state-level or district-level, and which Indian state(s) are covered. "
                "Call after profile_data once you know the geo column."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the data file."},
                    "geo_col":   {"type": "string", "description": "Name of the geography column."},
                },
                "required": ["file_path", "geo_col"],
            },
        ),
        types.Tool(
            name="match_names",
            description=(
                "Match geography names in the data against the India shapefile using "
                "4-tier matching (exact → alias → fuzzy → fallback). "
                "Returns match statistics and any unmatched names."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "geo_col":   {"type": "string", "description": "Geography column name."},
                    "level":     {"type": "string", "enum": ["state", "district"],
                                  "description": "Data level — from detect_level result."},
                },
                "required": ["file_path", "geo_col", "level"],
            },
        ),
        types.Tool(
            name="render_map",
            description=(
                "Render a choropleth map for India. Saves PNG + interactive HTML, "
                "opens the interactive map in the browser. "
                "Call after match_names with the column you want to visualise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path":  {"type": "string"},
                    "geo_col":    {"type": "string"},
                    "value_col":  {"type": "string", "description": "Numeric column to map."},
                    "level":      {"type": "string", "enum": ["state", "district"]},
                    "title":      {"type": "string", "description": "Map title (optional)."},
                    "cmap_name":  {"type": "string", "description": "Matplotlib colormap (default: YlOrRd)."},
                    "n_classes":  {"type": "integer", "description": "Number of colour classes (default: 5)."},
                    "output_dir": {"type": "string",
                                   "description": "Directory to save outputs (default: Desktop)."},
                },
                "required": ["file_path", "geo_col", "value_col", "level"],
            },
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── profile_data ──────────────────────────────────────────────────────
    if name == "profile_data":
        file_path = arguments["file_path"]
        try:
            raw_df       = _read_file(file_path)
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
        file_path = arguments["file_path"]
        geo_col   = arguments["geo_col"]
        try:
            df        = clean_dataframe(_read_file(file_path)).df
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
        file_path = arguments["file_path"]
        geo_col   = arguments["geo_col"]
        level     = arguments["level"]
        try:
            df        = clean_dataframe(_read_file(file_path)).df
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
        file_path  = arguments["file_path"]
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
            df        = clean_dataframe(_read_file(file_path)).df
            gdf       = _load_gdf()
            geo_names = df[geo_col].astype(str).tolist()

            # Match
            if level == "state":
                state_col = "STATE_UT" if "STATE_UT" in gdf.columns else gdf.columns[0]
                shp_vals  = gdf[state_col].dropna().unique().tolist()
                match_res = match_states(geo_names, shp_vals)
                name_map  = match_res.as_dict()
                merged    = join_state_to_geo(
                    gdf=gdf, data_df=df,
                    data_name_col=geo_col, state_col=state_col,
                    name_map=name_map, value_col=value_col,
                )
            else:
                dist_col  = detect_district_column(gdf)
                shp_names = gdf[dist_col].astype(str).tolist()
                match_res = match_districts(geo_names, shp_names)
                name_map  = match_res.as_dict()
                merged    = join_to_geo(
                    gdf=gdf, data_df=df,
                    data_name_col=geo_col, shp_name_col=dist_col,
                    name_map=name_map, value_col=value_col,
                )

            # Render
            stem     = Path(file_path).stem.replace(" ", "_")
            html_out = output_dir / f"{stem}_map.html"
            png_out  = output_dir / f"{stem}_map.png"

            fig = render_static(
                merged, value_col="_value",
                scheme="quantiles", cmap_name=cmap_name,
                n_classes=n_classes, title=title,
            )
            png_bytes = fig_to_bytes(fig, "png")
            png_out.write_bytes(png_bytes)

            folium_map = render_interactive(
                gdf=merged, value_col="_value",
                label_col=dist_col if level == "district" else state_col,
                title=title, cmap_name=cmap_name, n_classes=n_classes,
            )
            html_out.write_bytes(map_to_html_bytes(folium_map))

            # Open in browser
            webbrowser.open(f"file://{html_out}")

            out = {
                "status":        "ok",
                "title":         title,
                "level":         level,
                "matched":       len(match_res.high_confidence),
                "unmatched":     len(match_res.unmatched),
                "html_map":      str(html_out),
                "png_map":       str(png_out),
                "message":       f"Map saved and opened in browser. HTML: {html_out}",
            }
            return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

        except Exception as exc:
            return [types.TextContent(type="text",
                    text=json.dumps({"status": "error", "message": str(exc)}))]

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
