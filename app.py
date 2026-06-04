"""
app.py — Streamlit UI for District Map Generator
=================================================
Multi-step interface that mirrors the CrewAI Flow in flows.py but with
interactive widgets instead of CLI prompts.

Run with:
    streamlit run app.py

Steps:
  1  Upload data file (CSV / XLSX)
  2  Upload shapefile (or GeoJSON)
  3  Select columns (district name + value to map)
  4  District matching  — auto-runs tiers 1–4, then optional Groq
  5  Human-confirm gate — review low-confidence + unmatched matches
  6  Map configuration  — title, source, color ramp, classification
  7  Preview + download  — PNG, JPG, SVG, HTML, .py script, GeoJSON
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from state_agent import (
    detect_state,
    filter_by_state,
    resolve_ambiguous_state,
    StateDetectionResult,
)
from map_engine import (
    CLASSIFICATION_SCHEMES,
    COLOR_RAMPS,
    HIGH_CONF_THRESHOLD,
    MatchResult,
    apply_manual_corrections,
    detect_district_column,
    fig_to_bytes,
    generate_script,
    geojson_bytes,
    join_to_geo,
    load_shapefile,
    map_to_html_bytes,
    match_districts,
    render_interactive,
    render_static,
)
from groq_matcher import apply_groq_results, batch_resolve


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="mapgen — District Map Generator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,'
    'PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiI+'
    'PHJlY3Qgd2lkdGg9IjMyIiBoZWlnaHQ9IjMyIiBmaWxsPSIjMEEwQTBBIi8+PHRleHQgeD0iMTYiIHk9'
    'IjIzIiBmb250LWZhbWlseT0iR2VvcmdpYSxzZXJpZiIgZm9udC1zaXplPSIyMCIgZm9udC13ZWlnaHQ9'
    'IjMwMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI0ZGRkZGRiI+bTwvdGV4dD48cmVjdCB4PSI3'
    'IiB5PSIyNyIgd2lkdGg9IjE4IiBoZWlnaHQ9IjIiIGZpbGw9IiNENDJCMkIiLz48L3N2Zz4=">',
    unsafe_allow_html=True
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --cream:        #FDFAF6;
    --cream-mid:    #F2EAE0;
    --cream-dark:   #E8DDD0;
    --orange:       #C8511B;
    --orange-light: #E8621A;
    --orange-pale:  #FBF0E9;
    --ink:          #1C1208;
    --ink-mid:      #4A3728;
    --ink-light:    #8C7060;
    --rule:         #DDD0C4;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
    background-color: var(--cream);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main content area ── */
.main .block-container {
    padding: 2.5rem 3rem 3rem 3rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--cream-mid);
    border-right: 1px solid var(--rule);
}
[data-testid="stSidebar"] * { color: var(--ink-mid) !important; }
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.82rem;
    letter-spacing: 0.01em;
    padding: 0.2rem 0;
    color: var(--ink-mid) !important;
}

/* ── Wordmark ── */
.mapgen-wordmark {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--orange) !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
}
.mapgen-sub {
    font-size: 0.72rem;
    color: var(--ink-light) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Step progress in sidebar ── */
.step-done  { color: var(--orange) !important; font-size: 0.82rem; }
.step-active{ color: var(--ink) !important; font-size: 0.82rem; font-weight: 600; }
.step-todo  { color: var(--ink-light) !important; font-size: 0.82rem; }

/* ── Page headers ── */
h1, h2, h3 {
    font-family: 'Libre Baskerville', serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
}
h1 { font-size: 2rem !important; letter-spacing: -0.02em; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.4rem !important; margin-bottom: 0.2rem !important; }
h3 { font-size: 1.1rem !important; }

/* ── Step label ── */
.step-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--orange);
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* ── Dividers ── */
hr { border-color: var(--rule) !important; margin: 1.5rem 0 !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.4rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background-color: var(--orange) !important;
    border: none !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--orange-light) !important;
}
.stButton > button:not([kind="primary"]) {
    background-color: transparent !important;
    border: 1px solid var(--rule) !important;
    color: var(--ink-mid) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--orange) !important;
    color: var(--orange) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--cream-dark) !important;
    border-radius: 6px !important;
    background: var(--orange-pale) !important;
    padding: 0.5rem !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox > div > div {
    border-radius: 4px !important;
    border-color: var(--cream-dark) !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 2px rgba(200,81,27,0.12) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--orange-pale);
    border: 1px solid var(--cream-dark);
    border-radius: 6px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-light) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--orange) !important;
}

/* ── Alerts ── */
.stSuccess { background: #F0FAF0 !important; border-left: 3px solid #3D9A3D !important; border-radius: 4px !important; }
.stWarning { background: #FFF8EE !important; border-left: 3px solid var(--orange) !important; border-radius: 4px !important; }
.stInfo    { background: var(--orange-pale) !important; border-left: 3px solid var(--orange) !important; border-radius: 4px !important; }
.stError   { background: #FFF0F0 !important; border-left: 3px solid #C0392B !important; border-radius: 4px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--cream-dark) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--rule) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    color: var(--ink-light) !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: 0.03em !important;
}
.stTabs [aria-selected="true"] {
    color: var(--orange) !important;
    border-bottom-color: var(--orange) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--ink-mid) !important;
}

/* ── Caption / small text ── */
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: var(--ink-light) !important;
    font-size: 0.78rem !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding: 0.5rem 0 !important;
}

/* ── Sidebar reset button ── */
[data-testid="stSidebar"] .stButton > button {
    border: 1px solid var(--cream-dark) !important;
    color: var(--ink-light) !important;
    font-size: 0.78rem !important;
    background: transparent !important;
    margin-top: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--orange) !important;
    color: var(--orange) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "step":           0,
        "df":             None,
        "gdf":            None,
        "data_name_col":  None,
        "shp_name_col":   None,
        "value_col":      None,
        "match_result":       None,
        "corrections":        {},      # data_name → shp_name or None
        "state_detection":    None,    # StateDetectionResult
        "active_gdf":         None,    # GDF after optional state crop
        "merged_gdf":         None,
        "fig":            None,
        "folium_map":     None,
        "title":          "",
        "source":         "",
        "boundary_year":  "",
        "scheme":         "quantiles",
        "cmap_name":      "Blues",
        "n_classes":      5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar: progress tracker ─────────────────────────────────────────────────

STEPS = [
    "1  Upload data",
    "2  Upload shapefile",
    "3  Select columns",
    "4  Match districts",
    "5  Confirm matches",
    "6  Configure map",
    "7  Preview & export",
]

with st.sidebar:
    st.markdown('<div class="mapgen-wordmark">mapgen</div>', unsafe_allow_html=True)
    st.markdown('<div class="mapgen-sub">Pahle India Foundation</div>', unsafe_allow_html=True)
    st.divider()

    for i, label in enumerate(STEPS, start=1):
        cur = st.session_state.step
        if i < cur:
            st.markdown(f'<p class="step-done">— {label}</p>', unsafe_allow_html=True)
        elif i == cur:
            st.markdown(f'<p class="step-active">› {label}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="step-todo">  {label}</p>', unsafe_allow_html=True)

    st.divider()
    _groq_default = os.environ.get("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
    groq_key = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Used only for district names that survive all rule-based tiers. "
             "Free tier at console.groq.com. Leave blank to skip.",
        value=_groq_default,
    )

    st.caption(
        "Groq is called once per upload, batching all unresolved names. "
        "Leave blank to rely on rules and fuzzy matching alone."
    )

    if st.button("Start over", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 -- Landing page
# ===========================================================================

if st.session_state.step == 0:
    import os as _os
    _lpath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'landing.html')
    with open(_lpath) as _f:
        _html = _f.read()
    st.components.v1.html(_html, height=780, scrolling=False)
    _, col_btn, _ = st.columns([1.3, 1, 1.3])
    with col_btn:
        if st.button('Enter', type='primary', use_container_width=True):
            st.session_state.step = 1
            st.rerun()


# STEP 1 — Upload data file
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    st.header("Step 1: Upload your data file")
    st.caption("Accepted formats: CSV (.csv) or Excel (.xlsx)")

    uploaded = st.file_uploader("Data file", type=["csv", "xlsx"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")

            st.success(f"Loaded **{len(df):,} rows** × **{len(df.columns)} columns**")
            st.dataframe(df.head(8), use_container_width=True)

            st.session_state.df = df
            if st.button("Continue →", type="primary"):
                st.session_state.step = 2
                st.rerun()

        except Exception as e:
            st.error(f"Could not read file: {e}")

    st.divider()
    st.info(
        "Don't have a file yet? "
        "[Download the sample Karnataka CSV](sample_data/karnataka_sample.csv) "
        "to try the tool."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Upload shapefile
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    st.header("Step 2: Upload your district shapefile")
    st.caption(
        "Supported: GeoJSON (.geojson) or zipped shapefile (.zip containing "
        ".shp/.shx/.dbf/.prj). Must cover the same state as your data."
    )

    # ── Bundled all-India shapefile ───────────────────────────────────────
    BUNDLED_SHP = Path(__file__).parent / "shapefiles" / "india_districts.zip"
    use_bundled = BUNDLED_SHP.exists()

    col_info, col_upload = st.columns([1, 1])
    with col_info:
        if use_bundled:
            st.success(
                "**All-India district shapefile included** (820 districts, 2024 boundaries).\n\n"
                "Use the bundled file or upload your own below."
            )
        else:
            st.info(
                "**Where to get shapefiles:**\n"
                "- [Datameet India Maps](https://github.com/datameet/maps) — open, state-level\n"
                "- [GADM](https://gadm.org) — global, level-2 = districts\n"
                "- [MapCruzin](https://mapcruzin.com)\n\n"
                "Note the **boundary vintage year** — it determines which districts exist "
                "in the file. Post-2011 splits won't appear in a 2011-boundary shapefile."
            )

    with col_upload:
        uploaded_shp = st.file_uploader(
            "Upload your own shapefile or GeoJSON (optional)",
            type=["geojson", "json", "zip"],
        )
        boundary_year = st.text_input(
            "Boundary vintage year",
            placeholder="e.g. 2011 or 2023",
            value="2024" if use_bundled and not uploaded_shp else "",
            help="The year your shapefile's boundaries represent. Used in the map footnote.",
        )

    # Auto-load bundled shapefile if no upload and not already loaded
    if not uploaded_shp and use_bundled and st.session_state.gdf is None:
        try:
            with st.spinner("Loading bundled all-India district shapefile…"):
                gdf = load_shapefile(BUNDLED_SHP)
                _buf = io.BytesIO()
                gdf.to_file(_buf, driver="GeoJSON")
                st.session_state.gdf_bytes = _buf.getvalue()
                gdf = gpd.read_file(io.BytesIO(st.session_state.gdf_bytes))
                st.session_state.gdf = gdf
                st.session_state.boundary_year = "2024"
            st.success(
                f"Loaded bundled shapefile — **{len(gdf):,} districts** | "
                f"Columns: {[c for c in gdf.columns if c != 'geometry']}"
            )
            st.dataframe(gdf.drop(columns="geometry").head(5), use_container_width=True)
            if st.button("Continue →", type="primary"):
                st.session_state.step = 3
                st.rerun()
        except Exception as e:
            st.error(f"Could not load bundled shapefile: {e}")

    if uploaded_shp:
        try:
            import tempfile, zipfile

            suffix = Path(uploaded_shp.name).suffix.lower()
            with tempfile.TemporaryDirectory() as tmp:
                if suffix == ".zip":
                    zip_path = Path(tmp) / "shp.zip"
                    zip_path.write_bytes(uploaded_shp.read())
                    with zipfile.ZipFile(zip_path) as z:
                        z.extractall(tmp)
                    shp_files = list(Path(tmp).glob("**/*.shp"))
                    if not shp_files:
                        geojson_files = list(Path(tmp).glob("**/*.geojson"))
                        load_path = geojson_files[0] if geojson_files else None
                    else:
                        load_path = shp_files[0]
                    if not load_path:
                        st.error("No .shp or .geojson found inside the zip.")
                        st.stop()
                else:
                    load_path = Path(tmp) / uploaded_shp.name
                    load_path.write_bytes(uploaded_shp.read())

                gdf = load_shapefile(load_path)

                # Serialize to GeoJSON bytes BEFORE temp dir is deleted
                _buf = io.BytesIO()
                gdf.to_file(_buf, driver="GeoJSON")
                st.session_state.gdf_bytes = _buf.getvalue()

            # Deserialize from bytes so the GDF is independent of temp dir
            gdf = gpd.read_file(io.BytesIO(st.session_state.gdf_bytes))

            st.success(
                f"Loaded **{len(gdf):,} districts** | "
                f"CRS: `{gdf.crs.to_string()}` | "
                f"Columns: {list(gdf.columns)}"
            )
            st.dataframe(
                gdf.drop(columns="geometry").head(5),
                use_container_width=True,
            )

            st.session_state.gdf = gdf
            st.session_state.boundary_year = boundary_year

            if st.button("Continue →", type="primary"):
                st.session_state.step = 3
                st.rerun()

        except Exception as e:
            st.error(f"Could not read shapefile: {e}")

    if st.button("Back"):
        st.session_state.step = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Select columns
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    st.header("Step 3: Select columns to map")

    df  = st.session_state.df
    gdf = st.session_state.gdf

    # ── District name column in data ──────────────────────────────────────
    obj_cols     = list(df.select_dtypes("object").columns)
    num_cols     = list(df.select_dtypes("number").columns)

    if not obj_cols:
        st.error("No text columns found in your data file. Cannot detect district names.")
        st.stop()
    if not num_cols:
        st.error("No numeric columns found in your data file. Nothing to map.")
        st.stop()

    st.subheader("Your data file")
    data_name_col = st.selectbox(
        "Which column contains district names?",
        options=obj_cols,
        index=0,
    )

    # Surface ALL numeric columns — never guess
    st.markdown("**Which column should be mapped?**")
    st.caption(
        "If you see multiple candidates (e.g. pop_2011, pop_2021), "
        "both are shown — pick the one you need."
    )
    value_col = st.selectbox("Value column to map", options=num_cols)

    # ── District name column in shapefile ─────────────────────────────────
    st.subheader("Shapefile")
    try:
        auto_shp_col = detect_district_column(gdf)
    except Exception:
        auto_shp_col = None

    shp_str_cols = [c for c in gdf.columns if c != "geometry"]
    default_idx  = shp_str_cols.index(auto_shp_col) if auto_shp_col in shp_str_cols else 0

    shp_name_col = st.selectbox(
        "Which shapefile column contains district names?",
        options=shp_str_cols,
        index=default_idx,
        help=f"Auto-detected: '{auto_shp_col}'",
    )

    st.divider()
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("**Sample district names in your data:**")
        st.write(df[data_name_col].dropna().unique()[:12].tolist())
    with col_r:
        st.markdown("**Sample district names in shapefile:**")
        st.write(gdf[shp_name_col].dropna().unique()[:12].tolist())

    st.divider()
    col_b, col_c = st.columns([1, 5])
    with col_b:
        if st.button("Back"):
            st.session_state.step = 2
            st.rerun()
    with col_c:
        if st.button("Detect state & run matching →", type="primary"):
            st.session_state.data_name_col = data_name_col
            st.session_state.shp_name_col  = shp_name_col
            st.session_state.value_col     = value_col

            # ── State detection agent ─────────────────────────────────────
            gdf        = st.session_state.gdf
            data_names = df[data_name_col].astype(str).tolist()

            with st.spinner("Detecting state from district names…"):
                detection = detect_state(
                    data_names   = data_names,
                    gdf          = gdf,
                    shp_name_col = shp_name_col,
                    state_col    = "STATE_UT" if "STATE_UT" in gdf.columns else
                                   next((c for c in gdf.columns
                                         if "state" in c.lower()), None),
                )
            st.session_state.state_detection = detection

            # If ambiguous and Groq key available → resolve
            if detection.multi_state and groq_key and len(detection.all_states) > 1:
                with st.spinner("Ambiguous state — asking Groq to resolve…"):
                    resolved = resolve_ambiguous_state(
                        data_names = data_names,
                        candidates = detection.all_states,
                        api_key    = groq_key,
                    )
                if resolved:
                    detection.state       = resolved
                    detection.multi_state = False

            # Apply crop if single state detected
            state_col = "STATE_UT" if "STATE_UT" in gdf.columns else None
            if detection.state and state_col:
                try:
                    cropped = filter_by_state(gdf, detection.state, state_col)
                    st.session_state.active_gdf = cropped
                except Exception:
                    st.session_state.active_gdf = gdf
            else:
                st.session_state.active_gdf = gdf

            st.session_state.step = 4
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Match districts
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("Step 4: District name matching")

    df            = st.session_state.df
    gdf           = st.session_state.active_gdf or st.session_state.gdf
    data_name_col = st.session_state.data_name_col
    shp_name_col  = st.session_state.shp_name_col
    value_col     = st.session_state.value_col
    detection     = st.session_state.state_detection

    # ── State detection banner ────────────────────────────────────────────
    if detection:
        state_col = "STATE_UT" if "STATE_UT" in st.session_state.gdf.columns else None
        full_gdf  = st.session_state.gdf

        if detection.state and not detection.multi_state:
            st.success(
                f"**State detected: {detection.state}** — "
                f"shapefile cropped to {len(gdf):,} districts "
                f"({detection.matched}/{detection.total} data names matched, "
                f"{detection.confidence:.0%} confidence)"
            )
        elif detection.multi_state:
            st.warning(
                f"**Multi-state data detected:** {', '.join(detection.all_states)}. "
                "Using full all-India shapefile."
            )
        else:
            st.info("State could not be detected — using full all-India shapefile.")

        # Override picker
        if state_col and detection.all_states:
            with st.expander("Override state crop"):
                override = st.selectbox(
                    "Crop map to a specific state",
                    options=["(use full all-India shapefile)"] + sorted(
                        full_gdf[state_col].dropna().unique().tolist()
                    ),
                    index=0,
                )
                if st.button("Apply override"):
                    if override.startswith("(use full"):
                        st.session_state.active_gdf = full_gdf
                    else:
                        try:
                            st.session_state.active_gdf = filter_by_state(
                                full_gdf, override, state_col
                            )
                        except Exception as e:
                            st.error(str(e))
                    st.session_state.merged_gdf = None
                    st.session_state.fig        = None
                    st.session_state.folium_map = None
                    st.rerun()

    st.divider()

    data_names = df[data_name_col].astype(str).tolist()
    shp_names  = gdf[shp_name_col].astype(str).tolist()

    with st.spinner("Running tiered matching (rules + fuzzy)…"):
        result = match_districts(data_names, shp_names)

    # ── Groq for leftovers ────────────────────────────────────────────────
    needs_groq = result.unmatched + [m.data_name for m in result.low_confidence]

    if needs_groq and groq_key:
        with st.spinner(f"Sending {len(needs_groq)} unresolved name(s) to Groq (one call)…"):
            groq_matches = batch_resolve(needs_groq, shp_names, api_key=groq_key)
            apply_groq_results(result, groq_matches, HIGH_CONF_THRESHOLD)

    st.session_state.match_result = result

    # ── Summary metrics ───────────────────────────────────────────────────
    total = len(data_names)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Auto-matched", f"{len(result.high_confidence)}/{total}",
               help="Accepted without human review")
    mc2.metric("Needs review", len(result.low_confidence),
               help="Low-confidence or Groq suggestions requiring confirmation")
    mc3.metric("Unmatched", len(result.unmatched),
               help="No candidate found — you can assign manually")

    st.divider()

    # ── High-confidence table ─────────────────────────────────────────────
    with st.expander(f"Auto-matched ({len(result.high_confidence)})", expanded=False):
        if result.high_confidence:
            rows = [{"Data name": m.data_name, "Shapefile name": m.shp_name,
                     "Tier": m.tier, "Confidence": f"{m.confidence:.0%}",
                     "Note": m.note}
                    for m in result.high_confidence]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No auto-matched districts.")

    col_b, col_c = st.columns([1, 5])
    with col_b:
        if st.button("Back"):
            st.session_state.step = 3
            st.rerun()
    with col_c:
        lbl = "Review & confirm →" if (result.low_confidence or result.unmatched) \
              else "All matched! Continue →"
        if st.button(lbl, type="primary"):
            st.session_state.step = 5
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Human-confirm gate
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 5:
    st.header("Step 5: Confirm district matches")

    result        = st.session_state.match_result
    gdf           = st.session_state.active_gdf or st.session_state.gdf
    shp_name_col  = st.session_state.shp_name_col
    shp_names     = sorted(gdf[shp_name_col].dropna().astype(str).unique().tolist())

    if not result.low_confidence and not result.unmatched:
        st.success("All districts matched with high confidence — no review needed!")
        if st.button("Continue to map config →", type="primary"):
            st.session_state.step = 6
            st.rerun()
        st.stop()

    corrections: dict[str, str | None] = {}

    # ── Low-confidence matches ────────────────────────────────────────────
    if result.low_confidence:
        st.subheader(f"Low-confidence matches ({len(result.low_confidence)})")
        st.caption(
            "These were matched by fuzzy rules or LLM suggestion. "
            "Accept each suggestion or pick the correct shapefile district."
        )

        for i, m in enumerate(result.low_confidence):
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(f"**Data name:** `{m.data_name}`")
                    st.caption(f"Tier: {m.tier} | Confidence: {m.confidence:.0%} | {m.note}")
                with c2:
                    choice = st.selectbox(
                        "Map to shapefile district",
                        options=["(skip — omit from map)"] + shp_names,
                        index=shp_names.index(m.shp_name) + 1
                              if m.shp_name in shp_names else 0,
                        key=f"low_conf_{i}",
                        label_visibility="collapsed",
                    )
                with c3:
                    badge_color = ""
                    st.markdown(f"{m.confidence:.0%}")

                corrections[m.data_name] = None if choice.startswith("(skip") else choice
                st.divider()

    # ── Unmatched ─────────────────────────────────────────────────────────
    if result.unmatched:
        st.subheader(f"Unmatched ({len(result.unmatched)})")
        st.caption(
            "No match was found by any rule. Assign manually or skip. "
            "Skipped districts will render as 'No data' (gray hatching)."
        )

        for j, name in enumerate(result.unmatched):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"**`{name}`**")
            with c2:
                choice = st.selectbox(
                    "Assign to",
                    options=["(skip — omit from map)"] + shp_names,
                    index=0,
                    key=f"unmatched_{j}",
                    label_visibility="collapsed",
                )
            corrections[name] = None if choice.startswith("(skip") else choice
            st.divider()

    col_b, col_c = st.columns([1, 5])
    with col_b:
        if st.button("Back"):
            st.session_state.step = 4
            st.rerun()
    with col_c:
        if st.button("Apply corrections & configure map →", type="primary"):
            updated_result = apply_manual_corrections(result, corrections)
            st.session_state.match_result = updated_result
            st.session_state.corrections  = corrections
            st.session_state.step = 6
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Map configuration
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 6:
    st.header("Step 6: Configure your map")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Labels & attribution")
        title = st.text_input(
            "Map title",
            value=st.session_state.title or
                  f"{st.session_state.value_col.replace('_', ' ').title()} by District",
        )
        source = st.text_input(
            "Data source",
            value=st.session_state.source,
            placeholder="e.g. Census of India 2011",
        )
        boundary_year = st.text_input(
            "Shapefile boundary vintage year",
            value=st.session_state.boundary_year,
            placeholder="e.g. 2011",
        )

    with c_right:
        st.subheader("Colour & classification")

        # Classification scheme
        scheme = st.selectbox(
            "Classification scheme",
            options=list(CLASSIFICATION_SCHEMES.keys()),
            index=0,
            format_func=lambda k: k.replace("_", " ").title(),
        )
        st.caption(CLASSIFICATION_SCHEMES[scheme])

        # Colour ramp
        cmap_name = st.selectbox(
            "Colour ramp",
            options=list(COLOR_RAMPS.keys()),
            index=0,
        )
        st.caption(COLOR_RAMPS[cmap_name])

        n_classes = st.slider("Number of colour classes", min_value=3, max_value=8, value=5)

    col_b, col_c = st.columns([1, 5])
    with col_b:
        if st.button("Back"):
            st.session_state.step = 5
            st.rerun()
    with col_c:
        if st.button("Render map →", type="primary"):
            st.session_state.title         = title
            st.session_state.source        = source
            st.session_state.boundary_year = boundary_year
            st.session_state.scheme        = scheme
            st.session_state.cmap_name     = cmap_name
            st.session_state.n_classes     = n_classes
            st.session_state.step = 7
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Preview and export
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 7:
    st.header("Step 7: Preview & export")

    df            = st.session_state.df
    gdf           = st.session_state.active_gdf or st.session_state.gdf
    data_name_col = st.session_state.data_name_col
    shp_name_col  = st.session_state.shp_name_col
    value_col     = st.session_state.value_col
    result        = st.session_state.match_result

    # ── Build merged GDF (or use cached) ─────────────────────────────────
    if st.session_state.merged_gdf is None:
        with st.spinner("Joining data to shapefile…"):
            name_map = result.as_dict()
            merged = join_to_geo(
                gdf           = gdf,
                data_df       = df,
                data_name_col = data_name_col,
                shp_name_col  = shp_name_col,
                name_map      = name_map,
                value_col     = value_col,
            )
            st.session_state.merged_gdf = merged

    merged = st.session_state.merged_gdf

    # ── Render (or use cached) ────────────────────────────────────────────
    render_params = dict(
        value_col     = "_value",
        scheme        = st.session_state.scheme,
        cmap_name     = st.session_state.cmap_name,
        n_classes     = st.session_state.n_classes,
        title         = st.session_state.title,
        source        = st.session_state.source,
        boundary_year = st.session_state.boundary_year,
    )

    if st.session_state.fig is None:
        with st.spinner("Rendering static map…"):
            st.session_state.fig = render_static(merged, **render_params)

    if st.session_state.folium_map is None:
        with st.spinner("Building interactive map…"):
            # render_interactive doesn't use source/boundary_year (footnote is
            # static-map-only) or figsize, so filter them out of the shared dict.
            ri_params = {k: v for k, v in render_params.items()
                         if k not in ("figsize", "source", "boundary_year")}
            st.session_state.folium_map = render_interactive(
                gdf       = merged,
                label_col = shp_name_col,
                **ri_params,
            )

    fig        = st.session_state.fig
    folium_map = st.session_state.folium_map

    # ── Tabs: Static | Interactive | Data ────────────────────────────────
    tab_static, tab_interactive, tab_data = st.tabs(
        ["Static map", "Interactive map", "Matched data"]
    )

    with tab_static:
        st.pyplot(fig, use_container_width=True)

    with tab_interactive:
        # Embed folium map via HTML component
        html_bytes = map_to_html_bytes(folium_map)
        st.components.v1.html(html_bytes.decode("utf-8"), height=600, scrolling=False)

    with tab_data:
        name_map   = result.as_dict()
        mapped_df  = df.copy()
        mapped_df["__shp_name"] = mapped_df[data_name_col].map(name_map)
        st.dataframe(
            mapped_df[[data_name_col, value_col, "__shp_name"]].rename(
                columns={"__shp_name": "Shapefile district"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        unmatched_count = result.unmatched.__len__()
        if unmatched_count:
            st.warning(
                f"{unmatched_count} district(s) have no data and will appear as "
                f"gray hatching on the map: {result.unmatched}"
            )

    st.divider()
    st.subheader("Downloads")

    dl1, dl2, dl3, dl4, dl5, dl6 = st.columns(6)

    with dl1:
        st.download_button(
            "PNG",
            data     = fig_to_bytes(fig, "png"),
            file_name= "district_map.png",
            mime     = "image/png",
            use_container_width=True,
        )

    with dl2:
        st.download_button(
            "JPG",
            data     = fig_to_bytes(fig, "jpg"),
            file_name= "district_map.jpg",
            mime     = "image/jpeg",
            use_container_width=True,
        )

    with dl3:
        st.download_button(
            "SVG",
            data     = fig_to_bytes(fig, "svg"),
            file_name= "district_map.svg",
            mime     = "image/svg+xml",
            use_container_width=True,
        )

    with dl4:
        st.download_button(
            "Interactive HTML",
            data     = map_to_html_bytes(folium_map),
            file_name= "district_map.html",
            mime     = "text/html",
            use_container_width=True,
        )

    with dl5:
        # Generate standalone .py script
        name_map_full = result.as_dict()
        data_records  = {}
        for data_name, shp_name in name_map_full.items():
            if shp_name:
                rows = df.loc[df[data_name_col] == data_name, value_col]
                data_records[shp_name] = float(rows.values[0]) if len(rows) else None

        script_code = generate_script(
            shapefile_path  = "<path/to/your/shapefile>",
            shp_name_col    = shp_name_col,
            data_records    = data_records,
            value_col_label = value_col,
            scheme          = st.session_state.scheme,
            cmap_name       = st.session_state.cmap_name,
            n_classes       = st.session_state.n_classes,
            title           = st.session_state.title,
            source          = st.session_state.source,
            boundary_year   = st.session_state.boundary_year,
        )
        st.download_button(
            "Python script",
            data     = script_code.encode("utf-8"),
            file_name= "reproduce_map.py",
            mime     = "text/x-python",
            use_container_width=True,
            help     = "Standalone script that reproduces this exact map. "
                       "Update the SHAPEFILE_PATH before running.",
        )

    with dl6:
        st.download_button(
            "GeoJSON",
            data     = geojson_bytes(merged),
            file_name= "district_map.geojson",
            mime     = "application/geo+json",
            use_container_width=True,
            help     = "Shapefile polygons with your data values joined — "
                       "import into QGIS, Mapbox, or any GIS tool.",
        )

    st.divider()
    col_b, col_re = st.columns([1, 5])
    with col_b:
        if st.button("Back to config"):
            st.session_state.step = 6
            st.session_state.fig  = None
            st.session_state.folium_map = None
            st.rerun()
    with col_re:
        if st.button("New map", type="secondary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
