# MASTERPROMPT — mapgen(ai)

You are building **mapgen(ai)**, a production-ready district choropleth map generator for Indian policy researchers at Pahle India Foundation. All code already exists in this repo. Your job is to wire it together, fix any remaining issues, and make it deployable.

---

## What this tool does

Researchers upload a CSV/XLSX with district-level data (e.g. literacy rate by district) + a shapefile. The tool matches district names, renders a choropleth map, and exports PNG / JPG / SVG / interactive HTML / standalone Python script / GeoJSON.

---

## Repo structure (already written — do not rewrite from scratch)

```
mapgen(ai)/
├── app.py             # Streamlit UI — 7-step flow
├── map_engine.py      # Deterministic core: matching, rendering, exports
├── groq_matcher.py    # Single batched Groq API call for leftover names
├── flows.py           # CrewAI Flow for headless/CLI use
├── aliases.py         # India district name alias dictionary
├── requirements.txt
└── sample_data/
    └── karnataka_sample.csv
```

---

## Design rules — READ BEFORE TOUCHING ANYTHING

1. **Pipeline is deterministic Python.** No LLM touches the map rendering logic. The only LLM call is in `groq_matcher.py` — one batched call for district names that survive all 4 rule-based tiers.

2. **Matching tier chain** (in strict order — earlier tiers handle as much as possible):
   - Tier 1: Exact string match
   - Tier 2: Normalized match (lowercase, no accents/punctuation)
   - Tier 3: Alias lookup (`aliases.py`)
   - Tier 4: Fuzzy match (rapidfuzz)
   - Tier 5: Groq API — batch ALL leftovers in ONE call
   - Tier 6: Human confirmation gate — any match below 0.88 confidence shown to user BEFORE rendering

3. **Honest map defaults — non-negotiable:**
   - Missing-data districts render gray + hatching (`///`), NEVER as zero, NEVER blank
   - Every map output has: legend, title, footnote with source + boundary vintage year
   - The footnote ties boundary vintage to the SHAPEFILE, not the data

4. **Human-in-the-loop is mandatory.** A silently mis-assigned district in a published map is the worst failure mode.

5. **Out of scope for v1:** PDF input, point-level icon markers, multi-state maps, custom ML model for matching.

---

## Your tasks

### 1. Read the existing code first
Before writing a single line, read all 5 Python files. Understand the function signatures, especially:
- `match_districts()` → returns `MatchResult` dataclass
- `apply_manual_corrections()` → takes `MatchResult` + corrections dict
- `join_to_geo()` → left-joins data onto GeoDataFrame, NaN for unmatched
- `render_static()` → matplotlib, returns `plt.Figure`
- `render_interactive()` → folium, returns `folium.Map`
- `generate_script()` → returns standalone Python script as string

### 2. Install dependencies and do a smoke test
```bash
pip install -r requirements.txt
python -c "from map_engine import match_districts; r = match_districts(['Bangalore','XYZ'], ['Bengaluru','Mysuru']); print(r.summary())"
```
Expected: `1/2 auto-matched | 0 need review | 1 unmatched`

### 3. Test the full Streamlit app locally
```bash
streamlit run app.py
```
Upload `sample_data/karnataka_sample.csv` as the data file and a Karnataka shapefile. Walk through all 7 steps. Fix any runtime errors you find.

### 4. Fix known issue: shapefile upload in Streamlit
In `app.py` Step 2, the shapefile is loaded into a temp directory but the `GeoDataFrame` reference may not persist correctly across Streamlit reruns because the temp directory is cleaned up. Fix this by:
- Serializing the GeoDataFrame to GeoJSON bytes in `st.session_state` after loading
- Deserializing back to GeoDataFrame when needed in later steps
- Do NOT store the file path — store the actual geodata

```python
import io, geopandas as gpd

# After loading:
buf = io.BytesIO()
gdf.to_file(buf, driver="GeoJSON")
st.session_state.gdf_bytes = buf.getvalue()

# When reading back:
gdf = gpd.read_file(io.BytesIO(st.session_state.gdf_bytes))
```

### 5. Add a .gitignore
Create `.gitignore`:
```
__pycache__/
*.pyc
*.pyo
.env
.env.local
shapefiles/*.shp
shapefiles/*.shx
shapefiles/*.dbf
shapefiles/*.prj
shapefiles/*.geojson
shapefiles/*.zip
!shapefiles/.gitkeep
output/
*.png
*.jpg
*.svg
*.html
!sample_data/
.streamlit/secrets.toml
```

### 6. Add Streamlit config
Create `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#1a56db"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 7. Add environment variable handling for Groq key
In `app.py`, the Groq key input in the sidebar already exists. Also support reading it from `st.secrets` for deployment:
```python
groq_key = st.sidebar.text_input(...) or st.secrets.get("GROQ_API_KEY", "")
```

### 8. Deployment prep
The app uses Streamlit. **Do NOT deploy to Vercel** — Vercel cannot run Streamlit. Instead:

**Option A — Streamlit Community Cloud (recommended, free):**
- Push repo to GitHub
- Go to share.streamlit.io → connect repo → set main file as `app.py`
- Add `GROQ_API_KEY` in the Streamlit secrets UI

**Option B — Railway.app:**
- Add `Procfile`: `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Add env var `GROQ_API_KEY` in Railway dashboard

**Option C — If Vercel is required**, rewrite the frontend as a Next.js app that calls a Python FastAPI backend hosted separately (e.g. on Railway). The `map_engine.py` core stays unchanged — wrap it in FastAPI endpoints:
- `POST /match` → run tiered matching, return JSON
- `POST /render` → render map, return PNG/HTML bytes
- `GET /export/{format}` → download specific format

### 9. Final checklist before pushing to GitHub
- [ ] `streamlit run app.py` completes all 7 steps without error on sample data
- [ ] PNG download works
- [ ] Interactive HTML download opens in browser and shows tooltips
- [ ] Standalone `.py` script download runs cleanly with `python reproduce_map.py`
- [ ] Missing-data districts show gray + hatching (NOT zero)
- [ ] Map footnote shows source + boundary year
- [ ] No API key → tool still works (Groq step is skipped, unmatched go to human review)
- [ ] `.gitignore` excludes shapefiles and output files

---

## Key constraints reminder
- `rapidfuzz` not `difflib` (already in requirements — just use it)
- `mapclassify` for classification schemes (quantiles / equal_interval / fisher_jenks)
- `folium` for interactive HTML — self-contained, no external tile server required
- Groq model: `llama3-8b-8192` (free tier, sufficient for name matching)
- All exports happen client-side via Streamlit `st.download_button` — no server-side file writes needed in production
