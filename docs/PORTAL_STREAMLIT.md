# Portal UI: Streamlit (Dashboards + Ops Health)

Status: Draft  
Goal: add a polished, interactive UI layer **without** abandoning the CLI-first workflow.

---

## Why Streamlit here

Streamlit is a strong fit for `options-helper` because:
- It’s Python-native and quick to iterate.
- It works extremely well with embedded analytics backends like DuckDB.
- You can ship a “product-like” portal with minimal glue.

Important constraint:
- The portal should be **read-mostly**. It queries DuckDB and reads snapshot artifacts.
- Ingestion stays in the CLI and/or Dagster.

---

## Naming (avoid confusion with existing `dashboard`)

You already have a CLI command named `dashboard` that produces a read-only dashboard output.

To avoid confusion, we’ll refer to the Streamlit UI as:

- **Portal** (recommended)
- or “Web UI”

Suggested naming:
- directory: `apps/portal/` or `apps/streamlit/`
- command (optional): `options-helper ui` (just a launcher that shells out to Streamlit)

---

## App structure (recommended)

Use a multipage app with explicit navigation:

```
apps/streamlit/
  streamlit_app.py
  pages/
    01_Health.py
    02_Portfolio.py
    03_Symbol_Explorer.py
    04_Flow.py
    05_Derived_History.py
    06_Data_Explorer.py
  components/
    db.py
    queries.py
    filters.py
    charts.py
    layout.py
```

### Theme
This pack includes a Stockpeers-inspired theme file at:

- `.streamlit/config.toml`

Place it in your repo root (or run Streamlit from the directory that contains it).

See: `docs/PORTAL_STOCKPEERS_STYLE.md`

---

## Portal pages (MVP)

### 1) Health (the “ingestion visibility tool”)
**Purpose:** operational visibility for ingestion + snapshot + derived jobs.

Data sources:
- `meta.ingestion_runs`
- `meta.ingestion_run_assets`
- `meta.asset_watermarks`
- `meta.asset_checks`

Key UI widgets:
- date range (last 7/30/90 days)
- job selector
- asset selector
- symbol filter (watchlists + search)
- “show only failures” toggle

Key charts/tables:
- run timeline (status + duration)
- freshness table (watermark + staleness)
- missing partitions / gap counts
- quality check failures (latest + history)
- grouped errors (stack hash) for “top recurring failures”

### 2) Portfolio
**Purpose:** daily driver for positions + risk.

Data sources:
- portfolio JSON file
- any stored portfolio-level snapshots/outputs (as they exist today)
- candles cache for underlyings

Key visuals:
- positions table with drill-down to symbol pages
- portfolio Greeks + stress scenarios
- daily performance (best effort)

### 3) Symbol Explorer
**Purpose:** interactive replacement for common research flows.

Data sources:
- `candles_daily` (DuckDB)
- `options_snapshot_file` (latest snapshot, plus history if available)
- derived metrics (trend/regime state)

Key visuals:
- price chart with technical overlays (keep it clean)
- current snapshot summary (OI/volume by strike, IV smile)
- derived signals and historical context

### 4) Flow
**Purpose:** positioning proxy + “unusual activity” exploration.

Data sources:
- `options_flow` (DuckDB), derived from `snapshot-options` + `flow`

Key visuals:
- top call/put flows (ΔOI proxy) by expiry and DTE bucket
- filters: expiry, DTE, moneyness/delta buckets
- time-series view (requires derived history)

### 5) Derived History (optional for MVP, but high value)
- per-symbol metrics over time
- regime changes, trend state changes, volatility state changes

### 6) Data Explorer (debug tool)
- list DuckDB tables/views + row counts
- show schema
- preview recent rows
- allow “copy SQL” / export (nice-to-have)

---

## Database access & caching

### DuckDB connection
Use one cached resource connection for the session:

- `st.cache_resource` → DuckDB connection factory
- Streamlit pages call the shared `get_conn()` helper.

Recommended:
- Portal uses read-only connections if feasible.
- Ingestion can run concurrently (writer + readers).

### Query caching
Use `st.cache_data` for query results, keyed by:
- SQL string
- parameter tuple
- selected date ranges

Use a TTL for operational dashboards (e.g., 30–120 seconds).

---

## Deep links / shareable URLs

Support a “Stockpeers-like” pattern:

- `?symbols=AAPL,MSFT,NVDA`
- `?as_of=latest`
- `?expiry=2026-04-17`

Streamlit supports query params (`st.query_params`) for shareable state.
Use:
- read on startup
- update when the user changes the selection

This makes “bookmarkable research” possible.

---

## Dependency management (recommended: optional extras)

Add optional extras to `pyproject.toml`:

- `ui` extra:
  - `streamlit`
  - charting library (Altair or Plotly)
  - `pandas` (already present)
  - any UI utilities you want (e.g., `watchdog` for reload on some OS)

- `orchestrator` extra (if Dagster):
  - `dagster`
  - `dagster-webserver`

This keeps core CLI installs lightweight.

---

## Running locally

Once you add the app scaffolding:

```bash
pip install -e ".[dev,alpaca,ui]"
streamlit run apps/streamlit/streamlit_app.py
```

Theme:
- ensure `.streamlit/config.toml` is discoverable from the working directory.

---

## “Read-only portal” principle

Avoid side effects from the portal:
- No ingestion writes from Streamlit (except “trigger run” buttons that shell out are okay, but keep that for later).
- Portal should not mutate the portfolio JSON initially.
- First ship a stable read-only UI, then add “actions” carefully.
