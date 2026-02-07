# Streamlit Portal (Read-only)

Status: Implemented

The portal is a read-only interface for operational visibility and research. It is informational/educational only and **not financial advice**.

## Architecture diagram

![Streamlit portal architecture](assets/diagrams/generated/portal_readonly_architecture.svg)

## Install and launch

Install optional UI dependencies:

```bash
pip install -e ".[dev,ui]"
```

Launch from CLI:

```bash
./.venv/bin/options-helper ui
```

Equivalent subcommand:

```bash
./.venv/bin/options-helper ui run
```

Launcher flags:

- `--host` (default `127.0.0.1`)
- `--port` (default `8501`)
- `--path` (default `apps/streamlit/streamlit_app.py`)

If Streamlit is missing, the command exits with install guidance (`pip install -e ".[ui]"`).

## Portal pages

Implemented pages under `apps/streamlit/pages/`:

- `01_Health.py`
  - latest runs, recurring failure groups (stack hash), watermarks/freshness, failed checks
  - deterministic gap backfill planner (display-only; no command execution)
- `02_Portfolio.py`
  - positions table + best-effort risk summary
  - uses latest briefing artifact when available, computed fallback otherwise
- `03_Symbol_Explorer.py`
  - candles history, latest snapshot summary/chain breakdown, derived snippet
  - query param sync for `symbol`
- `04_Flow.py`
  - persisted `options_flow` partition summaries and row drill-down
  - query param sync for `symbol` and `group_by`
- `05_Derived_History.py`
  - derived history windows and regime metrics from `derived_daily`
- `06_Data_Explorer.py`
  - schema/table browser with preview rows and SQL snippet helper
- `07_Market_Analysis.py`
  - Monte Carlo tail-risk fan chart + horizon percentile table
  - move-percentile calculator and IV regime context from `derived_daily`
  - persisted research-metrics tabs from DuckDB:
    - IV surface tenor + delta-bucket sparklines and latest rows
    - dealer exposure by strike with net-GEX bar chart and flip marker
    - intraday flow summary with top strikes/contracts (when persisted)
- `08_Coverage.py`
  - universe and symbol-level data coverage views from persisted quality checks
  - latest completeness/freshness indicators for candles/options snapshots
- `09_SFP.py`
  - daily/weekly swing failure pattern (SFP) research summaries and event tables
  - extension-percentile and RSI context for each SFP event
- `10_MSB.py`
  - daily/weekly market structure break (MSB) research summaries and event tables
  - close-through break events against prior swings with extension/RSI context

## Read-only behavior

Portal pages do not mutate portfolio state.
Most pages are read-only and do not run ingestion.

Exception:
- `03_Symbol_Explorer.py` is DuckDB-first and can run a best-effort on-demand Alpaca sync for the selected symbol when `options_snapshot_headers` or `derived_daily` rows are missing in DuckDB.

## DuckDB path behavior

- CLI `db` commands default to: `data/warehouse/options.duckdb`
- Streamlit page helpers resolve DuckDB path in this order:
  1) sidebar `DuckDB path` input (if provided),
  2) `OPTIONS_HELPER_DUCKDB_PATH` environment variable (if set),
  3) fallback `data/warehouse/options.duckdb`.

Set once for your shell/session:

```bash
export OPTIONS_HELPER_DUCKDB_PATH=/Volumes/develop/options-helper/data/warehouse/options.duckdb
```

You can still set an explicit DuckDB path in the page sidebar when needed.

## Health page disclaimer

The Health page reports run status, failures, stale data signals, and quality-check signals. These are operational indicators only and **not trading recommendations or financial advice**.

## Related docs

- Observability runtime and `db health`: `docs/OBSERVABILITY.md`
- Optional Dagster asset graph/checks: `docs/DAGSTER_OPTIONAL.md`
