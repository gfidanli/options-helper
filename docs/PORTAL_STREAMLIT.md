# Streamlit Portal (Read-only)

Status: Implemented

The portal is a read-only interface for operational visibility and research. It is informational/educational only and **not financial advice**.

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

## Read-only behavior

Portal pages do not run ingestion and do not mutate portfolio or data artifacts.
Any command examples shown in Health are copy/paste suggestions only.

## DuckDB path behavior

- CLI `db` commands default to: `data/warehouse/options.duckdb`
- Streamlit page helpers currently default to: `data/options_helper.duckdb` when the sidebar path is left blank

For consistency with CLI defaults, set the DuckDB path explicitly in the page sidebar when needed.

## Health page disclaimer

The Health page reports run status, failures, stale data signals, and quality-check signals. These are operational indicators only and **not trading recommendations or financial advice**.

## Related docs

- Observability runtime and `db health`: `docs/OBSERVABILITY.md`
- Optional Dagster asset graph/checks: `docs/DAGSTER_OPTIONAL.md`
