# Next Stage: Visibility + Portal + Orchestration

Status: Implemented foundation (M1-M7 code complete; docs aligned in T16)

This project now ships a DuckDB-backed visibility control plane, a read-only Streamlit portal, and optional Dagster orchestration, while keeping the CLI-first workflow.

This tool is informational/educational only and is not financial advice.

## Delivered user-facing surfaces

### CLI

- `options-helper db health`
  - operational health summary from DuckDB `meta.*`
  - human and JSON output modes
  - graceful handling when DB or `meta.*` tables are missing
- `options-helper ui` / `options-helper ui run`
  - launches `apps/streamlit/streamlit_app.py`
  - optional dependency path with clear install guidance

### Streamlit portal

Implemented pages:

- `01 Health`
- `02 Portfolio`
- `03 Symbol Explorer`
- `04 Flow`
- `05 Derived History`
- `06 Data Explorer`

Portal is read-only and does not run ingestion jobs.

### Optional Dagster

Daily partitioned asset graph + checks in `apps/dagster/defs`:

- assets: `candles_daily` -> `options_bars` -> `options_snapshot_file` -> `options_flow` -> `derived_metrics` -> `briefing_markdown`
- checks: `*_quality` checks plus `briefing_markdown_nonempty`
- job/schedule: `daily_visibility_job`, `daily_visibility_schedule`

## Control-plane data model

All health surfaces read from DuckDB `meta.*` tables:

- `meta.ingestion_runs`
- `meta.ingestion_run_assets`
- `meta.asset_watermarks`
- `meta.asset_checks`

Producer commands and optional Dagster both write to this shared ledger.

## Important implementation notes

- Storage mode behavior:
  - DuckDB storage writes observability rows
  - Filesystem storage uses `NoopRunLogger` (warning + no ledger writes)
- Some asset-key names differ between CLI and Dagster (`option_bars` vs `options_bars`, `options_snapshots` vs `options_snapshot_file`, `derived_daily` vs `derived_metrics`)
- Streamlit helper defaults and CLI default DuckDB paths are different unless explicitly set; use explicit sidebar paths for consistency.

## Current source plan

Execution plan and task status:

- `docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`
