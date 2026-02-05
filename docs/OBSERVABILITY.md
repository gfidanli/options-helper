# Observability and Health (`db health`)

Status: Implemented

This observability layer is for operational visibility only. It is informational/educational and **not financial advice**.

## What is tracked

When producer jobs run on DuckDB storage, `options-helper` persists run metadata in DuckDB `meta.*` tables:

- `meta.ingestion_runs`: one row per run (status, timing, provider, args, error fields)
- `meta.ingestion_run_assets`: per-asset success/fail/skipped events and best-effort metrics
- `meta.asset_watermarks`: freshness watermarks (`asset_key`, `scope_key`)
- `meta.asset_checks`: persisted quality check results (`pass`/`fail`/`skip`)

The logger is storage-aware:

- `--storage duckdb`: uses `DuckDBRunLogger` and writes observability rows
- `--storage filesystem`: uses `NoopRunLogger` (commands still run; ledger writes are disabled)

## Instrumented producer jobs

Stable CLI job names recorded in `meta.ingestion_runs`:

- `ingest_candles`
- `ingest_options_bars`
- `snapshot_options`
- `compute_flow`
- `compute_derived`
- `build_briefing`
- `build_dashboard`

## Quality checks

Checks are persisted even when they fail (non-blocking unless a runtime exception occurs).
Current check names include:

- Candles: `candles_unique_symbol_date`, `candles_monotonic_date`, `candles_no_negative_prices`, `candles_gap_days_last_30`
- Option bars: `options_bars_monotonic_ts`, `options_bars_no_negative_prices`, `options_bars_duplicate_pk`
- Snapshots: `snapshot_parseable_contract_symbol`
- Flow: `flow_no_null_primary_keys`
- Derived: `derived_no_duplicate_keys`

## `options-helper db health`

Command:

```bash
./.venv/bin/options-helper db health
```

Useful flags:

- `--days`: lookback for latest runs/failures/failed checks (default `7`)
- `--limit`: max rows per section (default `50`)
- `--stale-days`: only include watermarks at/above this staleness
- `--json`: machine-readable output
- `--duckdb-path`: target DB path (defaults to `data/warehouse/options.duckdb`)

Human output sections:

- Latest run per job
- Recent run failures
- Watermarks and freshness
- Recent failed checks

JSON output includes:

- database metadata (`database_exists`, `meta_tables_present`)
- applied filters (`days`, `limit`, `stale_days`)
- the same four data sections as arrays

Graceful behavior:

- If the DB file does not exist, output is empty with guidance.
- If `meta.*` tables are missing, output is empty with guidance.

## Asset key note

Today, CLI jobs and optional Dagster assets use slightly different asset-key names in some places (for example `option_bars` vs `options_bars`, `options_snapshots` vs `options_snapshot_file`, `derived_daily` vs `derived_metrics`).
Health views show stored keys as written.

## Related docs

- Portal health page and gap planner: `docs/PORTAL_STREAMLIT.md`
- Optional orchestration and persisted checks: `docs/DAGSTER_OPTIONAL.md`
