# Dagster Orchestration (Optional)

Status: Implemented (optional extra)

Dagster support is optional. The CLI remains the primary interface.

## Install and run

```bash
pip install -e ".[dev,orchestrator]"
dagster dev -m apps.dagster.defs
```

Definitions entrypoint: `apps/dagster/defs/__init__.py` (`defs = build_definitions()`).

## What Dagster includes

### Daily partitioned assets

Asset group: `daily_visibility`

Asset order:

1. `candles_daily`
2. `options_bars`
3. `options_snapshot_file`
4. `options_flow`
5. `derived_metrics`
6. `briefing_markdown`

Partitioning:

- `DailyPartitionsDefinition`
- start date defaults to `2026-01-01`
- override with `OPTIONS_HELPER_DAGSTER_PARTITION_START`

### Asset checks

Defined checks:

- `candles_daily_quality`
- `options_bars_quality`
- `options_snapshot_file_quality`
- `options_flow_quality`
- `derived_metrics_quality`
- `briefing_markdown_nonempty`

### Job and schedule

- Job: `daily_visibility_job`
- Schedule: `daily_visibility_schedule`

## Shared observability contract

Dagster runs write into the same DuckDB control-plane tables as CLI runs:

- `meta.ingestion_runs`
- `meta.ingestion_run_assets`
- `meta.asset_watermarks`
- `meta.asset_checks`

Runtime details:

- `triggered_by='dagster'`
- Dagster run id is persisted as `parent_run_id`
- asset checks also persist rows to `meta.asset_checks` (not only Dagster UI)

This keeps Health views unified across CLI, cron, and Dagster.

## Resource/config environment variables

`apps/dagster/defs/resources.py` reads:

- `OPTIONS_HELPER_DATA_DIR`
- `OPTIONS_HELPER_PORTFOLIO_PATH`
- `OPTIONS_HELPER_WATCHLISTS_PATH`
- `OPTIONS_HELPER_DUCKDB_PATH`
- `OPTIONS_HELPER_PROVIDER` (default `alpaca`)

## Notes

- Dagster is optional by design.
- You can run the full project with CLI + DuckDB only and still get run ledger/health visibility.
