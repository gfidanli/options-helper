# apps/dagster/ — Optional Orchestrator Conventions

## Scope
- Dagster integration is optional and must not be required for base CLI usage.
- Keep Dagster code isolated under `apps/dagster/defs/`.

## Integration contract
- Reuse `options_helper` pipeline/service functions for execution.
- Persist observability data through the same DuckDB `meta.*` ledger used by CLI runs.
- Dagster-triggered run ledger rows must set `triggered_by='dagster'` and include Dagster `run_id` as `parent_run_id`.
- Persist Dagster asset-check outcomes into `meta.asset_checks` so portal health views work without Dagster UI.
- Keep stable asset/job naming once introduced (treat as public contract).

## Dependency and startup
- Avoid importing Dagster modules from core CLI code paths.
- Guard tests and runtime checks with `pytest.importorskip("dagster")` where appropriate.
