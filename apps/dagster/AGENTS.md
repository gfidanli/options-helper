# apps/dagster/ — Optional Orchestrator Conventions

## Scope
- Dagster integration is optional and must not be required for base CLI usage.
- Keep Dagster code isolated under `apps/dagster/defs/`.

## Integration contract
- Reuse `options_helper` pipeline/service functions for execution.
- Persist observability data through the same DuckDB `meta.*` ledger used by CLI runs.
- Keep stable asset/job naming once introduced (treat as public contract).

## Dependency and startup
- Avoid importing Dagster modules from core CLI code paths.
- Guard tests and runtime checks with `pytest.importorskip("dagster")` where appropriate.
