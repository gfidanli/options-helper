# Plan: Visibility Control Plane + Portal + Optional Dagster (Full M1–M7)

**Generated**: February 5, 2026

## Summary
The latest commit (`b4a17fd`) added design docs for observability, Streamlit portal, and optional Dagster orchestration, but no implementation code.  
This plan implements the full feature set end-to-end with a DuckDB-backed operational control plane (`meta.*`), a read-only Streamlit portal (including Health + gap planner), a `db health` CLI surface, a `options-helper ui` launcher, and a full optional Dagster daily asset graph that writes to the same ledger.

Implementation plan file target (when Plan Mode ends and file writes are allowed): `/Volumes/develop/options-helper/visibility-portal-plan.md`.

## Public API / Interface Changes
- New CLI command: `options-helper ui` (Streamlit launcher).
- New CLI command: `options-helper db health`.
- `flow` command additions:
  - `--as-of` (default `latest`) for partition-aware/deterministic runs.
  - `--persist/--no-persist` (default persist on DuckDB backend).
- New optional dependency extras in `/Volumes/develop/options-helper/pyproject.toml`:
  - `ui`: `streamlit>=1.44.2`, `altair>=5`
  - `orchestrator`: `dagster`, `dagster-webserver`
- New DuckDB schema migration v3:
  - `meta.ingestion_runs`
  - `meta.ingestion_run_assets`
  - `meta.asset_watermarks`
  - `meta.asset_checks`
  - `options_flow`
- New service/runtime interfaces:
  - Observability run logger protocol and implementations (`DuckDB` + `Noop`).
  - Pipeline service functions for CLI + Dagster shared execution.
  - Flow store abstraction for persisted flow rows.
- New app entrypoints:
  - `/Volumes/develop/options-helper/apps/streamlit/streamlit_app.py`
  - `/Volumes/develop/options-helper/apps/dagster/defs/__init__.py`

## Dependency Graph
```text
T1 ──┬── T9 ──┬── T10 ──┐
     │        ├── T11   ├── T16 ──┐
     │        └── T12 ──┘         │
T2 ──┬── T3 ──┬── T6 ──┬── T7 ──┬── T8 ──┐
     │        │        │        └── T15  │
     │        │        └── T13            │
     │        └── T4 ───────┬─────────────┘
     └───────────────┬──────┘
T5 ──────────────────┴── T6 ────────┐
T14 ────────────────────────────┬───┘
                                └── T15
T16 + T17 ──────────────────────────┬── T18
```

## Tasks

### T1: Add Optional Dependency Extras and Lazy Import Boundaries
- **depends_on**: []
- **location**: `/Volumes/develop/options-helper/pyproject.toml`, `/Volumes/develop/options-helper/options_helper/cli.py`, `/Volumes/develop/options-helper/options_helper/commands/ui.py` (new)
- **description**: Add `ui` and `orchestrator` extras; ensure CLI help/startup does not import Streamlit or Dagster modules unless command is invoked.
- **validation**: `./.venv/bin/python -m pytest tests/test_cli_contract.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T2: Create DuckDB Migration v3 (Observability + Flow Table)
- **depends_on**: []
- **location**: `/Volumes/develop/options-helper/options_helper/db/schema_v3.sql` (new), `/Volumes/develop/options-helper/options_helper/db/migrations.py`
- **description**: Add idempotent v3 migration with `meta.*` tables and `options_flow`; add indexes for run/job/time lookups and health queries; keep `schema_migrations` as canonical version source; ensure repeated init is safe.
- **validation**: `./.venv/bin/python -m pytest tests/test_duckdb_migrations.py tests/test_duckdb_migrations_v2.py tests/test_duckdb_cli_db.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3: Implement Observability Runtime (DuckDB + Noop)
- **depends_on**: [T2]
- **location**: `/Volumes/develop/options-helper/options_helper/data/observability_meta.py` (new), `/Volumes/develop/options-helper/options_helper/cli_deps.py`
- **description**: Implement `RunLogger` protocol with `DuckDBRunLogger` and `NoopRunLogger`; include start/finalize lifecycle, asset event logging, watermark upserts, check persistence, stack hashing, args JSON serialization, and centralized health query helpers; add `cli_deps.build_run_logger(...)` seam.
- **validation**: `./.venv/bin/python -m pytest tests/test_observability_meta.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4: Add Flow Persistence Store
- **depends_on**: [T2]
- **location**: `/Volumes/develop/options-helper/options_helper/data/flow_store.py` (new), `/Volumes/develop/options-helper/options_helper/data/store_factory.py`, `/Volumes/develop/options-helper/options_helper/cli_deps.py`
- **description**: Add `DuckDBFlowStore` with partition upsert semantics by `(symbol, from_date, to_date, window, group_by)` and read APIs for portal pages; provide no-op behavior for filesystem backend.
- **validation**: `./.venv/bin/python -m pytest tests/test_duckdb_flow_store.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5: Extract Shared Pipeline Service Functions
- **depends_on**: []
- **location**: `/Volumes/develop/options-helper/options_helper/pipelines/visibility_jobs.py` (new), `/Volumes/develop/options-helper/options_helper/commands/ingest.py`, `/Volumes/develop/options-helper/options_helper/commands/workflows.py`, `/Volumes/develop/options-helper/options_helper/commands/reports.py`, `/Volumes/develop/options-helper/options_helper/commands/derived.py`
- **description**: Move command bodies into reusable service functions returning structured results; CLI commands become thin wrappers for arguments/rendering; this is the shared execution layer for Dagster.
- **validation**: Existing command tests for ingest/snapshot/flow/briefing/dashboard/derived remain green.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6: Instrument Producer Commands with Run Ledger + Asset Events
- **depends_on**: [T3, T5]
- **location**: `/Volumes/develop/options-helper/options_helper/commands/ingest.py`, `/Volumes/develop/options-helper/options_helper/commands/workflows.py`, `/Volumes/develop/options-helper/options_helper/commands/reports.py`, `/Volumes/develop/options-helper/options_helper/commands/derived.py`
- **description**: Wrap producer jobs with logger context and stable job names:
  - `ingest_candles`
  - `ingest_options_bars`
  - `snapshot_options`
  - `compute_flow`
  - `compute_derived`
  - `build_briefing`
  - `build_dashboard`
  Log per-asset success/fail/skipped records and watermarks. Use `NoopRunLogger` for filesystem mode with one warning and no failures.
- **validation**: `./.venv/bin/python -m pytest tests/test_ingest_candles_command.py tests/test_ingest_options_bars_command.py tests/test_snapshot_options_full.py tests/test_flow_artifact_cli.py tests/test_derived_cli.py tests/test_briefing_cli.py tests/test_dashboard_cli.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7: Implement Non-Blocking Data Quality Checks and Auto-Run
- **depends_on**: [T3, T4, T6]
- **location**: `/Volumes/develop/options-helper/options_helper/data/quality_checks.py` (new), `/Volumes/develop/options-helper/options_helper/pipelines/visibility_jobs.py`
- **description**: Add persisted checks with `check_name`, `severity`, `status`, `metrics_json`, `message`, `scope_key`; run at end of producer jobs and persist all results. Check failures do not fail the run unless runtime exceptions occur.
  Required checks:
  - candles: uniqueness, monotonic dates, no negative prices, gap-days-last-30
  - options bars: monotonic ts, no negative prices, duplicate PK
  - snapshots/flow/derived: parseable contract symbol, flow PK null guard, derived duplicate guard
- **validation**: `./.venv/bin/python -m pytest tests/test_quality_checks.py tests/test_observability_checks_integration.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: Add `db health` CLI Surface
- **depends_on**: [T3, T7]
- **location**: `/Volumes/develop/options-helper/options_helper/commands/db.py`
- **description**: Add `db health` command with `--days`, `--limit`, `--stale-days`, `--json`; show latest run per job, recent failures, watermarks/freshness, and recent failed checks; graceful output when `meta.*` is absent.
- **validation**: `./.venv/bin/python -m pytest tests/test_db_health_cli.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9: Streamlit Portal Scaffold + Shared Data Access
- **depends_on**: [T1, T2]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/streamlit_app.py` (new), `/Volumes/develop/options-helper/apps/streamlit/components/db.py` (new), `/Volumes/develop/options-helper/apps/streamlit/components/queries.py` (new), `/Volumes/develop/options-helper/apps/streamlit/components/gap_planner.py` (new), `/Volumes/develop/options-helper/apps/streamlit/pages/`
- **description**: Create multipage app skeleton with `st.set_page_config`, disclaimer banner, read-only DuckDB connection via `st.cache_resource`, query cache via `st.cache_data`, and query-param synchronization for shareable state.
- **validation**: Streamlit app imports cleanly with `ui` extra installed; smoke test for page module import.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T10: Implement Health Page + Gap Backfill Planner
- **depends_on**: [T7, T9]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/pages/01_Health.py`, `/Volumes/develop/options-helper/apps/streamlit/components/gap_planner.py`
- **description**: Build Health page using centralized observability query helpers: latest runs timeline, grouped recurring failures, watermark freshness, recent failed checks, and deterministic backfill command suggestions (display-only).
- **validation**: `./.venv/bin/python -m pytest tests/portal/test_health_queries.py tests/portal/test_gap_planner.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T11: Implement Portfolio + Symbol Explorer Pages
- **depends_on**: [T9]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/pages/02_Portfolio.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/03_Symbol_Explorer.py`
- **description**: Portfolio page loads portfolio JSON and best-effort risk summary (briefing artifact fallback + live computed fallback). Symbol Explorer page provides symbol selector/query params, candles chart, latest snapshot summary (OI/IV views), and derived-history snippets.
- **validation**: `./.venv/bin/python -m pytest tests/portal/test_portfolio_page_queries.py tests/portal/test_symbol_explorer_queries.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T12: Implement Flow, Derived History, and Data Explorer Pages
- **depends_on**: [T4, T9]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/pages/04_Flow.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/05_Derived_History.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/06_Data_Explorer.py`
- **description**: Add interactive Flow page from `options_flow`; Derived page from `derived_daily`; Data Explorer for schema/table/preview browsing. Keep portal read-only.
- **validation**: `./.venv/bin/python -m pytest tests/portal/test_flow_queries.py tests/portal/test_derived_queries.py tests/portal/test_data_explorer_queries.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T13: Add `options-helper ui` Launcher Command
- **depends_on**: [T1, T9]
- **location**: `/Volumes/develop/options-helper/options_helper/commands/ui.py` (new), `/Volumes/develop/options-helper/options_helper/cli.py`
- **description**: Add thin launcher that shells out to `python -m streamlit run /Volumes/develop/options-helper/apps/streamlit/streamlit_app.py`; include host/port/path flags; provide install guidance when Streamlit extra is missing.
- **validation**: `./.venv/bin/python -m pytest tests/test_ui_cli.py`
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T14: Add Dagster Definitions Scaffold
- **depends_on**: [T1]
- **location**: `/Volumes/develop/options-helper/apps/dagster/defs/__init__.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/resources.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/jobs.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/schedules.py` (new)
- **description**: Create optional Dagster package layout and resource config wiring for paths/provider/portfolio/watchlists without impacting core CLI startup.
- **validation**: `pytest.importorskip("dagster")` defs import smoke test.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T15: Implement Full Dagster Daily Asset Graph + Checks
- **depends_on**: [T2, T3, T4, T5, T7, T14]
- **location**: `/Volumes/develop/options-helper/apps/dagster/defs/assets.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/checks.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/jobs.py`, `/Volumes/develop/options-helper/apps/dagster/defs/schedules.py`
- **description**: Implement daily-partitioned assets in order:
  - `candles_daily`
  - `options_bars`
  - `options_snapshot_file`
  - `options_flow`
  - `derived_metrics`
  - `briefing_markdown`
  Reuse pipeline service layer; persist run ledger with `triggered_by='dagster'` and parent Dagster run id; include Dagster asset checks that also write to `meta.asset_checks`.
- **validation**: `pytest.importorskip("dagster")` tests for Definitions load, dependency order, and one partitioned materialization with mocked providers.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T16: Documentation and MkDocs Navigation Updates
- **depends_on**: [T8, T10, T11, T12, T13, T15]
- **location**: `/Volumes/develop/options-helper/README.md`, `/Volumes/develop/options-helper/mkdocs.yml`, `/Volumes/develop/options-helper/docs/OBSERVABILITY.md`, `/Volumes/develop/options-helper/docs/PORTAL_STREAMLIT.md`, `/Volumes/develop/options-helper/docs/DAGSTER_OPTIONAL.md`, `/Volumes/develop/options-helper/docs/NEXT_STAGE_VISIBILITY_PORTAL.md`, `/Volumes/develop/options-helper/docs/MKDOCS_NAV_PATCH.md`
- **description**: Update docs to match actual behavior/commands; wire new docs into nav; include explicit “not financial advice” disclaimer in portal and health docs.
- **validation**: `mkdocs build` and docs-link sanity checks.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T17: Add/Update Deterministic Offline Tests
- **depends_on**: [T6, T7, T8, T10, T11, T12, T13, T15]
- **location**: `/Volumes/develop/options-helper/tests/`
- **description**: Add regression and feature tests:
  - migration v2→v3 upgrade path
  - command-level run ledger + asset rows + watermarks
  - non-blocking check failures
  - filesystem no-op observability mode
  - `db health` output
  - flow persistence table
  - UI launcher behavior
  - Streamlit query/helper tests
  - Dagster defs/asset smoke tests (skipped when extra absent)
- **validation**: targeted pytest matrix and full suite.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T18: End-to-End Validation and Release Gate
- **depends_on**: [T16, T17]
- **location**: `/Volumes/develop/options-helper`
- **description**: Run full validation, confirm no regressions in existing CLI commands, and verify portal + health + Dagster paths in local dry run.
- **validation**:
  - `./.venv/bin/python -m pytest`
  - `./.venv/bin/options-helper db init`
  - `./.venv/bin/options-helper db health`
  - `./.venv/bin/options-helper ui --help`
  - `streamlit run /Volumes/develop/options-helper/apps/streamlit/streamlit_app.py` (with `.[ui]`)
  - `dagster dev -m apps.dagster.defs` (with `.[orchestrator]`)
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|---|---|---|
| 1 | T1, T2, T5 | Immediately |
| 2 | T3, T4, T9, T14 | T1 + T2 complete (T14 needs T1 only) |
| 3 | T6, T13 | T3 + T5 complete (T13 also needs T9) |
| 4 | T7, T8, T10, T11, T12, T15 | Prerequisites per task satisfied |
| 5 | T16, T17 | Feature tasks complete |
| 6 | T18 | T16 + T17 complete |

## Testing Strategy
- Preserve all existing deterministic CLI tests and expand with observability assertions.
- Validate migration paths:
  - fresh DB init to v3
  - existing v2 DB upgrade to v3
  - idempotent repeated migrations
- Validate command instrumentation by invoking real Typer commands with monkeypatched providers/stores.
- Validate no-op observability path in filesystem backend.
- Validate portal logic mostly via pure helper/query tests; keep Streamlit import smoke minimal.
- Validate Dagster only when optional dependency is installed (`importorskip`).

## Risks and Mitigations
- Risk: migration/observability write conflicts with concurrent readers.
  - Mitigation: short transactions, read-only portal connections, no portal-triggered migrations.
- Risk: command behavior regressions from service extraction.
  - Mitigation: wrap extraction with unchanged command tests and golden output tests.
- Risk: optional dependencies breaking base install.
  - Mitigation: extras-only deps + lazy imports + explicit install guidance.
- Risk: checks causing noisy failures.
  - Mitigation: checks are non-blocking; persisted separately from run success/failure.
- Risk: flow history gaps reduce dashboard value.
  - Mitigation: gap planner suggests upstream backfill order (candles/snapshot before flow).

## Assumptions and Defaults (Locked)
- Full scope M1–M7 is in this implementation pass.
- Full Dagster daily asset graph is included.
- `options-helper ui` launcher is included.
- Checks run automatically after each producing job and are non-blocking.
- Filesystem backend uses no-op observability writes plus a clear warning.
- `db health` is implemented under the existing `db` command group.
- Portal is read-only and always includes “not financial advice” messaging.
- Asset and job names are centralized constants and treated as stable contract identifiers.
- Watermark timestamps are based on data period/event timestamps, not wall-clock run time.
