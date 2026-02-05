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
- **status**: Completed
- **log**: Added `ui` + `orchestrator` extras, wired a lightweight `ui` CLI group with no optional-dependency imports at module import time, and added CLI contract tests asserting `streamlit`/`dagster` are not imported during `--help` or `ui --help`.
- **files edited/created**: `/Volumes/develop/options-helper/pyproject.toml`, `/Volumes/develop/options-helper/options_helper/cli.py`, `/Volumes/develop/options-helper/options_helper/commands/ui.py`, `/Volumes/develop/options-helper/tests/test_cli_contract.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T2: Create DuckDB Migration v3 (Observability + Flow Table)
- **depends_on**: []
- **location**: `/Volumes/develop/options-helper/options_helper/db/schema_v3.sql` (new), `/Volumes/develop/options-helper/options_helper/db/migrations.py`
- **description**: Add idempotent v3 migration with `meta.*` tables and `options_flow`; add indexes for run/job/time lookups and health queries; keep `schema_migrations` as canonical version source; ensure repeated init is safe.
- **validation**: `./.venv/bin/python -m pytest tests/test_duckdb_migrations.py tests/test_duckdb_migrations_v2.py tests/test_duckdb_cli_db.py`
- **status**: Completed
- **log**:
  - Added `schema_v3.sql` with idempotent DDL for `meta.ingestion_runs`, `meta.ingestion_run_assets`, `meta.asset_watermarks`, `meta.asset_checks`, and `options_flow`.
  - Added v3 index set for run/job/time lookup and health-query paths (`meta.ingestion_runs`, `meta.ingestion_run_assets`, `meta.asset_checks`, `meta.asset_watermarks`, `options_flow`).
  - Updated `ensure_schema(...)` to apply v3 and record migration row `schema_version=3` idempotently using `schema_migrations` as source of truth.
  - Extended migration tests for fresh v3 init, simulated v2→v3 upgrade path, and repeated idempotent migration calls.
- **files edited/created**:
  - `/Volumes/develop/options-helper/options_helper/db/schema_v3.sql` (new)
  - `/Volumes/develop/options-helper/options_helper/db/migrations.py`
  - `/Volumes/develop/options-helper/tests/test_duckdb_migrations.py`
  - `/Volumes/develop/options-helper/tests/test_duckdb_migrations_v2.py`
  - `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T3: Implement Observability Runtime (DuckDB + Noop)
- **depends_on**: [T2]
- **location**: `/Volumes/develop/options-helper/options_helper/data/observability_meta.py` (new), `/Volumes/develop/options-helper/options_helper/cli_deps.py`
- **description**: Implement `RunLogger` protocol with `DuckDBRunLogger` and `NoopRunLogger`; include start/finalize lifecycle, asset event logging, watermark upserts, check persistence, stack hashing, args JSON serialization, and centralized health query helpers; add `cli_deps.build_run_logger(...)` seam.
- **validation**: `./.venv/bin/python -m pytest tests/test_observability_meta.py`
- **status**: Completed
- **log**: Added `options_helper.data.observability_meta` with a `RunLogger` protocol, `DuckDBRunLogger` + `NoopRunLogger`, run lifecycle start/success/failure finalization, per-asset upserted event rows, watermark upserts, persisted checks, stack hashing, args JSON serialization, and shared health query helpers (`latest runs`, `recent failures`, `watermarks`, `check failures`). Added storage-aware `cli_deps.build_run_logger(...)` that returns a started DuckDB logger in DuckDB mode and a started no-op logger in filesystem mode. Added deterministic offline tests covering runtime writes, failure hashing, query helpers, no-op behavior, and CLI deps seam.
- **files edited/created**:
  - `/Volumes/develop/options-helper/options_helper/data/observability_meta.py` (new)
  - `/Volumes/develop/options-helper/options_helper/cli_deps.py`
  - `/Volumes/develop/options-helper/tests/test_observability_meta.py` (new)
  - `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T4: Add Flow Persistence Store
- **depends_on**: [T2]
- **location**: `/Volumes/develop/options-helper/options_helper/data/flow_store.py` (new), `/Volumes/develop/options-helper/options_helper/data/store_factory.py`, `/Volumes/develop/options-helper/options_helper/cli_deps.py`
- **description**: Add `DuckDBFlowStore` with partition upsert semantics by `(symbol, from_date, to_date, window, group_by)` and read APIs for portal pages; provide no-op behavior for filesystem backend.
- **validation**: `./.venv/bin/python -m pytest tests/test_duckdb_flow_store.py`
- **status**: Completed
- **log**: Added `options_helper.data.flow_store` with `FlowStore` protocol, DuckDB-backed partition replace/upsert writes into `options_flow`, artifact/row/partition read APIs, and `NoopFlowStore` for filesystem mode. Wired stable builder seams through store factory and CLI deps, and added deterministic DuckDB + filesystem backend tests.
- **files edited/created**: `/Volumes/develop/options-helper/options_helper/data/flow_store.py`, `/Volumes/develop/options-helper/options_helper/data/store_factory.py`, `/Volumes/develop/options-helper/options_helper/cli_deps.py`, `/Volumes/develop/options-helper/tests/test_duckdb_flow_store.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T5: Extract Shared Pipeline Service Functions
- **depends_on**: []
- **location**: `/Volumes/develop/options-helper/options_helper/pipelines/visibility_jobs.py` (new), `/Volumes/develop/options-helper/options_helper/commands/ingest.py`, `/Volumes/develop/options-helper/options_helper/commands/workflows.py`, `/Volumes/develop/options-helper/options_helper/commands/reports.py`, `/Volumes/develop/options-helper/options_helper/commands/derived.py`
- **description**: Move command bodies into reusable service functions returning structured results; CLI commands become thin wrappers for arguments/rendering; this is the shared execution layer for Dagster.
- **validation**: Existing command tests for ingest/snapshot/flow/briefing/dashboard/derived remain green.
- **status**: Completed
- **log**: Added shared pipeline services in `options_helper/pipelines/visibility_jobs.py` and rewired producer command entrypoints (`ingest`, `snapshot-options`, `flow`, `briefing`, `dashboard`, `derived update`) to call reusable service functions while preserving CLI output semantics. Added filesystem-compatible snapshot/derived/candle fallbacks in the service layer to maintain existing offline artifact behavior and command test compatibility.
- **files edited/created**: `/Volumes/develop/options-helper/options_helper/pipelines/__init__.py`, `/Volumes/develop/options-helper/options_helper/pipelines/visibility_jobs.py`, `/Volumes/develop/options-helper/options_helper/commands/ingest.py`, `/Volumes/develop/options-helper/options_helper/commands/workflows.py`, `/Volumes/develop/options-helper/options_helper/commands/reports.py`, `/Volumes/develop/options-helper/options_helper/commands/derived.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

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
- **status**: Completed
- **log**: Added run-ledger instrumentation wrappers to producer commands with stable T6 job names, explicit run success/failure finalization, per-asset success/fail/skipped events, and watermarks where available. Filesystem storage now emits a single Noop-run-ledger warning per command without crashing. Added focused command-level observability tests validating DuckDB run/asset/watermark writes, failure-run error fields, and filesystem Noop behavior.
- **files edited/created**: `/Volumes/develop/options-helper/options_helper/commands/ingest.py`, `/Volumes/develop/options-helper/options_helper/commands/workflows.py`, `/Volumes/develop/options-helper/options_helper/commands/reports.py`, `/Volumes/develop/options-helper/options_helper/commands/derived.py`, `/Volumes/develop/options-helper/tests/test_command_run_ledger_instrumentation.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T7: Implement Non-Blocking Data Quality Checks and Auto-Run
- **depends_on**: [T3, T4, T6]
- **location**: `/Volumes/develop/options-helper/options_helper/data/quality_checks.py` (new), `/Volumes/develop/options-helper/options_helper/pipelines/visibility_jobs.py`
- **description**: Add persisted checks with `check_name`, `severity`, `status`, `metrics_json`, `message`, `scope_key`; run at end of producer jobs and persist all results. Check failures do not fail the run unless runtime exceptions occur.
  Required checks:
  - candles: uniqueness, monotonic dates, no negative prices, gap-days-last-30
  - options bars: monotonic ts, no negative prices, duplicate PK
  - snapshots/flow/derived: parseable contract symbol, flow PK null guard, derived duplicate guard
- **validation**: `./.venv/bin/python -m pytest tests/test_quality_checks.py tests/test_observability_checks_integration.py`
- **status**: Completed
- **log**: Added `options_helper.data.quality_checks` with deterministic candle/options-bars/snapshot/flow/derived checks, status/severity/metrics/message payloads, and persistence helpers wired to `RunLogger.log_check(...)`. Integrated auto-run check execution into producer pipeline jobs (`ingest_candles`, `ingest_options_bars`, `snapshot_options`, `compute_flow`, `compute_derived`) via shared visibility pipeline hooks. Check failures persist as `status='fail'` without failing successful runs; runtime exceptions during check execution still fail the run. Added focused unit tests for check calculations and integration tests proving observability persistence + non-blocking behavior.
- **files edited/created**: `/Volumes/develop/options-helper/options_helper/data/quality_checks.py` (new), `/Volumes/develop/options-helper/options_helper/pipelines/visibility_jobs.py`, `/Volumes/develop/options-helper/tests/test_quality_checks.py` (new), `/Volumes/develop/options-helper/tests/test_observability_checks_integration.py` (new), `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T8: Add `db health` CLI Surface
- **depends_on**: [T3, T7]
- **location**: `/Volumes/develop/options-helper/options_helper/commands/db.py`
- **description**: Add `db health` command with `--days`, `--limit`, `--stale-days`, `--json`; show latest run per job, recent failures, watermarks/freshness, and recent failed checks; graceful output when `meta.*` is absent.
- **validation**: `./.venv/bin/python -m pytest tests/test_db_health_cli.py`
- **status**: Completed
- **log**: Added `db health` under the existing `db` CLI group with `--days`, `--limit`, `--stale-days`, and `--json`. The command reuses centralized observability query helpers to emit latest run per job, recent failed runs, watermark freshness, and recent failed checks. Added graceful behavior for missing DuckDB/meta tables in both human and JSON output modes.
- **files edited/created**: `/Volumes/develop/options-helper/options_helper/commands/db.py`, `/Volumes/develop/options-helper/tests/test_db_health_cli.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T9: Streamlit Portal Scaffold + Shared Data Access
- **depends_on**: [T1, T2]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/streamlit_app.py` (new), `/Volumes/develop/options-helper/apps/streamlit/components/db.py` (new), `/Volumes/develop/options-helper/apps/streamlit/components/queries.py` (new), `/Volumes/develop/options-helper/apps/streamlit/components/gap_planner.py` (new), `/Volumes/develop/options-helper/apps/streamlit/pages/`
- **description**: Create multipage app skeleton with `st.set_page_config`, disclaimer banner, read-only DuckDB connection via `st.cache_resource`, query cache via `st.cache_data`, and query-param synchronization for shareable state.
- **validation**: Streamlit app imports cleanly with `ui` extra installed; smoke test for page module import.
- **status**: Completed
- **log**: Added a multipage Streamlit portal scaffold with landing navigation/disclaimer, read-only DuckDB + cached query helpers, minimal query-param sync utilities, six placeholder page scripts (`01`..`06`), and targeted portal smoke tests (imports/page execution plus read-only helper behavior with Streamlit-optional skips).
- **files edited/created**: `/Volumes/develop/options-helper/apps/streamlit/streamlit_app.py`, `/Volumes/develop/options-helper/apps/streamlit/components/db.py`, `/Volumes/develop/options-helper/apps/streamlit/components/queries.py`, `/Volumes/develop/options-helper/apps/streamlit/components/gap_planner.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/01_Health.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/02_Portfolio.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/03_Symbol_Explorer.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/04_Flow.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/05_Derived_History.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/06_Data_Explorer.py`, `/Volumes/develop/options-helper/tests/portal/test_streamlit_scaffold.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T10: Implement Health Page + Gap Backfill Planner
- **depends_on**: [T7, T9]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/pages/01_Health.py`, `/Volumes/develop/options-helper/apps/streamlit/components/gap_planner.py`
- **description**: Build Health page using centralized observability query helpers: latest runs timeline, grouped recurring failures, watermark freshness, recent failed checks, and deterministic backfill command suggestions (display-only).
- **validation**: `./.venv/bin/python -m pytest tests/portal/test_health_queries.py tests/portal/test_gap_planner.py`
- **status**: Completed
- **log**: Implemented a full read-only Health page that loads centralized observability data through shared helpers (latest runs, recurring failure aggregation by stack hash, watermark freshness, recent failed checks), includes resilient missing DB/meta guidance, and renders a deterministic display-only gap backfill planner with dependency-aware command ordering. Added offline deterministic portal tests for health query loading/normalization and gap planner command generation/order.
- **files edited/created**: `/Volumes/develop/options-helper/apps/streamlit/pages/01_Health.py`, `/Volumes/develop/options-helper/apps/streamlit/components/health_page.py`, `/Volumes/develop/options-helper/apps/streamlit/components/gap_planner.py`, `/Volumes/develop/options-helper/tests/portal/test_health_queries.py`, `/Volumes/develop/options-helper/tests/portal/test_gap_planner.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T11: Implement Portfolio + Symbol Explorer Pages
- **depends_on**: [T9]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/pages/02_Portfolio.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/03_Symbol_Explorer.py`
- **description**: Portfolio page loads portfolio JSON and best-effort risk summary (briefing artifact fallback + live computed fallback). Symbol Explorer page provides symbol selector/query params, candles chart, latest snapshot summary (OI/IV views), and derived-history snippets.
- **validation**: `./.venv/bin/python -m pytest tests/portal/test_portfolio_page_queries.py tests/portal/test_symbol_explorer_queries.py`
- **status**: Completed
- **log**: Implemented full Portfolio and Symbol Explorer pages with reusable helper modules for portfolio loading/risk-summary fallback and symbol-level DuckDB queries. Added deterministic offline tests covering portfolio parsing/risk-source fallback plus symbol query helpers, snapshot summaries, and derived snippets.
- **files edited/created**: `/Volumes/develop/options-helper/apps/streamlit/pages/02_Portfolio.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/03_Symbol_Explorer.py`, `/Volumes/develop/options-helper/apps/streamlit/components/portfolio_page.py`, `/Volumes/develop/options-helper/apps/streamlit/components/symbol_explorer_page.py`, `/Volumes/develop/options-helper/tests/portal/test_portfolio_page_queries.py`, `/Volumes/develop/options-helper/tests/portal/test_symbol_explorer_queries.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T12: Implement Flow, Derived History, and Data Explorer Pages
- **depends_on**: [T4, T9]
- **location**: `/Volumes/develop/options-helper/apps/streamlit/pages/04_Flow.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/05_Derived_History.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/06_Data_Explorer.py`
- **description**: Add interactive Flow page from `options_flow`; Derived page from `derived_daily`; Data Explorer for schema/table/preview browsing. Keep portal read-only.
- **validation**: `./.venv/bin/python -m pytest tests/portal/test_flow_queries.py tests/portal/test_derived_queries.py tests/portal/test_data_explorer_queries.py`
- **status**: Completed
- **log**: Implemented read-only Flow, Derived History, and Data Explorer pages with symbol/group/date and window filters, chart/table summaries, and graceful missing DB/table messaging. Added reusable deterministic query helpers in Streamlit components for each page and offline portal tests for filtering, aggregation, preview, and error-handling behavior.
- **files edited/created**: `/Volumes/develop/options-helper/apps/streamlit/pages/04_Flow.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/05_Derived_History.py`, `/Volumes/develop/options-helper/apps/streamlit/pages/06_Data_Explorer.py`, `/Volumes/develop/options-helper/apps/streamlit/components/flow_page.py`, `/Volumes/develop/options-helper/apps/streamlit/components/derived_history_page.py`, `/Volumes/develop/options-helper/apps/streamlit/components/data_explorer_page.py`, `/Volumes/develop/options-helper/tests/portal/test_flow_queries.py`, `/Volumes/develop/options-helper/tests/portal/test_derived_queries.py`, `/Volumes/develop/options-helper/tests/portal/test_data_explorer_queries.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T13: Add `options-helper ui` Launcher Command
- **depends_on**: [T1, T9]
- **location**: `/Volumes/develop/options-helper/options_helper/commands/ui.py` (new), `/Volumes/develop/options-helper/options_helper/cli.py`
- **description**: Add thin launcher that shells out to `python -m streamlit run /Volumes/develop/options-helper/apps/streamlit/streamlit_app.py`; include host/port/path flags; provide install guidance when Streamlit extra is missing.
- **validation**: `./.venv/bin/python -m pytest tests/test_ui_cli.py`
- **status**: Completed
- **log**: Finalized the launcher so `options-helper ui` runs Streamlit directly (with `run` alias kept), added `--host`/`--port`/`--path` options, and added explicit install guidance when Streamlit is unavailable before subprocess invocation.
- **files edited/created**: `/Volumes/develop/options-helper/options_helper/commands/ui.py`, `/Volumes/develop/options-helper/tests/test_ui_cli.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T14: Add Dagster Definitions Scaffold
- **depends_on**: [T1]
- **location**: `/Volumes/develop/options-helper/apps/dagster/defs/__init__.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/resources.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/jobs.py` (new), `/Volumes/develop/options-helper/apps/dagster/defs/schedules.py` (new)
- **description**: Create optional Dagster package layout and resource config wiring for paths/provider/portfolio/watchlists without impacting core CLI startup.
- **validation**: `pytest.importorskip("dagster")` defs import smoke test.
- **status**: Completed
- **log**: Added `apps/dagster/defs` scaffold with a minimal `Definitions` assembly, placeholder jobs/schedules registries, path/provider runtime resource placeholders, and a Dagster import smoke test guarded by `pytest.importorskip("dagster")`.
- **files edited/created**: `/Volumes/develop/options-helper/apps/dagster/defs/__init__.py`, `/Volumes/develop/options-helper/apps/dagster/defs/resources.py`, `/Volumes/develop/options-helper/apps/dagster/defs/jobs.py`, `/Volumes/develop/options-helper/apps/dagster/defs/schedules.py`, `/Volumes/develop/options-helper/tests/test_dagster_defs_scaffold.py`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

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
- **status**: Completed
- **log**: Added full daily-partitioned Dagster asset graph (`candles_daily` → `options_bars` → `options_snapshot_file` → `options_flow` → `derived_metrics` → `briefing_markdown`) that reuses `options_helper.pipelines.visibility_jobs` service functions, writes DuckDB run-ledger rows with `triggered_by='dagster'` and Dagster `run_id` as `parent_run_id`, and emits persisted watermarks/events. Added Dagster asset checks that run existing quality-check evaluators and persist equivalent check rows to `meta.asset_checks`. Wired Definitions/jobs/schedules to expose the graph and daily partitioned job schedule, and added Dagster `importorskip` tests for defs load plus one partitioned materialization order/ledger path with mocked service calls.
- **files edited/created**: `/Volumes/develop/options-helper/apps/dagster/defs/assets.py`, `/Volumes/develop/options-helper/apps/dagster/defs/checks.py`, `/Volumes/develop/options-helper/apps/dagster/defs/jobs.py`, `/Volumes/develop/options-helper/apps/dagster/defs/schedules.py`, `/Volumes/develop/options-helper/apps/dagster/defs/__init__.py`, `/Volumes/develop/options-helper/tests/test_dagster_daily_assets.py`, `/Volumes/develop/options-helper/apps/dagster/AGENTS.md`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

### T16: Documentation and MkDocs Navigation Updates
- **depends_on**: [T8, T10, T11, T12, T13, T15]
- **location**: `/Volumes/develop/options-helper/README.md`, `/Volumes/develop/options-helper/mkdocs.yml`, `/Volumes/develop/options-helper/docs/OBSERVABILITY.md`, `/Volumes/develop/options-helper/docs/PORTAL_STREAMLIT.md`, `/Volumes/develop/options-helper/docs/DAGSTER_OPTIONAL.md`, `/Volumes/develop/options-helper/docs/NEXT_STAGE_VISIBILITY_PORTAL.md`, `/Volumes/develop/options-helper/docs/MKDOCS_NAV_PATCH.md`
- **description**: Update docs to match actual behavior/commands; wire new docs into nav; include explicit “not financial advice” disclaimer in portal and health docs.
- **validation**: `mkdocs build` and docs-link sanity checks.
- **status**: Completed
- **log**: Updated README and visibility/portal/orchestration docs to match shipped behavior (`db health`, `ui`, implemented Streamlit pages, DuckDB observability runtime, optional Dagster assets/checks), added explicit not-financial-advice messaging in portal/health docs, and added a coherent `Visibility & Portal` MkDocs nav section.
- **files edited/created**: `/Volumes/develop/options-helper/README.md`, `/Volumes/develop/options-helper/mkdocs.yml`, `/Volumes/develop/options-helper/docs/OBSERVABILITY.md`, `/Volumes/develop/options-helper/docs/PORTAL_STREAMLIT.md`, `/Volumes/develop/options-helper/docs/DAGSTER_OPTIONAL.md`, `/Volumes/develop/options-helper/docs/NEXT_STAGE_VISIBILITY_PORTAL.md`, `/Volumes/develop/options-helper/docs/MKDOCS_NAV_PATCH.md`, `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

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
- **status**: Completed
- **log**:
  - Audited deterministic offline coverage for all T17 goals and confirmed existing tests already cover delivered features without gaps requiring new tests.
  - Confirmed coverage mapping: migration upgrade (`tests/test_duckdb_migrations_v2.py::test_duckdb_migrations_upgrade_v2_to_v3`), command run ledger + watermarks + filesystem noop (`tests/test_command_run_ledger_instrumentation.py`), non-blocking checks (`tests/test_observability_checks_integration.py`, `tests/test_quality_checks.py`), `db health` output (`tests/test_db_health_cli.py`), flow persistence table (`tests/test_duckdb_flow_store.py`), UI launcher (`tests/test_ui_cli.py`, `tests/test_cli_contract.py`), Streamlit helpers (`tests/portal/test_health_queries.py`, `tests/portal/test_gap_planner.py`, `tests/portal/test_portfolio_page_queries.py`, `tests/portal/test_symbol_explorer_queries.py`, `tests/portal/test_flow_queries.py`, `tests/portal/test_derived_queries.py`, `tests/portal/test_data_explorer_queries.py`, `tests/portal/test_streamlit_scaffold.py`), Dagster optional smoke/materialization (`tests/test_dagster_defs_scaffold.py`, `tests/test_dagster_daily_assets.py`).
  - Ran targeted pytest matrix for T6/T7/T8/T10/T11/T12/T13/T15 plus migration/flow persistence confirmations; all selected tests passed (Dagster tests skipped when `dagster` extra was absent).
- **files edited/created**: `/Volumes/develop/options-helper/docs/plan/VISIBILITY-PORTAL-ORCHESTRATION-PLAN.md`

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
