# Plan: Alpaca Research Metrics Expansion (IV Surface, Exposure, Intraday Flow, Levels, Scenarios)

**Generated**: 2026-02-07

## Overview
Expand the repo’s post-ingestion analytics to better support *options position management* (entries/adds/rolls/exits) by adding:

1) **IV surface + regime context** (tenor + delta-bucket structure with history/percentiles),
2) **Dealer/exposure-style “levels that matter”** (gamma exposure by strike/expiry + flip heuristics),
3) **Intraday options flow** from **trades + quotes** (directional classification + netting),
4) **Underlying actionable levels** (anchored VWAPs, volume profile, gaps, RS/beta),
5) **Position-centric scenario tables** (spot/IV/time grids + extrinsic/theta burn).

All outputs remain **informational only / not financial advice**, offline-first where feasible, deterministic, and testable.

## Decisions Locked in T1 (minimize rework)
These are locked to minimize schema churn in downstream tasks.

1) **Signed exposure conventions**:
   - A) `calls_positive_puts_negative` (heuristic retail convention)
2) **Persistence scope**:
   - B) artifacts + **DuckDB tables** for portal + percentiles (recommended for surface/exposure).
3) **Intraday source of truth**:
   - C) Hybrid approach: **stream capture** (`options-helper stream capture --quotes/--trades`) + offline summarizers and REST backfill via Alpaca historical option quotes/trades
   - The intraday artifact can carry fields like source=stream|rest and quote_coverage_pct, so reports stay honest.
4) **Tenor set for IV surface MVP**: **7/14/30/60/90/180 DTE** (stable order, configurable later).
5) **Delta buckets for IV surface + intraday grouping**:
   - `d00_20`, `d20_40`, `d40_60`, `d60_80`, `d80_100` based on `abs(bs_delta)`.

## Prerequisites
- Python env: `pip install -e ".[dev]"`.
- For Alpaca data capture/backfill: `pip install -e ".[alpaca]"` + `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`.
- Storage: DuckDB default (`data/warehouse/options.duckdb`) for persistence tasks.

## Dependency Graph

```
T1 ──┬─> T2 ──> T7.1 ──┬─> T9 ──┐
     ├─> T3 ──> T7.2 ──┘        │
     ├─> T4 ──> T7.3 ──> T10 ───┼─> T12 ──> T13
     ├─> T5 ──> T7.4 ──> T10 ───┤
     └─> T6 ──> T7.5 ──> T11 ───┘

T1,T2,T3 ───────────────> T8 ───┘
```

Legend:
- T1 = metric definitions + schemas (unblocks DB schema + CLI artifact contracts)
- T2–T6 = pure analysis modules (depend on T1 conventions)
- T7.* = artifact schemas + fixtures (one per feature area)
- T8 = persistence layer (DuckDB + stores)
- T9–T11 = CLI + report-pack + portal integration
- T12–T13 = checks/automation hardening

## Tasks

### T1: Metric Specs + Naming + MVP Contracts
- **depends_on**: []
- **location**:
  - `docs/` (new docs below)
  - `options_helper/schemas/` (new artifact schemas)
- **description**:
  - Write a short spec for each feature: inputs, outputs, stable field names, edge cases.
  - Decide the **signed exposure convention** and the **IV surface tenor + delta-bucket definitions**.
  - Define artifact boundaries (what is computed from snapshots vs intraday partitions vs candles).
- **validation**:
  - Specs reviewed for consistency with repo architecture (analysis pure, data I/O isolated, CLI thin).
- **status**: Completed
- **log**: Added concrete MVP contracts for IV surface/exposure/intraday flow/levels/scenarios (inputs, outputs, edge cases), locked signed exposure + tenor/delta-bucket conventions, and documented explicit artifact boundaries with informational-use language.
- **files edited/created**:
  - `docs/RESEARCH_METRICS_CONTRACTS.md`
  - `options_helper/schemas/research_metrics_contracts.py`
  - `options_helper/schemas/__init__.py`
  - `docs/plan/alpaca-research-metrics-plan.md`
- **errors/gotchas**: None.

### T2: IV Surface Extraction (Pure Functions)
- **depends_on**: [T1]
- **location**:
  - `options_helper/analysis/iv_surface.py` (new)
  - `tests/` (new unit tests)
- **description**:
  - Build pure functions that take a chain snapshot DataFrame + spot + as-of date and compute:
    - selected expiry for each target DTE (closest within bounds),
    - per-tenor ATM IV, straddle mark, expected move %, skew (25Δ, optional 10Δ),
    - delta-bucket IV summaries (calls/puts separately),
    - day-over-day changes when provided previous-day surface.
  - Provide deterministic handling for missing columns (`impliedVolatility`, `bs_delta`, quotes).
- **validation**:
  - `pytest -k iv_surface` on synthetic chain fixtures.
- **status**: Completed
- **log**: Added deterministic pure IV surface extraction with tenor selection, ATM/straddle/expected-move/skew metrics, delta-bucket summaries, and day-over-day change tables with robust warnings for sparse/missing fields.
- **files edited/created**:
  - `options_helper/analysis/iv_surface.py`
  - `tests/test_iv_surface.py`

### T3: Dealer/Exposure Metrics (Pure Functions)
- **depends_on**: [T1]
- **location**:
  - `options_helper/analysis/exposure.py` (new)
  - `tests/` (new unit tests)
- **description**:
  - Compute strike-level exposure tables from snapshot chains:
    - call/put OI by strike,
    - gamma exposure by strike using `bs_gamma * OI * spot^2 * 0.01 * 100` (align with existing `gamma_1pct`),
    - signed net exposure per the chosen convention (T1).
  - Compute summary levels:
    - top strikes by abs(net exposure),
    - “flip” heuristic (e.g., cumulative-net crossing),
    - per-expiry exposure slices (near/monthly).
  - Keep output schema versioned and deterministic.
- **validation**:
  - `pytest -k exposure` on synthetic strike ladders.
- **status**: Completed
- **log**: Implemented deterministic exposure slice computations (near/monthly/all), strike-level OI/GEX/net tables, top abs-net levels, and cumulative flip heuristic with explicit signed convention and missing-value guardrails.
- **files edited/created**:
  - `options_helper/analysis/exposure.py`
  - `tests/test_exposure.py`

### T4: Intraday Options Flow from Trades + Quotes (Pure Functions)
- **depends_on**: [T1]
- **location**:
  - `options_helper/analysis/intraday_flow.py` (new)
  - `tests/` (new unit tests)
- **description**:
  - Implement best-effort trade direction classification:
    - align trades to latest quote via `merge_asof`,
    - classify buy/sell/unknown using bid/ask bands (e.g., >= ask => buy; <= bid => sell).
  - Explicitly harden edge cases for determinism:
    - sort + de-dup timestamps before `merge_asof` (handle out-of-order events),
    - handle missing/zero bid/ask (classify as unknown + track unknown rates),
    - handle missing sizes/prices and zero-volume trades (drop or coerce with warnings).
  - Summarize per contract/day:
    - buy/sell volume, buy/sell notional, net notional, trade counts, unknown share.
  - Add aggregation helpers:
    - group by expiry/strike/optionType/delta-bucket and by time buckets (5m/15m).
  - No network calls; inputs are DataFrames loaded by callers.
- **validation**:
  - `pytest -k intraday_flow` with synthetic quote/trade sequences (including missing/late quotes).
- **status**: Completed
- **log**: Added deterministic trades+quotes alignment/classification pipeline with merge_asof hardening, buy/sell/unknown tagging, per-contract/day summaries, and 5m/15m time-bucket aggregations.
- **files edited/created**:
  - `options_helper/analysis/intraday_flow.py`
  - `tests/test_intraday_flow.py`

### T5: Underlying Actionable Levels (Anchored VWAP, Volume Profile, RS/Beta)
- **depends_on**: [T1]
- **location**:
  - `options_helper/analysis/levels.py` (new)
  - `tests/` (new unit tests)
- **description**:
  - Anchored VWAP from intraday bars:
    - compute VWAP from `vwap` when present, else typical price proxy * volume,
    - support anchors: session open, a specified timestamp, a specified date, and “breakout day” anchor.
  - Volume profile:
    - bin price into N buckets; compute volume per bin; return POC + HVN/LVN candidates.
  - Gaps + key daily levels:
    - previous close vs open, prior day high/low, rolling highs/lows.
  - Relative strength + beta/correlation:
    - vs SPY (MVP): ratio series, rolling correlation and beta on returns.
  - Add guardrails for empty/zero-volume inputs:
    - avoid divide-by-zero (sum(volume)=0) by returning `None` + warnings instead of raising,
    - treat missing `volume` as 0 and missing prices as NaN (skip bins).
- **validation**:
  - `pytest -k levels` on synthetic minute and daily candle inputs.
- **status**: Completed
- **log**: Implemented anchored VWAP (session/timestamp/date/breakout anchors), volume profile (POC/HVN/LVN), gap/key level summaries, and RS/beta/correlation metrics with deterministic empty/zero-volume handling.
- **files edited/created**:
  - `options_helper/analysis/levels.py`
  - `tests/test_levels.py`

### T6: Position Scenario Grid + Decision Table (Pure Functions)
- **depends_on**: [T1]
- **location**:
  - `options_helper/analysis/scenarios.py` (new)
  - `tests/` (new unit tests)
- **description**:
  - For a position (spot/strike/expiry/option_type/iv/mark/basis), compute:
    - intrinsic/extrinsic,
    - theta burn ($/day and % of premium),
    - scenario grids using `black_scholes_price`:
      - spot moves: ±5/10/20% (configurable),
      - IV moves: ±5/10 vol points (pp),
      - time forward: +7/+14/+30 days.
  - Output a compact, stable schema suitable for console tables + JSON artifacts.
  - Add explicit validation/guards so callers don’t crash:
    - past expiry => scenarios empty + warning,
    - missing spot/iv/mark => partial outputs + warnings,
    - IV <= 0 => treat as missing; never pass invalid params into BS helpers.
- **validation**:
  - `pytest -k scenarios` with known BS pricing invariants (monotonicity checks).
- **status**: Completed
- **log**: Added deterministic per-position scenario engine (intrinsic/extrinsic, theta burn, spot/IV/time grids) with robust guards for past expiry and missing spot/mark/iv inputs.
- **files edited/created**:
  - `options_helper/analysis/scenarios.py`
  - `tests/test_scenarios.py`

### T7.1: Artifact Schema + Fixture — IV Surface
- **depends_on**: [T2]
- **location**:
  - `options_helper/schemas/iv_surface.py` (new)
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/iv_surface_*.json` (new)
- **description**:
  - Add a versioned `IvSurfaceArtifact` schema + minimal fixture validation.
- **validation**:
  - `pytest -k artifact_fixtures`.
- **status**: Completed
- **log**: Added versioned IV surface artifact schema and minimal fixture validation as part of grouped T7 schema/fixture rollout.
- **files edited/created**:
  - `options_helper/schemas/iv_surface.py`
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/iv_surface_minimal.json`

### T7.2: Artifact Schema + Fixture — Exposure
- **depends_on**: [T3]
- **location**:
  - `options_helper/schemas/exposure.py` (new)
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/exposure_*.json` (new)
- **description**:
  - Add a versioned `ExposureArtifact` schema + minimal fixture validation.
- **validation**:
  - `pytest -k artifact_fixtures`.
- **status**: Completed
- **log**: Added versioned exposure artifact schema and minimal fixture validation as part of grouped T7 schema/fixture rollout.
- **files edited/created**:
  - `options_helper/schemas/exposure.py`
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/exposure_minimal.json`

### T7.3: Artifact Schema + Fixture — Intraday Flow
- **depends_on**: [T4]
- **location**:
  - `options_helper/schemas/intraday_flow.py` (new)
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/intraday_flow_*.json` (new)
- **description**:
  - Add a versioned `IntradayFlowArtifact` schema + minimal fixture validation.
- **validation**:
  - `pytest -k artifact_fixtures`.
- **status**: Completed
- **log**: Added versioned intraday-flow artifact schema and minimal fixture validation as part of grouped T7 schema/fixture rollout.
- **files edited/created**:
  - `options_helper/schemas/intraday_flow.py`
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/intraday_flow_minimal.json`

### T7.4: Artifact Schema + Fixture — Levels
- **depends_on**: [T5]
- **location**:
  - `options_helper/schemas/levels.py` (new)
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/levels_*.json` (new)
- **description**:
  - Add a versioned `LevelsArtifact` schema + minimal fixture validation.
- **validation**:
  - `pytest -k artifact_fixtures`.
- **status**: Completed
- **log**: Added versioned levels artifact schema and minimal fixture validation as part of grouped T7 schema/fixture rollout.
- **files edited/created**:
  - `options_helper/schemas/levels.py`
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/levels_minimal.json`

### T7.5: Artifact Schema + Fixture — Scenarios
- **depends_on**: [T6]
- **location**:
  - `options_helper/schemas/scenarios.py` (new)
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/scenarios_*.json` (new)
- **description**:
  - Add a versioned `ScenariosArtifact` schema + minimal fixture validation.
- **validation**:
  - `pytest -k artifact_fixtures`.
- **status**: Completed
- **log**: Added versioned scenarios artifact schema and minimal fixture validation as part of grouped T7 schema/fixture rollout.
- **files edited/created**:
  - `options_helper/schemas/scenarios.py`
  - `docs/ARTIFACT_SCHEMAS.md`
  - `tests/test_artifact_fixtures.py`
  - `tests/fixtures/artifacts/scenarios_minimal.json`

### T8: DuckDB Schema Migration v5 + Stores for New Tables
- **depends_on**: [T1, T2, T3]
- **location**:
  - `options_helper/db/migrations.py`
  - `options_helper/db/schema_v5.sql` (new)
  - `options_helper/data/stores_duckdb.py`
  - `options_helper/data/store_factory.py`
  - `docs/DUCKDB.md` (update “what gets stored where”)
- **description**:
  - Add new warehouse tables (recommended set):
    - `iv_surface_tenor` (symbol/date/tenor_target/expiry/dte/atm_iv/straddle/em/skew/updated_at/provider)
    - `iv_surface_delta_buckets` (symbol/date/tenor_target/option_type/delta_bucket/avg_iv/n/updated_at/provider)
    - `dealer_exposure_strikes` (symbol/date/strike/call_oi/put_oi/call_gex/put_gex/net_gex/updated_at/provider)
    - `intraday_option_flow` (optional MVP persistence) (contract_symbol/day/net_notional/buy/sell/unknown stats)
  - Implement DuckDB store APIs to upsert/query these tables (thin, deterministic).
- **validation**:
  - `options-helper db init` (migration applies) + unit tests for store read/write with temp duckdb path.
- **status**: Completed
- **log**: Added additive/idempotent schema v5 migration for research metrics tables and implemented DuckDB upsert/query APIs (`DuckDBResearchMetricsStore`) wired through store factory.
- **files edited/created**:
  - `options_helper/db/migrations.py`
  - `options_helper/db/schema_v5.sql`
  - `options_helper/data/stores_duckdb.py`
  - `options_helper/data/store_factory.py`
  - `docs/DUCKDB.md`
  - `tests/test_duckdb_migrations.py`
  - `tests/test_duckdb_migrations_v2.py`
  - `tests/test_duckdb_research_metrics_store.py`

### T9: CLI Commands — IV Surface + Exposure
- **depends_on**: [T2, T3, T7.1, T7.2] (and [T8] if persistence enabled)
- **location**:
  - `options_helper/commands/market_analysis.py`
  - `docs/IV_SURFACE.md` (new)
  - `docs/DEALER_EXPOSURE.md` (new)
- **description**:
  - Add:
    - `options-helper market-analysis iv-surface --symbol --as-of latest --format console|json --out ...`
    - `options-helper market-analysis exposure --symbol --as-of latest --format console|json --out ...`
  - Offline-first: read snapshots from `OptionsSnapshotStore` + derived/candles as needed.
  - Optional: upsert to DuckDB tables when backend is duckdb.
- **validation**:
  - CLI smoke tests using fixture snapshots (no network) + `pytest -k market_analysis`.
- **status**: Completed
- **log**: Added `market-analysis iv-surface` and `market-analysis exposure` commands with offline snapshot/candle inputs, console/json outputs, artifact writing, and optional DuckDB persistence hooks.
- **files edited/created**:
  - `options_helper/commands/market_analysis.py`
  - `docs/IV_SURFACE.md`
  - `docs/DEALER_EXPOSURE.md`
  - `tests/test_market_analysis_cli.py`

### T10: CLI Commands — Intraday Flow + Levels
- **depends_on**: [T4, T5, T7.3, T7.4] (and [T8] if persistence enabled)
- **location**:
  - `options_helper/commands/intraday.py` (extend) and/or `options_helper/commands/market_analysis.py` (extend)
  - `options_helper/data/intraday_store.py` (read helpers if needed)
  - `docs/INTRADAY_FLOW.md` (new)
  - `docs/LEVELS.md` (new)
- **description**:
  - Add an offline summarizer for captured tick data:
    - `options-helper intraday flow --underlying/--contract ... --day ... --format ... --out ...`
  - Add a levels report command:
    - `options-helper market-analysis levels --symbol ...` (uses candles + optional intraday bars).
  - Ensure timezones are handled consistently (`OH_ALPACA_MARKET_TZ` for day partitioning; UTC timestamps inside).
  - (Optional v2) REST backfill path: use Alpaca `OptionHistoricalDataClient.get_option_trades` to backfill missing days.
- **validation**:
  - `pytest -k intraday` and a no-network run over synthetic partitions in `tests/fixtures/intraday/`.
- **status**: Completed
- **log**: Added `intraday flow` offline summarizer over local quotes/trades partitions and `market-analysis levels` candle/intraday report command with deterministic JSON/console outputs.
- **files edited/created**:
  - `options_helper/commands/intraday.py`
  - `options_helper/commands/market_analysis.py`
  - `docs/INTRADAY_FLOW.md`
  - `docs/LEVELS.md`
  - `tests/test_intraday_cli.py`
  - `tests/test_market_analysis_cli.py`

### T11: CLI Command — Position Scenarios
- **depends_on**: [T6, T7.5]
- **location**:
  - `options_helper/commands/portfolio.py` (new command) or `options_helper/commands/workflows.py` (register)
  - `docs/POSITION_SCENARIOS.md` (new)
- **description**:
  - Add `options-helper scenarios portfolio.json` (or `position-scenarios`) that:
    - computes scenario tables per position,
    - prints a compact console view,
    - writes JSON artifacts under `{out}/scenarios/{PORTFOLIO_DATE}/...` when `--out` is provided.
  - Keep `analyze` output mostly unchanged; optionally add a `--show-scenarios` flag later.
- **validation**:
  - `pytest -k scenarios_command` + offline run on a tiny portfolio fixture.
- **status**: Completed
- **log**: Added `scenarios` / `position-scenarios` portfolio command with compact console output and per-position JSON artifacts under `{out}/scenarios/{PORTFOLIO_DATE}/...`.
- **files edited/created**:
  - `options_helper/commands/portfolio.py`
  - `docs/POSITION_SCENARIOS.md`
  - `docs/ARTIFACT_SCHEMAS.md`
  - `mkdocs.yml`
  - `tests/test_scenarios_command.py`

### T12: Report Pack + Daily Briefing Integration
- **depends_on**: [T9, T10, T11]
- **location**:
  - `options_helper/commands/reports.py`
  - `options_helper/pipelines/visibility_jobs.py`
  - `docs/REPORT_PACK.md` (update)
  - `docs/BRIEFING.md` (optional update)
- **description**:
  - Extend `report-pack` to generate:
    - IV surface artifact,
    - exposure artifact,
    - levels artifact,
    - (optional) scenarios artifact for portfolio symbols.
  - Treat DuckDB persistence as optional:
    - if `--storage filesystem` (or new tables not populated), still write artifacts from local snapshots/intraday data,
    - emit warnings and skip DB-only panels rather than failing.
  - Optionally add short one-liners in daily briefing sections:
    - IV regime summary (term slope + ATM IV rank by tenor),
    - exposure “top levels” near spot.
- **validation**:
  - Run `options-helper report-pack ...` against fixture stores in tests; ensure deterministic output paths and schemas.
- **status**: Completed
- **log**: Extended `report-pack` to generate iv-surface/exposure/levels artifacts and optional per-position scenarios artifacts with filesystem-first fallback behavior and skip/warn (no hard fail) semantics.
- **files edited/created**:
  - `options_helper/commands/reports.py`
  - `docs/REPORT_PACK.md`
  - `tests/test_report_pack_cli.py`

### T13: Portal (Streamlit) Surfaces
- **depends_on**: [T8, T9, T10, T12]
- **location**:
  - `options_helper/ui/` (existing pages)
  - `docs/PORTAL_STREAMLIT.md` (update)
- **description**:
  - Add portal panels/tabs for:
    - IV surface by tenor + delta buckets (sparklines + latest table),
    - exposure by strike (bar chart + flip marker),
    - intraday flow summary (if persisted) with top strikes/contracts.
- **validation**:
  - Manual: `options-helper ui` renders pages with empty-data fallbacks and no crashes.
- **status**: Completed
- **log**: Added Streamlit research-metrics portal tabs/panels for IV surface, exposure-by-strike (with flip marker), and persisted intraday-flow summaries with robust missing-db/table fallbacks.
- **files edited/created**:
  - `apps/streamlit/components/research_metrics_page.py`
  - `apps/streamlit/pages/07_Market_Analysis.py`
  - `docs/PORTAL_STREAMLIT.md`
  - `tests/portal/test_research_metrics_queries.py`

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1 | Immediately |
| 2 | T2, T3, T4, T5, T6 | Wave 1 complete |
| 3 | T7.1 | T2 complete |
| 3 | T7.2 | T3 complete |
| 3 | T7.3 | T4 complete |
| 3 | T7.4 | T5 complete |
| 3 | T7.5 | T6 complete |
| 3 | T8 | T1, T2, T3 complete |
| 4 | T9 | T2, T3, T7.1, T7.2 (+ T8 if persisting) complete |
| 4 | T10 | T4, T5, T7.3, T7.4 (+ T8 if persisting) complete |
| 4 | T11 | T6, T7.5 complete |
| 5 | T12 | T9, T10, T11 complete |
| 6 | T13 | T8, T9, T10, T12 complete |

## Testing Strategy
- Unit tests for pure computations (surface/exposure/intraday_flow/levels/scenarios) with synthetic inputs.
- Artifact schema validation tests using minimal fixtures.
- CLI-level offline tests using snapshot/candle/intraday fixtures (no network).
- Optional DuckDB store tests using temp duckdb file paths.

## Risks & Mitigations
- **Tick data volume** (quotes/trades): default to opt-in capture; add `--max-contracts` safeguards; summarize to daily aggregates.
- **Noisy/missing greeks/IV**: keep best-effort; add `warnings[]` fields; compute fallbacks (e.g., infer IV from mark where safe).
- **Timezones**: centralize “market day” partition logic (already in streaming runner); ensure commands use the same logic.
- **Overfitting to “dealer metrics” folklore**: keep conventions explicit, configurable, and clearly labeled as heuristic.
- **Schema churn**: version artifacts, keep DuckDB migrations additive/idempotent; document field meanings and units.
