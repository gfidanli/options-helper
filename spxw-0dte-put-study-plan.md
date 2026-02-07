# Plan: SPX/SPY 0DTE Put-Sell Intraday Probability Study (CLI + Streamlit)

**Generated**: 2026-02-07

## Overview
Design and implement a research workflow that estimates:
- How likely SPX/SPY is to finish below candidate end-of-day downside levels given current intraday move/extension.
- Which 0DTE put strikes align with user-selected breach-probability risk tiers (for example `<=1%`, `<=5%`).
- How the approach performs in walk-forward backtests and ongoing forward tests.

Core objective:
- At intraday decision time `t`, estimate `P(close_return <= strike_return | state_at_t)` and map that to strike candidates and expected trade-offs.

Execution principles:
- No lookahead. Intraday features use only data available at/through decision timestamp.
- If decision is computed on bar close, entry is anchored to next tradable bar/quote timestamp.
- Forward-test scoring always uses model artifacts trained only through prior sessions (`trained_through_session`).

This system is informational/educational decision support only and **not financial advice**.

## Prerequisites
- Python env with `.[dev]`; `.[ui]` for dashboard work.
- DuckDB warehouse and/or filesystem intraday partitions with SPY/SPX/SPXW coverage.
- Existing intraday pulls available (`options-helper intraday pull-stocks-bars`, `pull-options-bars`) or equivalent fixtures.
- Context7 references reviewed:
  - Streamlit multipage + caching patterns: `/streamlit/docs`
  - Typer command/validation patterns: `/fastapi/typer`
  - DuckDB time-series SQL patterns: `/websites/duckdb_stable`

## Locked Product Decisions
1. Default proxy underlying is `SPY` because Alpaca does not provide SPX index data in this workflow.
2. Reports/docs must clearly label outputs as **SPY proxy** (not true SPX/SPXW underlying behavior).
3. Default entry fill model is `bid`, with user-adjustable fill/slippage knobs.
4. Default breach-probability tiers are `[0.5%, 1%, 2%, 5%]`, with user-adjustable overrides.
5. Backtest and forward-study outputs must include **both** exit views:
   - `hold_to_close`
   - `adaptive_exit` (rule-based early-exit path)

## Decision Cadence (Option 2) - Tradeoff Note
- `Fixed time` (recommended default for primary benchmark, e.g. `10:30 ET`):
  - Pros: stable sample comparability day-to-day, easier calibration interpretation.
  - Cons: can miss later intraday dislocations.
- `Rolling checks` (e.g. every 15 minutes):
  - Pros: captures changing intraday state and opportunistic entries.
  - Cons: more multiple-testing noise, stricter concurrency control needed.
- Plan stance: implement both modes with one canonical benchmark (`fixed_time=10:30 ET`) and optional rolling mode for secondary analysis.

## Dependency Graph

```text
T0,T2 -> T1 -> T3
T0,T1,T2 -> T4,T4A
T2,T3 -> T4B
T3,T4,T4A,T4B -> T5 -> T6,T8,T9
T0,T3,T6,T4A -> T7
T9 -> T9A
T1,T9 -> T11 -> T10
T6,T7,T8,T9,T11,T4A,T4B -> T10 -> T11A
T11A,T10,T0 -> T12 -> T13
T11A -> T14 -> T15 <- T12
T10,T11A -> T16
T12,T13,T15,T16 -> T17 -> T18
```

## Tasks

### T0: Lock Defaults + Acceptance Criteria
- **depends_on**: []
- **location**: `docs/SPXW_0DTE_PUT_STUDY.md`, `options_helper/schemas/zero_dte_put_study.py`
- **description**: Lock baseline assumptions in one place: SPY proxy default, canonical fixed decision time benchmark (`10:30 ET`) plus optional rolling mode, bid-default fill model, default risk tiers, dual exit modes, and position concurrency caps. Define measurable acceptance criteria for calibration and backtest outputs.
- **validation**: A single config/contract fixture validates required defaults and proxy-disclaimer fields are present and referenced by CLI and dashboard layers.
- **status**: Completed
- **log**: Added `ZeroDteStudyAssumptions` defaults (SPY proxy, fixed `10:30` ET benchmark with rolling support, bid fill default, risk tiers, dual exit modes, and concurrency caps) and documented locked defaults plus measurable acceptance criteria in the new study doc.
- **files edited/created**: `options_helper/schemas/zero_dte_put_study.py`, `docs/SPXW_0DTE_PUT_STUDY.md`, `tests/test_zero_dte_schema.py`, `spxw-0dte-put-study-plan.md`
- **gotchas/errors**: Target files did not previously exist; added new schema/doc/test files and validated contract behavior through focused schema tests.

### T1: Define Research Contract + Assumption Schema
- **depends_on**: [T0]
- **location**: `options_helper/schemas/zero_dte_put_study.py`, `docs/ARTIFACT_SCHEMAS.md`
- **description**: Define typed contracts for decision timestamps, intraday state features, conditional probability outputs, strike ladders, quote quality statuses, skip reasons, and trade simulations. Include explicit assumption fields (`decision_bar`, `entry_anchor`, `fill_model`, `settlement_rule`) and disclaimer metadata.
- **validation**: Schema tests validate required fields, enum constraints (`quote_quality_status`, `skip_reason`, `no_entry_anchor`), and JSON round-trip stability.
- **status**: Completed
- **log**: Implemented typed contract models for anchors/assumptions/disclaimers and probability/strike-ladder/simulation rows, including strict enum constraints for `quote_quality_status` and `skip_reason` (`no_entry_anchor`), then documented the artifact contract in `ARTIFACT_SCHEMAS.md`.
- **files edited/created**: `options_helper/schemas/zero_dte_put_study.py`, `docs/ARTIFACT_SCHEMAS.md`, `tests/test_zero_dte_schema.py`, `spxw-0dte-put-study-plan.md`
- **gotchas/errors**: Enforced anti-lookahead fail-closed behavior at schema level (`entry_anchor_ts` missing requires `skip_reason=no_entry_anchor`) to prevent ambiguous downstream handling.

### T2: Build Intraday State Dataset Loader (Read-only I/O)
- **depends_on**: []
- **location**: `options_helper/data/zero_dte_dataset.py`, `options_helper/data/intraday_store.py`, `options_helper/data/stores_duckdb.py`
- **description**: Add loaders that assemble per-day/per-time snapshots from intraday underlying bars (SPY default proxy; optional SPX/SPXW sources when available) and optional same-day option bars/snapshots. Handle sparse/missing partitions, timezone normalization, holidays, half-days, and DST boundaries.
- **validation**: Deterministic fixture tests for missing files/tables, timezone conversion, half-day sessions, and DST boundary sessions.
- **status**: Completed
- **log**: Added a new read-only `ZeroDTEIntradayDatasetLoader` in `options_helper/data/zero_dte_dataset.py` with typed session/dataset interfaces, deterministic timezone normalization (`America/New_York`), US-equity holiday/half-day session windows, per-time snapshot assembly, sparse partition handling, and optional same-day option snapshot/bars loading (filesystem-first with safe DuckDB missing-table fallback). Added deterministic fixture coverage for missing files/tables, timezone conversion, half-day behavior, and DST-boundary sessions.
- **files edited/created**: `options_helper/data/zero_dte_dataset.py`, `tests/test_zero_dte_dataset_loader.py`, `spxw-0dte-put-study-plan.md`

### T3: Add Strike/Premium Snapshot + Contract Eligibility Loader
- **depends_on**: [T0, T1, T2]
- **location**: `options_helper/data/zero_dte_dataset.py`, `options_helper/analysis/osi.py`
- **description**: Resolve candidate strikes (as `%` of previous close) to tradable contracts, enforce settlement/eligibility filters (for example SPX AM-settled monthly vs SPXW PM-settled), and attach entry premium fields from nearest valid quote/bar snapshot with freshness + microstructure quality tags.
- **validation**: Tests cover no-contract cases, tie-breaking, stale/locked/crossed markets, zero or negative bid/ask, max-spread threshold failures, settlement filtering, and fallback behavior.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4: Build Intraday Extension Feature Engine (Pure Analysis)
- **depends_on**: [T0, T1, T2]
- **location**: `options_helper/analysis/zero_dte_features.py`
- **description**: Compute features available at decision time: intraday return vs prior close, drawdown from open, distance from VWAP, realized intraday volatility, bar-range percentile, and time-of-day bucket; optionally join IV regime context.
- **validation**: Unit tests verify feature formulas, NaN handling, and strict time-cutoff enforcement.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4A: Build Label/Anchor Contract Module (Anti-Lookahead)
- **depends_on**: [T0, T1, T2]
- **location**: `options_helper/analysis/zero_dte_labels.py`
- **description**: Implement deterministic label generation and anchor semantics (`decision_ts`, `entry_anchor_ts`, `close_label_ts`) including fail-closed handling when no next tradable anchor exists (`skip_reason=no_entry_anchor`).
- **validation**: Regression tests for late-session decisions, early-close sessions, and missing final bars prove no lookahead leakage.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4B: Add Data Sufficiency + Session Preflight Gates
- **depends_on**: [T2, T3]
- **location**: `options_helper/analysis/zero_dte_preflight.py`
- **description**: Add preflight checks for minimum sample thresholds per time bucket/regime and minimum quote-quality pass rates before model fitting/backtesting/forward scoring.
- **validation**: Tests verify fail-fast behavior and user-friendly diagnostics when coverage thresholds are not met.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5: Implement Conditional Close-Tail Probability Model
- **depends_on**: [T3, T4, T4A, T4B]
- **location**: `options_helper/analysis/zero_dte_tail_model.py`
- **description**: Implement baseline empirical conditional model estimating end-of-day close-return distribution conditioned on intraday state bucket (time-of-day + extension bucket + volatility regime), with small-sample shrinkage (for example Beta prior smoothing). Output strike-threshold breach probabilities with confidence intervals and sample counts.
- **validation**: Offline tests verify monotonic probability behavior across deeper strikes, stable results on fixed fixtures, and sensible outputs for low-sample bins.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6: Build Strike Recommendation Policy Layer
- **depends_on**: [T3, T5]
- **location**: `options_helper/analysis/zero_dte_policy.py`
- **description**: Map modeled probabilities to ranked strike options by risk tier while incorporating premium availability/quality flags. Produce strike distance, breach probability, premium estimate, and EV proxy bands with explicit skip/fallback reasons when quotes are invalid.
- **validation**: Unit tests confirm farther OTM strikes for tighter risk tiers, deterministic ranking, and deterministic skip behavior for quote-quality failures.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7: Implement Trade Outcome Simulator (0DTE Put Seller)
- **depends_on**: [T0, T3, T4A, T6]
- **location**: `options_helper/backtesting/zero_dte_put.py`
- **description**: Simulate short-put outcomes per decision event using selected strike/premium assumptions and close-settlement intrinsic payoff. Include fees/slippage, position sizing modes, explicit same-day position concurrency caps, and parallel outcome tracks for `hold_to_close` and `adaptive_exit`.
- **validation**: Scenario tests for full-win, partial-win, ITM loss, deep-loss tails, settlement-mode correctness, concurrency-cap enforcement, and parity checks between both exit tracks.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: Build Calibration Metrics Module
- **depends_on**: [T5]
- **location**: `options_helper/analysis/zero_dte_calibration.py`
- **description**: Add forecast-quality metrics for the probability model (Brier score, reliability bins, observed-vs-predicted breach rates, sharpness). Keep outputs independent of PnL simulation.
- **validation**: Unit tests assert metric formulas and reliability-bin aggregation on synthetic known distributions.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9: Persist Frozen Model State for Inference
- **depends_on**: [T5]
- **location**: `options_helper/data/zero_dte_artifacts.py`, `options_helper/schemas/zero_dte_put_study.py`
- **description**: Persist fit-time model state and metadata (`model_version`, `trained_through_session`, `assumptions_hash`) so forward-test inference cannot accidentally refit on future data.
- **validation**: Tests verify deterministic model snapshots, hash stability, and explicit rejection of forward scoring when model state is stale/missing.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9A: Add Active Model Registry + Promotion Rules
- **depends_on**: [T9]
- **location**: `options_helper/data/zero_dte_artifacts.py`, `options_helper/schemas/zero_dte_put_study.py`
- **description**: Implement model registry metadata (`active_model_version`, promoted_at, compatibility metadata, rollback pointer) and deterministic promotion rules so forward scoring always resolves one auditable active model.
- **validation**: Tests verify active-model resolution, compatibility checks, promotion/rollback behavior, and deterministic failure when no valid active model exists.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T10: Implement Walk-Forward Backtest Orchestrator
- **depends_on**: [T6, T7, T8, T9, T11, T4A, T4B]
- **location**: `options_helper/backtesting/zero_dte_walk_forward.py`
- **description**: Run rolling train/test windows where each test day uses only prior history/model snapshots. Produce calibration metrics and trading outcomes by risk tier, decision-time mode (`fixed` vs `rolling`), decision time bucket, regime, and exit mode (`hold_to_close`, `adaptive_exit`).
- **validation**: Deterministic tests verify split boundaries, no future leakage, and reproducible outputs for fixed fixtures.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T11: Define Artifact Contracts + Upsert Keys (Pre-Implementation)
- **depends_on**: [T1, T9]
- **location**: `options_helper/schemas/zero_dte_put_study.py`, `docs/ARTIFACT_SCHEMAS.md`
- **description**: Finalize schema version and storage contracts before writer implementation, including unique upsert key for forward rows (`symbol`, `session_date`, `decision_ts`, `risk_tier`, `model_version`, `assumptions_hash`).
- **validation**: Schema and contract tests confirm required fields/key composition before writer code starts.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T11A: Implement Artifact Writers + Idempotent Upserts
- **depends_on**: [T10, T11]
- **location**: `options_helper/data/zero_dte_artifacts.py`
- **description**: Implement writer/upsert code for probability curves, strike ladders, calibration tables, backtest summaries, forward snapshots, and trade ledgers.
- **validation**: Artifact writer tests verify idempotent upserts, no duplicate rows on rerun, and backward-compatible reads.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T12: Add CLI Command for Study + Backtest
- **depends_on**: [T0, T10, T11A]
- **location**: `options_helper/commands/market_analysis.py`, `options_helper/cli.py`, `options_helper/cli_deps.py`
- **description**: Add command group (for example `market-analysis zero-dte-put-study`) with options for symbol (default SPY proxy), date range, decision mode/time(s), risk tiers, strike grid, fill model/slippage (default bid), and output format/path. Keep command layer thin and delegate to analysis/backtesting modules.
- **validation**: CLI tests validate registration, option parsing, locked default binding, error messages, proxy-disclaimer presence, and deterministic JSON/console outputs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T13: Add Forward-Test Snapshot Job (Paper Tracking)
- **depends_on**: [T0, T3, T4A, T4B, T6, T9, T9A, T11A, T12]
- **location**: `options_helper/commands/market_analysis.py`, `options_helper/data/zero_dte_artifacts.py`
- **description**: Add command to score current/most-recent session intraday state using frozen model state only, emit recommendation snapshots, and later reconcile realized close outcome for ongoing forward-test calibration tracking across both exit modes.
- **validation**: Tests verify as-of cutoff integrity, idempotent writes via unique key, day reconciliation logic, and handling of incomplete sessions.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T14: Build Streamlit Data Component (Read-only)
- **depends_on**: [T11A]
- **location**: `apps/streamlit/components/zero_dte_put_page.py`
- **description**: Add cached query helpers for probability surfaces, strike tables, calibration curves, and backtest summaries from persisted artifacts/DuckDB with clear fallback notes for missing data. Forward-test panels should be optional/empty-state friendly until data exists.
- **validation**: Portal query tests verify shape/contracts and missing-table/file behavior with temporary fixtures.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T15: Add Streamlit Page for 0DTE Put Study
- **depends_on**: [T12, T14]
- **location**: `apps/streamlit/pages/11_0DTE_Put_Study.py`, `apps/streamlit/streamlit_app.py`
- **description**: Implement dashboard with controls for symbol, decision mode/time, risk tier, strike distance, and fill assumptions; display current-state probability table, recommended strike ladder, calibration/reliability charts, and walk-forward/forward-test tabs with side-by-side `hold_to_close` and `adaptive_exit` results plus clear missing-data states.
- **validation**: Streamlit smoke/import tests and page-level tests confirm controls/sections render and remain read-only.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T16: Expand Core Automated Tests (Analysis + Backtest + Artifacts)
- **depends_on**: [T10, T11A]
- **location**: `tests/test_zero_dte_*.py`
- **description**: Add deterministic fixture-based tests for features, label anchors, preflight gates, tail model, policy, simulator, walk-forward no-leakage behavior, model snapshot lifecycle, and artifact upsert contracts.
- **validation**: Targeted core pytest subset passes consistently offline.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T17: Add Integration Tests (CLI + Portal)
- **depends_on**: [T12, T13, T15, T16]
- **location**: `tests/test_zero_dte_cli.py`, `tests/portal/test_zero_dte_put_page.py`
- **description**: Validate end-to-end command wiring, forward-test command behavior, and Streamlit query/page rendering against seeded temporary fixtures.
- **validation**: CLI and portal integration tests pass with deterministic outputs and read-only guarantees.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T18: Documentation + End-to-End Verification
- **depends_on**: [T15, T17]
- **location**: `docs/SPXW_0DTE_PUT_STUDY.md`, `docs/index.md`, `mkdocs.yml`
- **description**: Document methodology, assumptions, caveats, CLI usage, and dashboard interpretation. Run end-to-end checks (`cli --help`, study/backtest/forward-test command smoke, portal page import) and ensure all outputs explicitly state not-financial-advice, anti-lookahead assumptions, and SPY-proxy caveat (not true SPX/SPXW underlying).
- **validation**: Docs build and targeted/full tests pass; manual smoke confirms command/page discoverability.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T0, T2 | Immediately |
| 2 | T1 | T0 complete |
| 3 | T3, T4, T4A | T0 + T1 + T2 complete |
| 4 | T4B | T2 + T3 complete |
| 5 | T5 | T3 + T4 + T4A + T4B complete |
| 6 | T6, T8, T9 | T6 after T3 + T5; T8/T9 after T5 |
| 7 | T7 | T0 + T3 + T4A + T6 complete |
| 8 | T9A, T11 | T9 for T9A; T1 + T9 for T11 |
| 9 | T10 | T6 + T7 + T8 + T9 + T11 + T4A + T4B complete |
| 10 | T11A | T10 + T11 complete |
| 11 | T12, T14, T16 | Task-specific dependencies satisfied |
| 12 | T13 | T0 + T3 + T4A + T4B + T6 + T9 + T9A + T11A + T12 complete |
| 13 | T15 | T12 + T14 complete |
| 14 | T17 | T12 + T13 + T15 + T16 complete |
| 15 | T18 | T15 + T17 complete |

## Testing Strategy
- Unit tests for intraday feature engineering and conditional probability estimation (including sparse-bin behavior and confidence bands) across fixed-time and rolling decision modes.
- Label/anchor regression tests proving no future leakage and explicit `no_entry_anchor` fail-closed behavior.
- Walk-forward regression tests proving stable train/test boundaries with frozen model state usage only.
- PnL scenario tests for settlement correctness, quote quality skips, concurrency caps, and side-by-side `hold_to_close` vs `adaptive_exit` outputs.
- Artifact tests for model snapshot (`trained_through_session`) and idempotent forward-row upsert contracts.
- CLI contract tests for new study/forward-test commands and validation failures.
- Streamlit query/page tests using temporary DuckDB/artifact fixtures and read-only constraints.

## Risks & Mitigations
- **SPY proxy basis risk vs true SPX/SPXW behavior**: label every report/dashboard panel as SPY proxy and include caveat metadata in artifacts.
- **Settlement/contract mismatch risk**: enforce contract eligibility filters and explicit settlement-mode tests.
- **Model instability across volatility regimes**: include regime-aware segmentation and calibration diagnostics by regime.
- **Quote quality / stale premium snapshots**: enforce freshness, spread, and crossed-market checks; propagate `quote_quality_status` + `skip_reason` through policy, simulator, and artifacts.
- **Overfitting with many conditioning buckets**: start with coarse bins, apply shrinkage, and gate low-sample bins from recommendation ranking.
- **Lookahead leakage in intraday labeling/inference**: centralize `decision_ts` vs `entry_anchor_ts`, persist `trained_through_session`, and fail closed when cutoff metadata is missing.
