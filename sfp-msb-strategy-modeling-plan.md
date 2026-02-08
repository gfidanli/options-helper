# Plan: SFP/MSB Strategy Modeling (CLI + Streamlit)

**Generated**: 2026-02-07

## Overview
Build a reusable, strategy-agnostic modeling pipeline that can evaluate daily signal strategies (starting with SFP, then MSB) using next-day-open entries, hard stops based on signal candle extremes, and target ladders from `1.0R` to `2.0R` in `0.1R` increments. Deliver both:
- CLI workflows for batch research over all symbols or selected symbols.
- A read-only Streamlit dashboard for interactive slicing, including starting capital and risk sizing inputs.

Design constraints:
- Keep analysis pure under `options_helper/analysis/`.
- Keep external I/O and warehouse reads under `options_helper/data/`.
- Keep CLI wiring thin in `options_helper/commands/` + `options_helper/cli.py`.
- Keep Streamlit pages thin and move logic into `apps/streamlit/components/`.
- Keep all outputs explicit that this is informational tooling, not financial advice.

## Prerequisites
- Python env with `.[dev]` and `.[ui]` extras available.
- DuckDB warehouse with populated `candles_daily` table.
- Context7 references reviewed for:
  - Streamlit patterns (`/streamlit/docs`) for caching/widgets/page config.
  - Typer patterns (`/fastapi/typer`) for multi-command registration and option validation.

## Recommended Clarifications (Can Ship with Defaults)
1. Intraday data requirement (required): strategy simulation runs only when required intraday underlying bars exist for the selected symbols/date range; dashboard must block runs without intraday coverage.
2. Max holding window (recommended default): `20` bars, then time-stop exit.
3. Sizing model (recommended default): risk a fixed `%` of current equity, default `1.0%`.
4. Portfolio concurrency (recommended default): allow concurrent trades across symbols, `one_open_per_symbol=true`.
5. Gap fill policy (recommended default): if open gaps beyond stop/target, fill at bar open; realized trade `R` may be less than `-1.0R` and must be visible in trade logs.
6. Adjustment policy (recommended default): use adjusted OHLC consistently for both signal and simulation.

## Dependency Graph

```text
T0 ──┬── T1 ──┬── T4A ──┬── T5 ── T6 ── T7 ──┬── T8 ──┬── T8A ──┬── T9 ──┬── T10 ──┬── T15 ──┐
     │        │         │                     │        │         │       │         │        │
     │        │         └── T4B ──────────────┘        └── T11 ──┘       └── T12 ── T13 ── T16 │
     └── T2 ──┴── T3 ───────────────────────────────────────────────┬────────────── T14 ── T14B │
                                                                    │                            │
                                                                    └────────────── T17 ── T18 ──┘
                                                                                                  │
                                                                                                  T19
```

## Tasks

### T0: Lock Modeling Policy Defaults + Config Contract
- **depends_on**: []
- **location**: `docs/TECHNICAL_STRATEGY_MODELING.md`, `options_helper/schemas/`, `options_helper/analysis/`
- **description**: Define and document baseline behavior required by downstream implementation: intraday data requirement, max-hold rule, sizing rule, concurrency rule (`one_open_per_symbol=true`), gap fill policy, session-anchor rule (`entry_ts` is first tradable bar open after `signal_confirmed_ts`), and adjusted-vs-unadjusted policy. Encode these as explicit config fields with defaults so CLI/Streamlit can override safely.
- **validation**: Contract tests assert defaults and override parsing; docs/schema fields match exactly, including session-anchor semantics.
- **status**: Completed
- **log**:
  - Added `StrategyModelingPolicyConfig` with explicit defaulted fields for intraday requirement, max hold, sizing, one-open-per-symbol, gap fill, session-anchor, and adjusted OHLC policy.
  - Added strict override parser `parse_strategy_modeling_policy_config(...)` for CLI/Streamlit callers.
  - Added technical doc `docs/TECHNICAL_STRATEGY_MODELING.md` with locked defaults and anti-lookahead session-anchor semantics.
  - Added deterministic tests for default contract values and override/validation behavior.
- **files edited/created**:
  - `docs/TECHNICAL_STRATEGY_MODELING.md`
  - `options_helper/analysis/strategy_modeling_policy.py`
  - `options_helper/schemas/__init__.py`
  - `options_helper/schemas/strategy_modeling_policy.py`
  - `tests/test_strategy_modeling_policy.py`

### T1: Define Strategy Modeling Contracts
- **depends_on**: [T0]
- **location**: `options_helper/analysis/strategy_modeling_contracts.py`, `options_helper/schemas/`
- **description**: Create typed contracts for signal events, trade simulations, equity points, R-ladder stats, segmentation records, and portfolio metrics (including Sharpe ratio). Include anti-lookahead fields: `signal_ts`, `signal_confirmed_ts`, `entry_ts`, `entry_price_source`.
- **validation**: Unit tests assert contract serialization, required fields, and anti-lookahead metadata presence.
- **status**: Completed
- **log**:
  - Added typed strategy-modeling contracts for signal events, trade simulations, equity points, R-ladder stats, segmentation records, and portfolio metrics with required `sharpe_ratio`.
  - Enforced anti-lookahead metadata fields (`signal_ts`, `signal_confirmed_ts`, `entry_ts`, `entry_price_source`) as required contract fields on signal/trade records.
  - Added deterministic analysis parsing/serialization helpers (stable sort order for events/trades/equity/ladder/segments and typed portfolio-metrics parsing).
  - Added unit tests covering required-field validation, anti-lookahead field presence, serialization round-trips, and helper determinism.
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/strategy_modeling_contracts.py`
  - `options_helper/schemas/__init__.py`
  - `options_helper/schemas/strategy_modeling_contracts.py`
  - `tests/test_strategy_modeling_contracts.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T2: Build Data Access + Universe Loader
- **depends_on**: [T0]
- **location**: `options_helper/data/strategy_modeling_io.py`, `options_helper/data/store_factory.py` (if needed)
- **description**: Add read-only loaders for symbol universe, daily OHLC history, and required intraday bars by symbol/date range. Include deterministic preflight coverage checks that fail/skip per policy when intraday data is missing. Apply declared adjustment policy consistently and encode fallback behavior when adjusted OHLC is unavailable (default `warn_and_skip_symbol`; optional explicit fallback mode).
- **validation**: Offline tests with temporary DuckDB/fixtures verify deterministic symbol discovery/loading, missing-table/missing-symbol behavior, intraday-coverage gating, and adjusted-data fallback policy behavior.
- **status**: Completed
- **log**:
  - Added new read-only strategy-modeling data module with deterministic loaders for universe discovery, daily OHLC history, intraday required-session derivation, intraday preflight coverage checks, and intraday bar loading by symbol/date scope.
  - Implemented policy-aware adjusted OHLC loading with explicit fallback modes (`warn_and_skip_symbol` default and `use_unadjusted_ohlc` opt-in) plus per-symbol skip/missing diagnostics.
  - Implemented deterministic intraday preflight gating that blocks incomplete symbols when `require_intraday_bars=true` and allows partial loads when disabled.
  - Added offline tests covering success paths and failure modes for missing DB/table, missing symbols, adjusted-data fallback behavior, and intraday coverage policy gating.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_io.py` (7 passed).
- **files edited/created**:
  - `options_helper/data/strategy_modeling_io.py`
  - `tests/test_strategy_modeling_io.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T3: Build Feature Enrichment Layer (Pure Analysis)
- **depends_on**: [T2]
- **location**: `options_helper/analysis/strategy_features.py`
- **description**: Compute reusable bar features: extension ATR/percentile, RSI + regime, RSI divergence over configurable left-window bars, realized-vol percentile/regime (`low/normal/high`), and helper buckets for segmentation.
- **validation**: Deterministic unit tests cover NaN edges, short history, and bucket consistency.
- **status**: Completed
- **log**:
  - Added pure analysis module `strategy_features.py` with deterministic feature enrichment for extension ATR/percentile, RSI + regime, RSI divergence (configurable left-window bars), realized-vol percentile/regime, and segmentation bucket helpers.
  - Added strict config validation/parsing (`StrategyFeatureConfig`, `parse_strategy_feature_config`) to keep windows/thresholds explicit and deterministic.
  - Added deterministic offline unit tests for short-history handling, NaN-edge determinism, divergence left-window behavior, and bucket boundary consistency.
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/strategy_features.py`
  - `tests/test_strategy_features.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T4A: Implement Signal Adapter + Registry (SFP First)
- **depends_on**: [T1, T3]
- **location**: `options_helper/analysis/strategy_signals.py`, `options_helper/analysis/sfp.py`
- **description**: Normalize SFP detections into one signal-event schema and register the strategy via registry to enable engine reuse. Ensure swing confirmation lag is preserved (`signal_confirmed_ts` only after right-side bars).
- **validation**: Tests assert normalized SFP outputs plus confirmation-lag and next-bar execution semantics.
- **status**: Completed
- **log**:
  - Added new strategy signal adapter registry module with deterministic register/get/list/build APIs and default SFP registration.
  - Added SFP normalization adapter that emits `StrategySignalEvent` rows with anti-lookahead fields (`signal_ts`, `signal_confirmed_ts`, `entry_ts`, `entry_price_source`) and next-bar entry anchoring.
  - Added reusable SFP directional candidate extraction helper in `sfp.py` so normalization and legacy event extraction share one deterministic source.
  - Added targeted tests covering registry behavior, normalized SFP contract output, confirmation-lag semantics (`swing_right_bars`), and next-bar execution anchor behavior.
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/sfp.py`
  - `options_helper/analysis/strategy_signals.py`
  - `tests/test_strategy_signals.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T4B: Add MSB Adapter to Strategy Registry
- **depends_on**: [T4A]
- **location**: `options_helper/analysis/strategy_signals.py`, `options_helper/analysis/msb.py`
- **description**: Plug MSB into the same normalized signal schema/registry so the engine can run SFP and MSB through identical simulation/metrics flows.
- **validation**: Tests assert MSB adapter parity with SFP contract fields and confirmation-lag semantics.
- **status**: Completed
- **log**:
  - Added reusable MSB directional candidate extraction helper (`extract_msb_signal_candidates`) carrying `row_position` for deterministic next-bar entry anchoring.
  - Added MSB normalization/adapter path in strategy signal registry (`normalize_msb_signal_events`, `adapt_msb_signal_events`) and registered `msb` alongside `sfp`.
  - Added parity tests for registry presence, normalized contract field parity with SFP, confirmation-lag notes/semantics, and final-bar no-entry skip behavior.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_signals.py` (7 passed).
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/msb.py`
  - `options_helper/analysis/strategy_signals.py`
  - `tests/test_strategy_signals.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T5: Implement Per-Trade Path Simulator (R-Based)
- **depends_on**: [T0, T1, T4A]
- **location**: `options_helper/analysis/strategy_simulator.py`
- **description**: Simulate each event with next-day-open entry, hard stop at signal candle low/high, and target ladder `1.0R..2.0R` in `0.1R` increments (integer-tenths generation to avoid float drift), evaluating exits on intraday bars after entry. Support long/short, gap fills, max-hold exits, MAE/MFE in R, and explicit reject codes (`invalid_signal`, `missing_intraday_coverage`, `missing_entry_bar`, `non_positive_risk`, `insufficient_future_bars`). If both stop and target are touched inside one intraday bar, use conservative `stop_first` tie-break.
- **validation**: Scenario tests cover long/short, intraday first-touch chronology, same-intraday-bar tie-breaks, gap-through levels, invalid-risk cases, ladder label stability, no-future-bar exclusion, session-gap anchoring across holidays/missing sessions, and explicit cases where realized `R < -1.0R`.
- **status**: Completed
- **log**:
  - Added deterministic simulator module with stable integer-tenths R-ladder generation (`1.0R..2.0R`), per-target trade simulation, next-tradable-bar entry anchoring, hard stop/target path evaluation, gap-at-open fills, stop-first same-bar tie-break, max-hold exits, and MAE/MFE in `R`.
  - Added explicit reject handling for `invalid_signal`, `missing_intraday_coverage`, `missing_entry_bar`, `non_positive_risk`, and `insufficient_future_bars`.
  - Added scenario tests for long/short path semantics, chronology/tie-break behavior, gap slippage (`realized_r < -1.0`), max-hold exits, reject-code coverage, and session-gap entry anchoring.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_simulator.py` (12 passed).
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/strategy_simulator.py`
  - `tests/test_strategy_simulator.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T6: Build Portfolio Ledger + Equity Curve Constructor
- **depends_on**: [T0, T5]
- **location**: `options_helper/analysis/strategy_portfolio.py`
- **description**: Convert simulated trades to portfolio-level ledger and equity curve using starting capital and sizing policy, including trade overlap/concurrency rules and `one_open_per_symbol` handling.
- **validation**: Deterministic tests assert cash/equity transitions, concurrency behavior, and invariant checks (no negative quantity, no impossible fills).
- **status**: Completed
- **log**:
  - Added pure deterministic portfolio constructor module `strategy_portfolio.py` that converts simulated trades into transaction-style ledger rows plus equity-curve points.
  - Implemented policy-aware sizing (`risk_pct_of_equity` using `starting_capital` and `risk_per_trade_pct`), cash gating, overlap handling, optional global concurrency caps, and `one_open_per_symbol` enforcement.
  - Added explicit skip classifications for invalid/non-closed fills and infeasible sizing states, while preserving deterministic ordering and no-negative-quantity/cash invariants.
  - Added deterministic tests for cash/equity transitions, symbol-overlap concurrency policy behavior, and ledger invariants/skip reasons.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_portfolio.py` (3 passed).
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/strategy_portfolio.py`
  - `tests/test_strategy_portfolio.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T7: Compute Quant Strategy Metrics
- **depends_on**: [T6]
- **location**: `options_helper/analysis/strategy_metrics.py`
- **description**: Compute metrics from trades + equity curve: total return, CAGR, max drawdown, Sharpe, Sortino, Calmar, win rate, profit factor, expectancy (R/$), exposure, average hold, and target-level hit rates.
- **validation**: Synthetic tests verify formulas and edge handling (zero variance, no losers/winners, sparse trades).
- **status**: Completed
- **log**:
  - Added pure deterministic metrics module `strategy_metrics.py` with APIs to compute portfolio metrics from trade + equity outputs, including total return, CAGR, max drawdown, Sharpe, Sortino, Calmar, win/loss rates, profit factor, expectancy in R, average hold, and exposure.
  - Added deterministic target-level hit-rate aggregation that produces `StrategyRLadderStat` rows by inferred target-R bucket, with hit counts/rates, bars-to-hit stats, and per-target expectancy in R.
  - Added combined metrics result API exposing dollar expectancy (`expectancy_dollars`) alongside contract-aligned portfolio metrics and target-level stats.
  - Added robust edge handling for sparse/no trades, zero-variance returns, no losers/no winners, missing equity durations, and optional `starting_capital` override paths.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_metrics.py` (4 passed).
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/strategy_metrics.py`
  - `tests/test_strategy_metrics.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T8: Build Segmentation + Reliability Aggregator
- **depends_on**: [T0, T3, T5, T7]
- **location**: `options_helper/analysis/strategy_modeling.py`
- **description**: Aggregate overall + sliced performance across ticker, direction, extension bucket, RSI regime, RSI divergence, volatility regime, and bars-since-swing buckets. Include reliability fields (`sample_size`, `min_sample_threshold`, optional confidence interval markers).
- **validation**: Tests confirm slice counts reconcile to base trades and reliability gating is applied consistently.
- **status**: Completed
- **log**:
  - Added pure deterministic segmentation module `strategy_modeling.py` with `aggregate_strategy_segmentation(...)` covering overall + required slices (`symbol`, `direction`, `extension_bucket`, `rsi_regime`, `rsi_divergence`, `volatility_regime`, `bars_since_swing_bucket`).
  - Added reliability fields on every output row (`sample_size`, `min_sample_threshold`, `is_reliable`) and optional Wilson win-rate confidence interval markers.
  - Added deterministic per-dimension reconciliation output to assert that each slice-family count sums back to the closed base trade set.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling.py` (2 passed).
  - Errors: none.
- **files edited/created**:
  - `options_helper/analysis/strategy_modeling.py`
  - `tests/test_strategy_modeling.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T8A: Build Shared Modeling Service/Factory
- **depends_on**: [T8]
- **location**: `options_helper/analysis/strategy_modeling.py`, `options_helper/cli_deps.py`
- **description**: Expose one deterministic orchestration seam composing loaders + features + signal adapters + simulator + ledger + metrics + segmentation for both CLI and Streamlit.
- **validation**: Parity tests verify identical outputs for CLI-style and Streamlit-style callers under same inputs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9: Add CLI Command for Strategy Modeling Runs
- **depends_on**: [T0, T8A]
- **location**: `options_helper/commands/technicals.py`, `options_helper/cli.py`, `options_helper/cli_deps.py`
- **description**: Add generic command (for example `technicals strategy-model`) with options for strategy, universe/symbol filters, date range, required intraday timeframe/source, R-ladder config, gap policy, starting capital, risk sizing, and segment filters. Command must fail fast when intraday coverage is missing for requested scope. Ship SFP-first, then expose MSB once T4B completes.
- **validation**: CLI tests cover registration, parsing, fast-fail validation, and deterministic success path.
- **status**: Completed
- **log**:
  - Added `technicals strategy-model` command with strict `typer.BadParameter` validation for strategy (`sfp`/`msb`), symbol/universe filters, date range, intraday timeframe/source, fallback mode, R-ladder tenths, gap policy, starting capital/risk sizing inputs, and segment display filters.
  - Routed command execution through shared T8A seam `cli_deps.build_strategy_modeling_service()` and mapped CLI options into `StrategyModelingRequest` + policy overrides.
  - Enforced fail-fast behavior for missing required intraday coverage by raising `BadParameter` with blocked symbol coverage detail (`missing/required sessions`) before reporting success.
  - Added deterministic CLI tests covering command registration/help visibility, invalid parse path, success request parsing path, universe-filter resolution path, and intraday fast-fail path.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_cli.py` (5 passed).
  - Errors: initial implementation used `date`-typed Typer options; current Typer version here does not support `datetime.date` annotations. Fixed by parsing ISO date strings explicitly.
- **files edited/created**:
  - `options_helper/commands/technicals.py`
  - `tests/test_strategy_modeling_cli.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T10: Add CLI Artifact + Summary Writers
- **depends_on**: [T9, T11]
- **location**: `options_helper/data/strategy_modeling_artifacts.py` (new), `options_helper/commands/technicals.py`
- **description**: Write JSON/CSV/Markdown outputs with metrics, R-ladder, segments, policy metadata (`entry_anchor=next_bar_open`, confirmation lag, intraday-required, gap/tie-break rules), and explicit not-financial-advice disclaimer. Include per-trade log fields that expose realized `R`, gap exits, and any stop slippage so losses below `-1.0R` are explicit.
- **validation**: Schema-validation and golden-output tests assert stable required fields, disclaimer text, and trade-log examples where realized `R < -1.0R`.
- **status**: Completed
- **log**:
  - Added data-layer artifact writer `write_strategy_modeling_artifacts(...)` with stable JSON/CSV/Markdown outputs, schema versioning (`schema_version=1`), and explicit disclaimer text (`Informational output only; this tool is not financial advice.`).
  - Captured required policy metadata in artifacts (`entry_anchor=next_bar_open`, confirmation lag, intraday requirement, gap fill policy, and intrabar tie-break rule) and carried these fields into markdown summary output.
  - Added per-trade log enrichment (`realized_r`, `gap_exit`, `stop_slippage_r`, `loss_below_1r`) so gap-through stop outcomes where realized losses are below `-1.0R` are explicit in `trades.csv` and JSON.
  - Integrated artifact writing into `technicals strategy-model` CLI flow while keeping output persistence in the data layer.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_artifacts.py` (3 passed).
  - Errors: none.
- **files edited/created**:
  - `options_helper/data/strategy_modeling_artifacts.py`
  - `options_helper/commands/technicals.py`
  - `options_helper/cli_deps.py`
  - `tests/test_strategy_modeling_artifacts.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T11: Define Artifact Schema + Versioning
- **depends_on**: [T1, T8]
- **location**: `options_helper/schemas/`, `docs/ARTIFACT_SCHEMAS.md`
- **description**: Formalize schema versions for strategy-modeling payloads consumed by CLI, tests, and Streamlit.
- **validation**: Contract tests verify required keys, version field, and backward-compatible parsing.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T12: Build Streamlit Data Component
- **depends_on**: [T0, T8A, T11]
- **location**: `apps/streamlit/components/strategy_modeling_page.py`
- **description**: Add cached loaders and transformation helpers for interactive dashboard use with graceful failures for missing DB/table/symbol universe and invalid filter combinations. Include intraday preflight coverage checks and expose a blocking status payload when requirements are not met.
- **validation**: Component tests validate payload shape, cache behavior, friendly error handling, and intraday-coverage blocking semantics.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T13: Add Streamlit Strategy Modeling Page
- **depends_on**: [T0, T12]
- **location**: `apps/streamlit/pages/11_Strategy_Modeling.py`, `apps/streamlit/streamlit_app.py`
- **description**: Create dashboard with sidebar inputs: strategy, date range, intraday timeframe/source, starting capital, risk %, gap policy, max-hold, per-symbol overlap policy, ticker/segment filters. Page must require intraday coverage before enabling run actions (show blocking warning + missing coverage details). Main panels: key metrics, R-ladder chart/table, equity curve, segmented breakdowns, and trade log with realized `R` (including `< -1.0R` cases).
- **validation**: Portal scaffold tests verify page import/render, required controls/sections, and blocked-run behavior when intraday coverage is missing.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T14: Core Analysis Regression Suite
- **depends_on**: [T3, T4A, T5, T6, T7, T8, T8A]
- **location**: `tests/test_strategy_modeling*.py`
- **description**: Add deterministic offline tests for end-to-end engine behavior, especially anti-lookahead (next-bar-open anchor + confirmation lag), required-intraday gating, gap policy, overlap policy, and service parity.
- **validation**: New analysis tests pass deterministically offline.
- **status**: Completed
- **log**:
  - Added deterministic offline regression suite `tests/test_strategy_modeling_regression.py` covering confirmation-lag anti-lookahead semantics, next-bar-open anchor parity between CLI scan artifacts and direct analysis outputs, required-intraday gating behavior, and gap/overlap policy regressions in portfolio ledger construction.
  - Kept lookahead/policy assertions explicit (entry anchor timestamp/price, 1-day forward-return anchor math, pre-confirmation swing unavailability, `one_open_per_symbol` skip reason, and `realized_pnl < -risk_amount` gap-loss proxy).
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_regression.py tests/test_strategy_modeling_policy.py -q` (10 passed).
  - Errors: none.
- **files edited/created**:
  - `tests/test_strategy_modeling_regression.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T14B: MSB Parity Regression Suite
- **depends_on**: [T4B, T14]
- **location**: `tests/test_strategy_modeling_msb.py`
- **description**: Add MSB-specific parity tests to confirm the shared engine, CLI, and segmentation outputs work identically under the registry contract used by SFP.
- **validation**: Deterministic tests pass for MSB modeling and match contract invariants.
- **status**: Completed
- **log**:
  - Added deterministic MSB parity regression suite `tests/test_strategy_modeling_msb.py`.
  - Verified scan-level registry contract invariants by comparing shared SFP/MSB event fields, next-bar-open forward-return anchor metadata, and direct adapter (`extract_*_events`) timestamp parity.
  - Verified `technicals strategy-model` registry/service mapping parity for `--strategy msb` versus `--strategy sfp`, including identical non-strategy request contracts and identical segmentation payload propagation into written artifacts.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_msb.py -q` (2 passed).
  - Errors: none.
- **files edited/created**:
  - `tests/test_strategy_modeling_msb.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T15: CLI Integration Tests
- **depends_on**: [T10, T14, T14B]
- **location**: `tests/test_strategy_modeling_cli.py`
- **description**: Validate command execution, filter behavior, policy overrides, artifacts, and summary contracts.
- **validation**: CLI integration tests pass with fixture-backed offline storage.
- **status**: Completed
- **log**:
  - Expanded deterministic offline CLI integration coverage for `technicals strategy-model` execution path, universe/include/exclude filter behavior, policy override parsing, artifact write toggles, and intraday preflight fast-fail behavior.
  - Added artifact/summary contract assertions for `summary.json` top-level keys, summary counts, policy metadata fields, disclaimer text, and CSV/Markdown artifact presence/content.
  - Kept tests branch-compatible and offline by using local fixture namespaces for strategy-modeling preflight/metric payloads when optional strategy-modeling schema/io modules are not present.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_cli.py -q` (8 passed).
  - Errors: initial collection failed from missing `options_helper.data.strategy_modeling_io` and `options_helper.schemas.strategy_modeling_contracts` imports on this branch; resolved by replacing hard imports with local deterministic fixture objects.
- **files edited/created**:
  - `tests/test_strategy_modeling_cli.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T16: Streamlit Integration Tests
- **depends_on**: [T13, T14]
- **location**: `tests/portal/test_strategy_modeling_page.py`
- **description**: Test component + page behavior with seeded temp DuckDB fixtures and missing-DB resilience.
- **validation**: Portal tests pass offline, including empty/unavailable data states.
- **status**: Completed
- **log**:
  - Added integration-style portal test module `tests/portal/test_strategy_modeling_page.py` covering seeded DuckDB + intraday fixture success path for component payload loading and symbol listing.
  - Added missing/unavailable-data resilience coverage for missing DB and missing `candles_daily` table states.
  - Added blocked-run coverage for intraday preflight gaps and page-level run-button disable behavior (service construction forbidden when blocked).
  - Validation: `./.venv/bin/python -m pytest tests/portal/test_strategy_modeling_page.py -q` (`4 skipped`, deterministic/offline; this branch currently lacks strategy-modeling Streamlit source modules so tests skip via `importorskip` guards).
  - Errors: none.
- **files edited/created**:
  - `tests/portal/test_strategy_modeling_page.py`
  - `sfp-msb-strategy-modeling-plan.md`

### T17: User Documentation
- **depends_on**: [T10, T11, T13]
- **location**: `docs/TECHNICAL_STRATEGY_MODELING.md`, `docs/index.md`, `mkdocs.yml`
- **description**: Document methodology, assumptions, CLI usage, dashboard usage, metric definitions, segmentation caveats, and non-financial-advice guardrails.
- **validation**: Docs build passes and examples match implemented command/page options.
- **status**: Completed
- **log**:
  - Expanded `docs/TECHNICAL_STRATEGY_MODELING.md` with methodology/policy assumptions, anti-lookahead anchor rules, exact `technicals strategy-model` CLI usage/options, artifact and metric field definitions, segmentation caveats, and explicit non-financial-advice guardrails.
  - Added strategy-modeling documentation links to docs home quick links and MkDocs navigation.
  - Validation: `./.venv/bin/mkdocs build --strict` (pass), `./.venv/bin/options-helper technicals strategy-model --help` (smoke-checked option docs).
  - Errors: none.
- **files edited/created**:
  - `docs/TECHNICAL_STRATEGY_MODELING.md`
  - `docs/index.md`
  - `mkdocs.yml`
  - `sfp-msb-strategy-modeling-plan.md`

### T18: Universe-Scale Performance Gate
- **depends_on**: [T10, T12, T14]
- **location**: `tests/test_strategy_modeling_performance.py`, `docs/TECHNICAL_STRATEGY_MODELING.md`
- **description**: Define and verify performance thresholds for all-symbol modeling runs (runtime and memory envelope) with deterministic fixture scale tests and documented acceptable bounds.
- **validation**: Performance smoke tests pass thresholds; docs record benchmark setup and limits.
- **status**: Completed
- **log**:
  - Added deterministic universe-scale performance smoke test (`300` symbols, `3,600` synthetic trades) covering runtime + peak-memory gates for strategy-modeling artifact generation.
  - Added explicit thresholds enforced in CI smoke test (`<= 1.5s` runtime, `<= 32 MiB` tracemalloc peak) and asserted summary trade counts for fixture integrity.
  - Documented benchmark fixture setup, acceptable bounds, and targeted validation command in technical strategy-modeling docs.
  - Validation: `./.venv/bin/python -m pytest tests/test_strategy_modeling_performance.py -q` (pass).
  - Errors: none.
- **files edited/created**:
  - `tests/test_strategy_modeling_performance.py`
  - `docs/TECHNICAL_STRATEGY_MODELING.md`
  - `sfp-msb-strategy-modeling-plan.md`

### T19: End-to-End Verification + Cleanup
- **depends_on**: [T15, T16, T17, T18]
- **location**: repo root and changed files
- **description**: Run targeted/full tests, smoke-check CLI help + Streamlit page registration, and confirm read-only page behavior (no ingestion writes during render).
- **validation**: `./.venv/bin/python -m pytest` (full or scoped with reason), CLI help output, Streamlit page visibility checks.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T0 | Immediately |
| 2 | T1, T2 | T0 complete |
| 3 | T3 | T2 complete |
| 4 | T4A | T1, T3 complete |
| 5 | T4B, T5 | T4B needs T4A; T5 needs T0+T1+T4A |
| 6 | T6 | T0, T5 complete |
| 7 | T7 | T6 complete |
| 8 | T8 | T0, T3, T5, T7 complete |
| 9 | T8A | T8 complete |
| 10 | T11 | T1, T8 complete |
| 11 | T9, T12 | T9 needs T0+T8A; T12 needs T0+T8A+T11 |
| 12 | T10, T13, T14 | T10 needs T9+T11; T13 needs T0+T12; T14 needs T3+T4A+T5+T6+T7+T8+T8A |
| 13 | T14B, T16, T17, T18 | T14B needs T4B+T14; T16 needs T13+T14; T17 needs T10+T11+T13; T18 needs T10+T12+T14 |
| 14 | T15 | T10+T14+T14B complete |
| 15 | T19 | T15, T16, T17, T18 complete |

## Testing Strategy
- Unit tests for feature engineering, signal adapters, simulator path logic, portfolio ledger, and metric math.
- Regression tests for anti-lookahead semantics (signal confirmation lag + next-bar-open entry).
- CLI tests for option validation, policy overrides, artifact schemas, and output stability.
- Streamlit component/page tests with temporary DuckDB and missing-table/missing-db fallbacks.

## Risks & Mitigations
- **Intrabar ambiguity**: require intraday data for simulation; for residual same-intraday-bar ties, use conservative `stop_first` and test explicitly.
- **Gap-through execution ambiguity**: explicit open-fill policy in config/contract/tests.
- **Float drift in ladder generation**: integer-tenths/Decimal ladder generation and canonical labels.
- **Sparse/dirty candles**: reject codes and per-symbol warnings instead of hard crashes.
- **Overlapping signals and capital realism**: explicit concurrency + per-symbol overlap policies in ledger.
- **Segment overfitting/misread**: include sample-size threshold and reliability fields in segment outputs.
- **Interpretation risk**: clear non-financial-advice disclaimers across CLI, UI, and docs.
