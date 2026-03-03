# Plan: Mean-Reversion IBS Backtest + Reddit Parity Validation

**Generated**: 2026-03-03
**Status**: ready

## Overview
Implement the Reddit strategy in `technicals_backtesting` with:
1. As-is daily backtesting on one or many tickers.
2. Per-ticker + equal-weight aggregate results.
3. SPY buy-and-hold benchmark comparison.
4. CLI HTML report with similar dashboard-style charts.
5. AND-style optional trend overlays.
6. A mandatory validation gate against the reference markdown copy:
   `/Users/sergio/Library/Mobile Documents/iCloud~md~obsidian/Documents/Personal/Clippings/Found a simple mean reversion setup with 70% win rate but only invested 20% of the time.md`

This tool remains informational research only, not financial advice.

## Public API / Interface Changes
1. New strategy ID in technicals engine: `MeanReversionIBS`.
2. New CLI command under `technicals`: `backtest-batch` (exact name finalized during implementation).
3. Extended `technical_backtesting` config/schema for:
   - strategy block `MeanReversionIBS`
   - optional per-strategy cost overrides
   - overlay toggles/params
4. New batch artifact contract:
   - `summary.json` (schema-versioned)
   - per-ticker and aggregate CSVs
   - `report.html`

## Dependency Graph

```text
T0 -> T1 -> T1b
T1 -> T2 -> T3 -> T5 -> T6 -> T7a -> T7 -> T8 -> T9c
T1 -> T4 ------^
T3 -> T9a
T6 -> T9b
T1 + T7a -> T9d
T8 + T7a -> T9e
T8 + T7a + T9e -> T10
T9a + T9b + T9c + T9d + T9e + T10 -> T11
```

## Tasks

### T0: Contract Lock + Reference-File Canon
- **depends_on**: []
- **location**: `docs/technical_backtesting/MEAN_REVERSION_IBS.md` (new)
- **description**:
  - Lock canonical rule semantics from the reference markdown file.
  - Lock parity targets for SPY/QQQ headline metrics from that file.
  - Lock execution anchor (`signal at close`, fill next bar open when `trade_on_close=false`).
  - Lock aggregate rule (equal-weight daily return composite).
  - Lock benchmark rule (SPY buy-and-hold).
- **validation**: Canon spec lists exact formulas, expected reference metrics, and tolerance policy.
- **status**: Completed
- **log**:
  - 2026-03-03: Added canonical contract doc with locked formulas, execution anchor semantics, equal-weight aggregate rule, SPY benchmark rule, and SPY/QQQ reference parity targets with explicit tolerance policy.
- **files edited/created**:
  - `docs/technical_backtesting/MEAN_REVERSION_IBS.md` (new)
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T1: Config + Schema Extension
- **depends_on**: [T0]
- **location**: `config/technical_backtesting.schema.json`, `config/technical_backtesting.yaml`, `options_helper/data/technical_backtesting_config.py`
- **description**:
  - Add `MeanReversionIBS` strategy config block.
  - Add strategy-level cost override support.
  - Keep backward compatibility with existing strategy configs.
- **validation**: Legacy and new configs both load; invalid configs fail cleanly.
- **status**: Completed
- **log**:
  - 2026-03-03: Added `strategies.MeanReversionIBS` defaults/search space/constraints to the default YAML config.
  - 2026-03-03: Extended strategy schema with optional `cost_overrides` (`commission`, `slippage_bps`) and kept it optional for backward compatibility.
  - 2026-03-03: Added loader-side validation for strategy-level cost override type/keys/non-negative numeric values to produce clear `ConfigError` messages.
- **files edited/created**:
  - `config/technical_backtesting.schema.json`
  - `config/technical_backtesting.yaml`
  - `options_helper/data/technical_backtesting_config.py`
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T1b: Config Regression Tests
- **depends_on**: [T1]
- **location**: `tests/test_technical_backtesting_config.py`
- **description**:
  - Add tests for new schema keys and backward compatibility.
- **validation**: Targeted config pytest passes.
- **status**: Completed
- **log**:
  - 2026-03-03: Added regression coverage for `strategies.MeanReversionIBS` defaults, valid optional `cost_overrides`, invalid `cost_overrides` key/value/type handling with `ConfigError`, and legacy config compatibility without new strategy/fields.
- **files edited/created**:
  - `tests/test_technical_backtesting_config.py`
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T2: Strategy Class + Registry Wiring
- **depends_on**: [T1]
- **location**: `options_helper/technicals_backtesting/strategies/mean_reversion_ibs.py` (new), `.../strategies/registry.py`, `.../feature_selection.py`
- **description**:
  - Implement entry:
    `close < (rolling_high_10 - 2.5 * avg(high-low,25))` AND `IBS < 0.3`.
  - Implement exit:
    `close > yesterday_high`.
  - Implement deterministic `IBS` zero-range handling.
  - Register strategy.
- **validation**: Strategy is runnable through existing runner paths.
- **status**: Completed
- **log**:
  - 2026-03-03: Added `MeanReversionIBS` strategy with canonical entry (`close < rolling_high - range_mult * avg_range` and `IBS < threshold`) and exit (`close > high[t-exit_lookback]`) semantics.
  - 2026-03-03: Implemented deterministic IBS zero-range handling with explicit neutral fallback (`ibs_zero_range_value=0.5`).
  - 2026-03-03: Wired registry lookup (`MeanReversionIBS`) and feature selection requirements for strategy frame selection.
  - 2026-03-03: Added focused strategy tests for entry/exit contract, zero-range fallback behavior, and registry/feature-selection wiring.
- **files edited/created**:
  - `options_helper/technicals_backtesting/strategies/mean_reversion_ibs.py` (new)
  - `options_helper/technicals_backtesting/strategies/registry.py`
  - `options_helper/technicals_backtesting/feature_selection.py`
  - `tests/test_mean_reversion_ibs_strategy.py` (new)
  - `tests/test_technical_backtesting_strategy_smoke.py`
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T3: Overlay Gates (AND)
- **depends_on**: [T2]
- **location**: strategy module + helper module(s) in `technicals_backtesting/strategies`
- **description**:
  - Add optional gates:
    - SMA trend gate
    - weekly trend gate
    - MA direction gate
  - Validate overlay params.
- **validation**: Deterministic tests show correct acceptance/rejection behavior.
- **status**: Completed
- **log**:
  - 2026-03-03: Added optional AND entry overlays to `MeanReversionIBS`: SMA trend gate (`close > SMA(window)`), weekly trend gate (`weekly_trend_up` true), and MA direction gate (`SMA(window) > SMA(window-lookback)`).
  - 2026-03-03: Added explicit overlay parameter validation with clear `ValueError` messages for invalid booleans/integers and missing weekly trend column when weekly gate is enabled.
  - 2026-03-03: Added deterministic gate tests covering per-gate accept/reject paths, combined AND behavior, and validation errors; updated IBS feature selection to include `weekly_trend_up` when weekly gate is enabled in strategy config.
- **files edited/created**:
  - `options_helper/technicals_backtesting/strategies/mean_reversion_ibs.py`
  - `options_helper/technicals_backtesting/feature_selection.py`
  - `tests/test_mean_reversion_ibs_strategy.py`
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T4: Cost Override Precedence Resolver
- **depends_on**: [T1]
- **location**: batch runtime module + runner integration seam
- **description**:
  - Resolve costs with precedence:
    CLI > strategy override > global backtest config.
- **validation**: Table-driven precedence tests pass.
- **status**: Completed
- **log**:
  - 2026-03-03: Added `backtest_batch_runtime_costs` helper with deterministic field-level precedence resolution (`commission`, `slippage_bps`) and a merge helper for runtime integration.
  - 2026-03-03: Added table-driven tests for CLI/strategy/global combinations, including partial overrides and missing override fallbacks.
- **files edited/created**:
  - `options_helper/commands/technicals/backtest_batch_runtime_costs.py` (new)
  - `tests/test_technical_backtest_batch_runtime_costs.py` (new)
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T5: Batch Runner Runtime (Thin CLI, Fat Runtime)
- **depends_on**: [T3, T4, T1b]
- **location**: `options_helper/commands/technicals/backtest_batch_runtime.py` (new), `.../backtest/batch_runner.py` (new)
- **description**:
  - Run one or many symbols as-is.
  - Capture per-symbol stats, equity, trades, warnings/errors.
  - Continue on symbol-level failures.
  - Emit stage timings/progress.
- **validation**: Deterministic runtime tests with partial-failure behavior.
- **status**: Completed
- **log**:
  - 2026-03-03: Added a fat runtime batch runner with deterministic symbol normalization, per-symbol outcomes (`stats`, `_equity_curve`, `_trades`, warnings/error), and symbol-level fault isolation so failures do not abort remaining symbols.
  - 2026-03-03: Added stage progress/timing emission (`batch`, `symbol`, `load_ohlc`, `compute_features`, `select_strategy_features`, `run_backtest`) with both per-symbol and aggregate timing metadata.
  - 2026-03-03: Added a thin technicals command runtime wrapper that wires config, strategy defaults, feature selection, and T4 cost precedence (`CLI > strategy override > global`) into the batch runner.
  - 2026-03-03: Added deterministic runtime tests covering normal path, warning capture for missing private stats frames, runtime wiring, and partial-failure continuation behavior.
- **files edited/created**:
  - `options_helper/technicals_backtesting/backtest/batch_runner.py` (new)
  - `options_helper/commands/technicals/backtest_batch_runtime.py` (new)
  - `tests/test_technical_backtest_batch_runner.py` (new)
  - `tests/test_technical_backtest_batch_runtime.py` (new)
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T6: Aggregate + Benchmark Analytics
- **depends_on**: [T5]
- **location**: `options_helper/technicals_backtesting/backtest/batch_analytics.py` (new)
- **description**:
  - Build equal-weight aggregate equity/returns.
  - Build SPY benchmark curve aligned to analysis window.
  - Compute monthly/yearly returns + summary metrics.
- **validation**: Sparse date alignment and benchmark edge-case tests pass.
- **status**: Completed
- **log**:
  - 2026-03-03: Added pure batch analytics module with deterministic helpers to extract per-symbol daily returns from batch outcomes, build equal-weight aggregate return/equity curves with active-symbol denominators, and align SPY benchmark returns/equity to the aggregate analysis window.
  - 2026-03-03: Added monthly/yearly compounded return table generation and summary metric computation (date window, total return, ending equity, CAGR, max drawdown, annualized volatility, Sharpe) for aggregate and benchmark curves.
  - 2026-03-03: Added deterministic sparse-alignment tests for active-denominator aggregate math plus benchmark missing-edge/no-overlap handling and period-table/summary outputs.
- **files edited/created**:
  - `options_helper/technicals_backtesting/backtest/batch_analytics.py` (new)
  - `tests/test_technical_backtest_batch_analytics.py` (new)
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T7a: Batch Artifact Schema
- **depends_on**: [T6]
- **location**: `options_helper/schemas/technical_backtest_batch.py` (new), `docs/ARTIFACT_SCHEMAS.md`
- **description**:
  - Define schema-versioned `summary.json` model.
- **validation**: Contract tests validate fixtures and writer outputs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7: Artifact + HTML Writer
- **depends_on**: [T7a]
- **location**: `options_helper/data/technical_backtest_batch_artifacts.py` (new), `options_helper/reporting/technical_backtest_html.py` (new)
- **description**:
  - Write `summary.json`, CSVs, and `report.html`.
  - Render cards/charts:
    - total return, win rate, profit factor, max drawdown
    - equity + drawdown (strategy vs SPY)
    - monthly returns
    - yearly returns
- **validation**: HTML renders for normal and empty/partial datasets.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: CLI Command + Registration
- **depends_on**: [T5, T6, T7]
- **location**: `options_helper/commands/technicals/backtest_batch.py` (new), `options_helper/commands/technicals/__init__.py`
- **description**:
  - Add CLI entrypoint for multi/single ticker runs and overlay/cost flags.
  - Keep command thin; delegate to runtime module.
- **validation**: CLI help and invocation tests pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9a: Strategy Semantics Tests
- **depends_on**: [T3]
- **location**: `tests/test_mean_reversion_ibs_strategy.py` (new)
- **description**:
  - Validate entry/exit formulas and no-lookahead execution semantics.
  - Validate IBS divide-by-zero behavior.
- **validation**: Deterministic unit tests pass.
- **status**: Completed
- **log**:
  - 2026-03-03: Added deterministic regression assertions that lock contract formulas (`rolling_high`, `avg_range`, threshold, IBS, entry/exit signal bars), next-bar-open fill semantics for `trade_on_close=false` (including no-fill when a signal appears on the final bar), and IBS zero-range fallback behavior (`0.5`) that blocks entry on a zero-range bar.
- **files edited/created**:
  - `tests/test_mean_reversion_ibs_strategy.py`
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T9b: Aggregate/Benchmark Tests
- **depends_on**: [T6]
- **location**: `tests/test_technical_backtest_batch_analytics.py` (new)
- **description**:
  - Validate equal-weight math and SPY alignment behavior.
  - Validate monthly/yearly bucket generation.
- **validation**: Deterministic analytics tests pass.
- **status**: Completed
- **log**:
  - 2026-03-03: Added deterministic regression tests for sparse-overlap equal-weight denominator semantics, benchmark alignment edge behavior (left-edge return anchoring plus trailing gap handling), and monthly/yearly bucket output across multi-month/multi-year period unions.
- **files edited/created**:
  - `tests/test_technical_backtest_batch_analytics.py`
  - `docs/plans/MEAN_REVERSION_IBS-PLAN.md`

### T9c: CLI + Artifact Tests
- **depends_on**: [T8]
- **location**: `tests/test_technical_backtest_batch_cli.py` (new), `tests/test_technical_backtest_batch_artifacts.py` (new)
- **description**:
  - Validate single/multi ticker paths, partial failures, artifact files, and required HTML sections.
- **validation**: CLI/artifact tests pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9d: Config/Contract Regression Tests
- **depends_on**: [T1, T7a]
- **location**: config + schema test modules
- **description**:
  - Ensure existing technical backtesting behavior remains intact.
- **validation**: Regression suites pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9e: Reference Markdown Parity Validation Gate
- **depends_on**: [T8, T7a]
- **location**: `tests/test_mean_reversion_reference_parity.py` (new), `scripts/validate_mean_reversion_reference.py` (new), `tests/fixtures/technicals/reddit_mean_reversion_reference.json` (new)
- **description**:
  - Parse the reference markdown file to extract canonical rules and headline reference metrics.
  - Generate/refresh a normalized fixture for CI-stable parity checks.
  - Add local parity check that reads:
    `/Users/sergio/Library/Mobile Documents/iCloud~md~obsidian/Documents/Personal/Clippings/Found a simple mean reversion setup with 70% win rate but only invested 20% of the time.md`
  - Compare run output vs reference targets with explicit tolerance bands.
- **validation**:
  - Local path-based check passes when file exists.
  - CI fixture-based parity test passes deterministically.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T10: Documentation
- **depends_on**: [T8, T7a, T9e]
- **location**: `docs/technical_backtesting/MEAN_REVERSION_IBS.md` (new), `docs/technical_backtesting/RUNBOOK.md`, `docs/technical_backtesting/CONFIG_SCHEMA.md`, `docs/index.md`
- **description**:
  - Document formulas, overlays, cost precedence, benchmark semantics, and parity workflow using the reference markdown file.
- **validation**: Docs match implemented CLI/options/artifacts.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T11: End-to-End Validation + Perf Smoke
- **depends_on**: [T9a, T9b, T9c, T9d, T9e, T10]
- **location**: tests + small timing check
- **description**:
  - Run targeted pytest matrix.
  - Run local reference-path parity command.
  - Run timing smoke for multi-ticker aggregation.
- **validation**: All targeted checks pass within budget.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|---|---|---|
| 1 | T0 | Immediately |
| 2 | T1 | T0 |
| 3 | T1b, T2, T4 | T1 |
| 4 | T3 | T2 |
| 5 | T9a | T3 |
| 6 | T5 | T3, T4, T1b |
| 7 | T6 | T5 |
| 8 | T7a, T9b | T6 |
| 9 | T7, T9d | T7a (and T1 for T9d) |
| 10 | T8 | T5, T6, T7 |
| 11 | T9c, T9e | T8, T7a |
| 12 | T10 | T8, T7a, T9e |
| 13 | T11 | T9a, T9b, T9c, T9d, T9e, T10 |

## Testing Strategy
1. Entry/exit correctness on synthetic fixtures.
2. No-lookahead anchor with `trade_on_close=false`.
3. IBS zero-range handling.
4. Overlay gate toggles individually and in combination.
5. Cost precedence matrix.
6. Multi-symbol aggregate with sparse overlapping dates.
7. SPY benchmark missing/late-start alignment.
8. Artifact schema + HTML section presence.
9. Reference markdown parity:
   - rules extracted match implemented canonical formulas
   - SPY/QQQ headline metrics are within declared tolerances.

## Risks & Mitigations
1. Ambiguous signal semantics can skew parity.
   - Mitigation: T0 contract lock + deterministic regression fixtures.
2. Config compatibility breaks legacy workflows.
   - Mitigation: explicit backward-compat tests in T1b/T9d.
3. HTML report brittleness under sparse/no-trade data.
   - Mitigation: robust empty-state rendering and tests.
4. Aggregate math drift with staggered symbol coverage.
   - Mitigation: explicit denominator rule and synthetic sparse-date tests.
5. File growth in existing large command modules.
   - Mitigation: add new command/runtime modules instead of expanding existing monolith files.

## Assumptions and Defaults
1. Canonical source of truth for parity is the reference markdown file at the absolute path above.
2. If the file is unavailable in CI, fixture-based parity tests remain authoritative; local path-check is skip-with-note.
3. Strategy default has no hard stop.
4. Benchmark default is SPY.
5. Aggregate default is equal-weight daily return composite.
6. Outputs are informational research only; not financial advice.
