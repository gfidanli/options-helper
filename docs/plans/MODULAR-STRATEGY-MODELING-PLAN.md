# Modular Strategy Modules (SFP/MSB/ORB) + Entry Filters + Directional Results - PLAN

## Summary
Build a modular strategy-modeling layer where `sfp`, `msb`, and `orb` can run as primary strategies, and where optional entry filters can be combined as gates.

Locked decisions:
- Composition model: base strategy + optional gates.
- ORB mode: standalone strategy + confirmation gate.
- ORB definition: 15-minute opening-range breakout by close.
- ORB entry timing: first bar after ORB confirmation.
- ORB cutoff: configurable, default 10:30 ET.
- Volatility regime method: existing realized-vol percentile.
- RSI extreme filter: side-aware (long=oversold, short=overbought).
- EMA regime filter: close vs 9EMA + 9EMA slope.
- ATR floor: initial risk >= 0.5 ATR.
- Short toggle: default ON.
- Directional reporting: counterfactual reruns for combined/long-only/short-only.
- Optimization in this phase: no sweeps, config-driven only.
- New filters default: OFF.

## Scope and Non-Goals
- In scope: modular strategy execution, ORB implementation, configurable entry filters, short toggle, directional result reporting in CLI/artifacts.
- In scope: additive schema/contract updates required for `orb` and filter metadata.
- Out of scope: automatic parameter search/grid sweep and optimization orchestration.

## Public Interface Changes

### CLI (`options-helper technicals strategy-model`)
Additive flags:
- `--strategy sfp|msb|orb`
- `--allow-shorts/--no-allow-shorts` (default `--allow-shorts`)
- `--enable-orb-confirmation/--no-enable-orb-confirmation` (default off)
- `--orb-range-minutes` (default `15`)
- `--orb-confirmation-cutoff-et` (default `10:30`)
- `--enable-atr-stop-floor/--no-enable-atr-stop-floor` (default off)
- `--atr-stop-floor-multiple` (default `0.5`)
- `--enable-rsi-extremes/--no-enable-rsi-extremes` (default off)
- `--enable-ema9-regime/--no-enable-ema9-regime` (default off)
- `--ema9-slope-lookback-bars` (default `3`)
- `--enable-volatility-regime/--no-enable-volatility-regime` (default off)
- `--allowed-volatility-regimes low,normal,high` (default all three)

### Contracts / Types
- Update `StrategyId` to include `"orb"` in `options_helper/schemas/strategy_modeling_contracts.py`.
- Add strict filter config model (new schema) for reuse by CLI/service:
  - `StrategyEntryFilterConfig` with all flags and thresholds above.
- Extend `StrategyModelingRequest` with `filter_config`.
- Extend run/artifact payloads with additive fields:
  - `filter_metadata` (effective filter settings)
  - `filter_summary` (event counts by pass/fail reason)
  - `directional_metrics` (combined, long_only, short_only)

## Architecture and Implementation Steps

## 1) ORB module implementation (analysis)
Files:
- New `options_helper/analysis/orb.py`
- Update `options_helper/analysis/strategy_signals.py`

Behavior:
- Opening range: regular session bars in `09:30:00 <= ts < 09:30 + range_minutes` (ET).
- Confirmation: first later bar with:
  - bullish ORB: `close > opening_range_high`
  - bearish ORB: `close < opening_range_low`
- Confirmation must occur by cutoff ET (default `10:30`), configurable.
- Standalone ORB event fields:
  - `strategy="orb"`
  - `direction` from breakout side
  - `signal_ts = breakout bar ts`
  - `signal_confirmed_ts = breakout bar ts`
  - `entry_ts = first tradable bar after breakout bar`
  - `stop_price = opening_range_low` for long, `opening_range_high` for short
  - notes include opening-range highs/lows, range window, cutoff, breakout timestamp.

## 2) Entry filter framework (analysis + schema)
Files:
- New `options_helper/schemas/strategy_modeling_filters.py`
- New parse/resolve helper in `options_helper/analysis/strategy_modeling_filters.py`
- Update `options_helper/analysis/strategy_modeling.py`

Pipeline changes:
- Keep base event generation by primary strategy (`sfp|msb|orb`).
- Add filter stage before simulation that evaluates enabled filters with AND logic.
- Filter order (cheap-to-expensive): shorts toggle -> ATR floor -> RSI extremes -> EMA9 regime -> volatility regime -> ORB confirmation.
- Maintain audit rows per event with pass/fail reason.

Filter definitions:
- Short toggle: if disabled, drop short-direction events.
- ATR floor: require `initial_risk / ATR_signal >= atr_stop_floor_multiple`.
  - `ATR_signal` is daily ATR at signal candle close.
- RSI extremes:
  - long requires `rsi <= rsi_oversold`
  - short requires `rsi >= rsi_overbought`
- EMA9 regime:
  - long requires `close > ema9` and `ema9_slope > 0`
  - short requires `close < ema9` and `ema9_slope < 0`
  - slope = `ema9[t] - ema9[t-lookback]`, lookback default `3`.
- Volatility regime:
  - use existing `realized_vol_regime` (`low/normal/high`)
  - require event regime in `allowed_volatility_regimes`.
- ORB confirmation gate (for `sfp/msb` base events):
  - evaluate next session ORB confirmation in event direction
  - if pass, set combined `signal_confirmed_ts` to ORB confirmation ts and set `entry_ts` to first bar after confirmation
  - if fail by cutoff, event is filtered out with reason.

## 3) Feature enrichment updates
Files:
- Update `options_helper/analysis/strategy_features.py`

Add fields:
- `ema9`
- `ema9_slope` (lookback-configurable in filter config; computed as needed in filter stage)
Keep existing realized-vol percentile regime fields unchanged.

## 4) Service orchestration and reporting model
Files:
- Update `options_helper/analysis/strategy_modeling.py`
- Update `options_helper/data/strategy_modeling_artifacts.py`
- Update `options_helper/commands/technicals.py`

Changes:
- Resolve `filter_config` in request and apply deterministic filter stage.
- Track and return:
  - base event count
  - filtered event count
  - counts by filter-reason
- Directional counterfactual results when shorts enabled:
  - recompute portfolio+metrics with long-only trades
  - recompute portfolio+metrics with short-only trades
  - keep existing combined results unchanged.
- CLI output:
  - print filter summary counts
  - print combined + long-only + short-only headline metrics when shorts enabled.
- Artifacts (`summary.json` and `summary.md`):
  - include `filter_metadata`, `filter_summary`, `directional_metrics`.

## 5) CLI parsing and wiring
Files:
- Update `options_helper/commands/technicals.py`
- Update `tests/test_strategy_modeling_cli.py`

Changes:
- Accept `orb` as valid strategy.
- Parse/validate new filter flags and time cutoff.
- Construct `filter_config` and pass into `StrategyModelingRequest`.
- Preserve backward compatibility: all new filters default off, shorts default on.

## 6) Documentation updates
Files:
- Update `docs/TECHNICAL_STRATEGY_MODELING.md`
- Optional cross-link from docs index if needed

Add:
- ORB strategy and ORB confirmation-gate semantics
- Exact filter formulas and defaults
- Short toggle behavior
- Directional counterfactual result definitions
- Clarification that this remains informational, not financial advice

## Test Plan

## Unit tests
- `tests/test_orb.py` (new):
  - opening-range construction
  - breakout detection by close
  - cutoff handling
  - no-breakout cases
- `tests/test_strategy_features.py`:
  - `ema9` and slope correctness
- `tests/test_strategy_signals.py`:
  - registry includes `orb`
  - ORB standalone events satisfy anti-lookahead fields and contract parity.

## Service/integration tests
- `tests/test_strategy_modeling_service.py`:
  - SFP/MSB + ORB gate shifts `signal_confirmed_ts`/`entry_ts` correctly
  - filter ordering and reason accounting
  - shorts disabled removes short events
  - directional counterfactual metrics shape/content.
- `tests/test_strategy_modeling_regression.py`:
  - no lookahead in ORB gate (entry strictly after ORB confirmation)
  - ORB cutoff skip behavior
  - EMA/RSI/ATR/volatility gate regressions.

## CLI and artifact tests
- `tests/test_strategy_modeling_cli.py`:
  - new flags parse/validation
  - strategy `orb` accepted
  - shorts toggle and directional output lines.
- `tests/test_strategy_modeling_artifacts.py`:
  - JSON includes `filter_metadata`, `filter_summary`, `directional_metrics`
  - markdown includes combined/long-only/short-only section
  - timezone formatting remains correct.

## Performance/smoke
- Re-run targeted suites:
  - `tests/test_strategy_simulator.py`
  - `tests/test_strategy_modeling_policy.py`
  - `tests/test_strategy_modeling_service.py`
  - `tests/test_strategy_modeling_cli.py`
  - `tests/test_strategy_modeling_artifacts.py`
- Run representative command with `--show-progress` and verify stage timings remain acceptable.

## Explicit Assumptions and Defaults
- All new filters are opt-in (`False` by default) to preserve baseline behavior.
- Shorts are enabled by default (`allow_short_trades=True`).
- ORB confirmation uses ET session clock, default cutoff `10:30`.
- ORB confirmation gate and standalone ORB both use breakout-by-close logic.
- Directional result blocks are counterfactual reruns, not same-run slices.
- No automatic parameter optimization is included in this phase.
