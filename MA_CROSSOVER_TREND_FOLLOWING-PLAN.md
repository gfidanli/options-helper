# Plan: Add `ma_crossover` and `trend_following` to Strategy Modeling Suite

## Summary
Implement two new modelable strategies in the existing strategy-modeling pipeline (`technicals strategy-model` + Streamlit page), with deterministic anti-lookahead semantics and existing portfolio simulation flow unchanged.

Locked decisions:
- Surface: strategy-model CLI + Streamlit.
- New strategies: `ma_crossover`, `trend_following`.
- Directionality: long + short symmetric.
- Risk model: ATR-based stop at signal time, then existing target ladder/time-stop simulation.
- Default windows: crossover `20/50`, trend MA `200`.
- MA configurability: fast/slow/trend windows and MA type (`sma`/`ema`) are all user-configurable.

Out of scope for this milestone:
- `technicals_backtesting` strategy registry (`TrendPullbackATR`, `MeanReversionBollinger`) changes.
- New filter families beyond current entry filters.

## Public API / Interface Changes
1. `StrategyId` expansion in `options_helper/schemas/strategy_modeling_contracts.py`
- Add literals: `"ma_crossover"`, `"trend_following"`.

2. CLI strategy enum in `options_helper/commands/technicals.py`
- Extend `--strategy` valid set to: `sfp|msb|orb|ma_crossover|trend_following`.
- Add strategy-specific signal parameters:
  - `--ma-fast-window` (int, default `20`)
  - `--ma-slow-window` (int, default `50`)
  - `--ma-trend-window` (int, default `200`)
  - `--ma-fast-type` (`sma|ema`, default `sma`)
  - `--ma-slow-type` (`sma|ema`, default `sma`)
  - `--ma-trend-type` (`sma|ema`, default `sma`)
  - `--trend-slope-lookback-bars` (int, default `3`)
  - `--atr-window` (int, default `14`)
  - `--atr-stop-multiple` (float, default `2.0`)
- Validate:
  - all windows `>=1`
  - `ma_fast_window < ma_slow_window` for `ma_crossover`
  - `atr_stop_multiple > 0`
  - MA types in `{sma, ema}`

3. Streamlit strategy input in `apps/streamlit/pages/11_Strategy_Modeling.py`
- Add strategy options: `ma_crossover`, `trend_following`.
- Add matching controls for MA/ATR parameters and pass via `signal_kwargs` in request.

4. Signal adapter registry in `options_helper/analysis/strategy_signals.py`
- Register two new adapters:
  - `adapt_ma_crossover_signal_events`
  - `adapt_trend_following_signal_events`

## Implementation Design

### 1) New pure signal modules
Create:
- `options_helper/analysis/ma_crossover.py`
- `options_helper/analysis/trend_following.py`

Both modules will:
- Normalize OHLC input (reuse `normalize_ohlc_frame` style).
- Compute MA series with helper supporting `sma`/`ema`.
- Produce deterministic candidate rows with `row_position`.
- Emit close-confirmed signal rows only.
- Never call network/filesystem.

### 2) Strategy definitions (decision-complete)

#### `ma_crossover`
- Inputs: fast/slow window + type, ATR window, ATR multiple.
- Long signal at bar `t` when:
  - `fast_ma[t-1] <= slow_ma[t-1]` and `fast_ma[t] > slow_ma[t]`.
- Short signal at bar `t` when:
  - `fast_ma[t-1] >= slow_ma[t-1]` and `fast_ma[t] < slow_ma[t]`.
- `signal_ts = t`, `signal_confirmed_ts = t`.
- `entry_ts = t+1` bar timestamp; skip if no next bar.
- Stop:
  - long: `stop_price = close[t] - atr_multiple * atr[t]`
  - short: `stop_price = close[t] + atr_multiple * atr[t]`
- Notes include MA params, ATR params, and explicit `entry_ts_policy=next_bar_open_after_signal_confirmed_ts`.

#### `trend_following`
- Inputs: trend window/type, optional fast window/type for alignment, slope lookback, ATR window/multiple.
- Long signal at bar `t` when all true:
  - `close[t-1] <= trend_ma[t-1]` and `close[t] > trend_ma[t]` (price crosses into up regime)
  - trend slope over lookback `>= 0`
  - fast MA alignment: `fast_ma[t] >= trend_ma[t]`
- Short signal mirrors above:
  - `close[t-1] >= trend_ma[t-1]` and `close[t] < trend_ma[t]`
  - trend slope `<= 0`
  - fast MA alignment: `fast_ma[t] <= trend_ma[t]`
- Anti-lookahead and stop semantics identical to `ma_crossover`.

### 3) Adapter integration
In `options_helper/analysis/strategy_signals.py`:
- Add normalizers for each strategy that map raw candidate rows to `StrategySignalEvent`.
- Enforce:
  - `entry_ts > signal_confirmed_ts`
  - final-bar signals are skipped
  - symbol/timeframe normalization parity with existing adapters.

### 4) Request wiring
In `options_helper/commands/technicals.py`:
- Build `signal_kwargs` for new strategies and pass into `StrategyModelingRequest(..., signal_kwargs=...)`.
- Keep existing strategies unchanged.
- Keep ORB confirmation filter behavior scoped to SFP/MSB only (already strategy-gated in filters).

In `apps/streamlit/pages/11_Strategy_Modeling.py`:
- Add equivalent UI fields and pass `signal_kwargs` in `StrategyModelingRequest`.

### 5) Docs updates
Update `docs/TECHNICAL_STRATEGY_MODELING.md`:
- CLI strategy list includes new strategy IDs.
- Parameter table for MA/ATR knobs.
- Explicit anti-lookahead section for both.
- Example commands for each strategy.
- Maintain disclaimer language.

## Test Plan

### Contract and schema
- `tests/test_strategy_modeling_contracts.py`
  - verify `StrategySignalEvent`/`StrategyTradeSimulation` accept new strategy IDs.

### Signal adapters
- `tests/test_strategy_signals.py`
  - registry includes `ma_crossover` + `trend_following`.
  - long/short signal generation fixtures.
  - anti-lookahead assertions:
    - `entry_ts > signal_confirmed_ts`
    - last-bar signal skipped.
  - MA type coverage (`sma` and `ema`).
  - ATR stop presence/shape.

### CLI
- `tests/test_strategy_modeling_cli.py`
  - strategy validation message includes new values.
  - command accepts each new strategy.
  - request captures `signal_kwargs` values correctly.
  - invalid MA/ATR inputs raise `typer.BadParameter`.

### Service parity
- `tests/test_strategy_modeling_service.py`
  - deterministic run with each new strategy through full service path using stubbed data loaders.

### Streamlit
- `tests/portal/test_strategy_modeling_page.py`
  - strategy selectbox includes new options.
  - request contains new `signal_kwargs` when selected.
  - existing blocked/preflight behavior remains unchanged.

### Performance guardrail
- Add targeted check in `tests/test_strategy_modeling_performance.py`:
  - signal generation for new strategies on synthetic multi-symbol daily set remains within a bounded runtime budget comparable to current adapters.

## Rollout / Compatibility
- Additive only: existing strategies/artifacts remain valid.
- No artifact schema bump required if payload shape unchanged (strategy value is additive enum extension).
- Existing filters/portfolio simulation stay untouched; only signal source expands.

## Assumptions and Defaults
- Default params:
  - `ma_fast_window=20`, `ma_slow_window=50`, `ma_trend_window=200`
  - `ma_fast_type=sma`, `ma_slow_type=sma`, `ma_trend_type=sma`
  - `trend_slope_lookback_bars=3`
  - `atr_window=14`, `atr_stop_multiple=2.0`
- New strategies are modeled on daily signal bars with next-bar entry anchoring consistent with current anti-lookahead policy.
- ORB confirmation gate is not applied to these new strategies in v1.
