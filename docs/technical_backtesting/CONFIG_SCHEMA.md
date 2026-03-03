# Configuration Schema (`technical_backtesting.yaml`)

This document summarizes the runtime contract for `config/technical_backtesting.yaml` and `config/technical_backtesting.schema.json`.

This configuration powers research tooling only (not financial advice).

## 1) Top-Level Sections
Required top-level keys:
- `schema_version`
- `timezone`
- `data`
- `indicators`
- `weekly_regime`
- `backtest`
- `optimization`
- `walk_forward`
- `calibration`
- `strategies`
- `artifacts`
- `logging`
- `extension_percentiles`

## 2) Data
`data.candles`:
- `source` (string)
- `frequency` (string)
- `weekly_resample_rule` (string)
- `price_adjustment.auto_adjust` (bool)
- `price_adjustment.back_adjust` (bool)
- `required_columns` (array of strings)
- `optional_columns` (array of strings, optional)
- `dropna_ohlc` (bool)

`data.warmup_bars`:
- integer `>= 0`

Validation rule:
- `auto_adjust` and `back_adjust` cannot both be `true`.

## 3) Indicators
Required blocks:
- `provider` (`ta` or `talib`)
- `atr` (`window_default`, `window_grid`)
- `sma` (`window_default`, `window_grid`)
- `zscore` (`window_default`, `window_grid`)
- `bollinger` (`window_default`, `dev_default`, `window_grid`, `dev_grid`)
- `rsi` (`enabled`, optional windows)

## 4) Weekly Regime
Fields:
- `enabled`
- `resample_rule`
- `ma_type` (optional, defaults to `sma` in runtime)
- `fast_ma`
- `slow_ma`
- `logic`

No-lookahead runtime behavior:
- weekly values are shifted by one completed week before forward-fill.

## 5) Backtest
Required fields:
- `engine`, `cash`, `commission`, `trade_on_close`, `exclusive_orders`, `hedging`, `margin`, `slippage_bps`

## 6) Optimization
Required:
- `enabled`, `method`, `maximize`

Optional:
- `min_train_bars`
- `custom_score.*`
- `sambo.*`

## 7) Walk-Forward
Supported modes:
- Date mode: `train_years`, `validate_months`, `step_months`, optional `min_history_years`
- Bar mode: `train_bars`, `validate_bars`, `step_bars`, optional `min_history_bars`

Bar mode is enabled when `train_bars > 0`.

Validation invariants:
- If `train_bars > 0`, then `validate_bars > 0`.
- If `train_bars > 0` and `step_bars` is present, then `step_bars > 0`.
- If `train_bars > 0` and `min_history_bars > 0`, then `min_history_bars >= train_bars + validate_bars`.
- If `train_bars == 0`, then `validate_bars` and `step_bars` must both be `0`.

## 8) Strategies
Each strategy entry requires:
- `enabled`
- `defaults`
- `search_space`
- `constraints`

Optional per-strategy cost block:
- `cost_overrides.commission` (number, `>= 0`)
- `cost_overrides.slippage_bps` (number, `>= 0`)

Validation notes:
- `cost_overrides` must be an object when present.
- Only `commission` and `slippage_bps` are allowed keys.
- Unknown keys or negative values fail config loading.

Default config includes:
- `TrendPullbackATR`
- `MeanReversionBollinger`
- `MeanReversionIBS`
- `CvdDivergenceMSB` (disabled by default)

`MeanReversionIBS` strategy block:
- Canonical defaults:
  - `lookback_high`
  - `range_window`
  - `range_mult`
  - `ibs_threshold`
  - `exit_lookback`
- Optional overlay/default fields used by strategy runtime:
  - `use_sma_trend_gate`, `sma_trend_window`
  - `use_weekly_trend_gate`, `weekly_trend_col`
  - `use_ma_direction_gate`, `ma_direction_window`, `ma_direction_lookback`

Backtest-batch runtime precedence for effective costs:
1. CLI (`--commission`, `--slippage-bps`)
2. `strategies.<strategy>.cost_overrides`
3. `backtest.commission` and `backtest.slippage_bps`

## 9) Artifacts
Fields:
- `base_dir`
- `overwrite`
- `write_heatmap`
- `write_trades`
- `params_path_template`
- `report_path_template`
- `heatmap_path_template`

Templates support `{ticker}`, `{strategy}`, and `{interval}` tokens.

## 10) Extension Percentiles
Fields:
- `days_per_year` (int >= 1)
- `windows_years` (non-empty int array)
- `tail_high_pct` (50..100)
- `tail_low_pct` (0..50)
- `forward_days` (non-empty int array)
- optional: `forward_days_daily`, `forward_days_weekly`

Validation rules:
- `tail_low_pct < tail_high_pct`
- arrays above must be non-empty where required.

## 11) Timezone Assumptions
`timezone` is a config metadata field. Runtime OHLC normalization and intraday loader behavior are UTC-based:
- tz-aware timestamps are normalized to UTC-naive
- tz-naive timestamps are treated as UTC
- weekly/intraday resampling boundaries therefore use UTC-normalized timestamps
