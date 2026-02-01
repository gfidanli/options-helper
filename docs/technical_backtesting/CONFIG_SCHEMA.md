# Configuration Schema — technical_backtesting.yaml

This file documents the expected **fields and types** in the core config for the Technical Indicators + Backtesting/Optimization feature. Maintaining this schema ensures consistency between the config and implementation.

This schema corresponds exactly to `config/technical_backtesting.yaml`. You can use this document as a reference when writing code to parse, validate, and apply configuration settings.

---

## Top-Level Keys

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `schema_version` | integer | Yes | Version of this configuration schema. |
| `timezone` | string | Yes | Default timezone (IANA format) used for datetime handling. |
| `extension_percentiles` | object | Yes | Extension percentile + tail-event settings. |

---

## `data` Section

| Field | Type | Required | Valid Values | Description |
|-------|------|----------|--------------|-------------|
| `candles.source` | string | Yes | `"yfinance"` | Source of candle data. |
| `candles.frequency` | string | Yes | e.g., `"1d"` | Frequency of bars. |
| `candles.weekly_resample_rule` | string | Yes | Pandas resample rule e.g., `"W-FRI"` | Week aggregation rule. |
| `candles.price_adjustment` | object | Yes | | Controls yfinance adjustment behavior. |
| `candles.price_adjustment.auto_adjust` | bool | Yes | | If True, prices are adjusted automatically. |
| `candles.price_adjustment.back_adjust` | bool | Yes | | If True, adjust for corporate actions. |
| `candles.required_columns` | list[string] | Yes | ["Open","High","Low","Close"] | Required OHLC columns. |
| `candles.optional_columns` | list[string] | No | ["Volume"] | Optional fields. |
| `candles.dropna_ohlc` | bool | Yes | | Drop rows missing OHLC. |
| `warmup_bars` | integer | Yes | >= 0 | Number of bars to skip before indicators are valid. |

---

## `indicators` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | Yes | `"ta"` or `"talib"` |
| `atr.window_default` | integer | Yes | Default ATR window. |
| `atr.window_grid` | list[int] | Yes | ATR grid for optimization. |
| `sma.window_default` | integer | Yes | Default SMA window. |
| `sma.window_grid` | list[int] | Yes | SMA grid. |
| `zscore.window_default` | integer | Yes | Default zscore window. |
| `zscore.window_grid` | list[int] | Yes | Zscore window grid. |
| `bollinger.window_default` | integer | Yes | BB window. |
| `bollinger.dev_default` | number | Yes | BB deviation. |
| `bollinger.window_grid` | list[int] | Yes | BB windows grid. |
| `bollinger.dev_grid` | list[number] | Yes | BB deviation grid. |
| `rsi.enabled` | bool | Yes | Enable RSI. |
| `rsi.window_default` | integer | No | Default RSI window. |
| `rsi.window_grid` | list[int] | No | RSI windows grid. |

---

## `weekly_regime` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | Yes | Whether weekly regime is computed. |
| `resample_rule` | string | Yes | Weekly resample rule. |
| `ma_type` | string | No | Weekly MA type (`"sma"` default, or `"ema"`). |
| `fast_ma` | int | Yes | Fast weekly MA. |
| `slow_ma` | int | Yes | Slow weekly MA. |
| `logic` | string | Yes | Weekly regime logic: `"close_above_fast_and_fast_above_slow"` (strict), `"close_above_fast"` (relaxed), or `"fast_above_slow"` (most relaxed). |

---

## `backtest` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `engine` | string | Yes | `"backtesting.py"`. |
| `cash` | number | Yes | Starting cash for simulation. |
| `commission` | number | Yes | Commission % per trade. |
| `trade_on_close` | bool | Yes | Execution assumption. |
| `exclusive_orders` | bool | Yes | Prevent overlapping orders. |
| `hedging` | bool | Yes | Hedging allowed (false). |
| `margin` | number | Yes | Margin multiplier. |
| `slippage_bps` | number | Yes | Slippage in bps. |

---

## `optimization` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | Yes | Run parameter optimization. |
| `method` | string | Yes | `"grid"` or `"sambo"`. |
| `maximize` | string | Yes | Which metric to maximize, e.g., `"custom_score"`. |
| `min_train_bars` | int | No | Minimum bars required in a train slice for optimization. |

### `custom_score`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `return_key` | string | Yes | Backtesting metric key for return. |
| `max_drawdown_key` | string | Yes | Metric key for drawdown. |
| `trades_key` | string | Yes | Metric key for trade count. |
| `weights.drawdown_lambda` | number | Yes | Weight on drawdown penalty. |
| `weights.turnover_mu` | number | Yes | Weight on turnover. |
| `min_trades` | int | Yes | Minimum required trades for valid score. |

### `sambo`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_tries` | int | Yes | Max randomized tries. |
| `random_state` | int | Yes | Seed for reproducibility. |

---

## `walk_forward` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | Yes | Whether walk-forward is enabled. |
| `train_years` | number | Yes | Years for training. |
| `validate_months` | number | Yes | Months for validation. |
| `step_months` | number | Yes | Step size. |
| `min_history_years` | number | Yes | Minimum bars required. |

### `selection`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metric` | string | Yes | Metric for selecting best params. |

### `selection.stability`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_validate_score_cv` | number | Yes | Max CV threshold for validate scores. |

---

## `calibration` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mode` | string | Yes | `"per_ticker"` or `"vol_bucket"`. |

### `vol_bucket`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | Yes | Enable vol buckets. |
| `atrp_window` | int | Yes | ATR% window. |
| `buckets` | object | Yes | ATR% based buckets. |

---

## `strategies` Section (Per Strategy)

For each strategy (`TrendPullbackATR`, `MeanReversionBollinger`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enabled` | bool | Yes | Whether strategy is active. |
| `defaults` | object | Yes | Default values for parameters. |
| `search_space` | object | Yes | Grid definitions for optimization. |
| `constraints` | list[string] | Yes | Constraint expressions applied in optimizer. |

---

## `artifacts` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `base_dir` | string | Yes | Root dir for all output artifacts. |
| `overwrite` | bool | Yes | Overwrite existing artifacts? |
| `write_heatmap` | bool | Yes | Write optimization heatmap CSV? |
| `write_trades` | bool | Yes | Write individual trade detail? |
| `params_path_template` | string | Yes | Template for params JSON path. |
| `report_path_template` | string | Yes | Template for summary report. |
| `heatmap_path_template` | string | Yes | Template for heatmap CSV. |

---

## `logging` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `level` | string | Yes | Logging verbosity (`DEBUG`, `INFO`, etc.). |
| `log_dir` | string | Yes | Directory for logs. |

---

## `extension_percentiles` Section

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `days_per_year` | int | Yes | Trading days per year (default 252). |
| `windows_years` | list[int] | Yes | Rolling percentile windows (years). |
| `tail_high_pct` | number | Yes | Upper tail threshold (percentile). |
| `tail_low_pct` | number | Yes | Lower tail threshold (percentile). |
| `forward_days` | list[int] | Yes | Forward windows (trading-day offsets). |
| `windows_years` | list[int] | Yes | Rolling windows in years (default [3]). |

### Notes

- This document is a human-readable schema; it is *not* a machine schema (like JSON Schema). You can generate a machine schema from this if you need IDE validation or stricter enforcement.  [oai_citation:0‡Red Hat Developer](https://developers.redhat.com/blog/2020/11/25/how-to-configure-yaml-schema-to-make-editing-files-easier?utm_source=chatgpt.com)
