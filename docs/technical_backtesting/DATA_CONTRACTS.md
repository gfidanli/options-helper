# Data Contracts — Candles for Indicators + Backtesting

## 1) CandleFrame Contract (Input to this feature)

### Required columns
- `Open`, `High`, `Low`, `Close`
- `Volume` is optional (but recommended)

### Index
- DatetimeIndex (timezone-naive OK, but must be consistent)
- Sorted ascending
- Unique timestamps

### Frequency
- Daily bars are the primary input.
- Weekly bars are derived via resampling (weekly alignment must be documented and consistent).

## 2) yfinance Adjustment Settings (must be explicit)

yfinance can adjust OHLC for corporate actions depending on settings.

### yfinance.download
- Parameter `auto_adjust` controls whether OHLC are adjusted (default True).
- Parameter `back_adjust` can be used to “mimic true historical prices”.

**Policy for this project:**
- Always set `auto_adjust` and `back_adjust` explicitly in the upstream fetch code.
- Persist those settings in backtest artifact metadata.

Recommended defaults for technical backtests:
- `auto_adjust=True`, `back_adjust=False` (robust long-history continuity)
Alternative for “raw tape”:
- `auto_adjust=False`, `back_adjust=False` (requires handling splits/divs effects)

## 3) Adapter Rules (yfinance -> CandleFrame)
- If input columns include extras (e.g., `Dividends`, `Stock Splits`), ignore/drop unless explicitly configured to keep them.
- Normalize column names to `Open/High/Low/Close/Volume` (title case).
- Ensure float dtype for price columns; int/float for volume.
- Remove rows with all-NaN OHLC.
- Validate High >= max(Open, Close) and Low <= min(Open, Close) for non-NaN rows (warn; don’t hard-fail unless configured).

## 4) Output Frames

### FeatureFrame
CandleFrame + computed columns such as:
- `atr_14`, `atrp_14`
- `bb_mavg_20`, `bb_hband_20_2`, `bb_lband_20_2`, `bb_pband_20_2`, `bb_wband_20_2`
- `rsi_14`
- `sma_20`, `zscore_20`, `extension_atr_20_14`
- `weekly_sma_10`, `weekly_sma_20`, `weekly_trend_up` (boolean)

## 5) References
- yfinance download API docs:
  https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- yfinance PriceHistory.history docs:
  https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
- backtesting.py data requirements:
  https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html