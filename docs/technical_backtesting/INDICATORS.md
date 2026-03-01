# Indicator Catalog

This subsystem is for research and decision-support only. It is not financial advice.

## 1) Core Computed Features
The default provider is `ta`.

### ATR / ATR%
- `atr_<window>` from `AverageTrueRange`
- `atrp_<window> = atr_<window> / Close`

### SMA
- `sma_<window> = rolling mean(Close)`

### Z-score
- `zscore_<window> = (Close - rolling_mean) / rolling_std`
- `rolling_std == 0` is treated as `NaN`

### Bollinger
For each configured window/dev pair:
- `bb_mavg_<w>`
- `bb_hband_<w>_<dev>`
- `bb_lband_<w>_<dev>`
- `bb_pband_<w>_<dev>`
- `bb_wband_<w>_<dev>`

### RSI (optional)
- `rsi_<window>` when `indicators.rsi.enabled=true`

### Extension
For each SMA/ATR combination:
- `extension_atr_<sma_window>_<atr_window> = (Close - sma) / atr`

## 2) Weekly Regime Columns
Weekly regime is computed from resampled OHLC (`weekly_regime.resample_rule`) and projected back to the base timeframe.

Columns:
- `weekly_sma_<fast_ma>` (or EMA when `ma_type=ema`)
- `weekly_sma_<slow_ma>`
- `weekly_trend_up`

`weekly_trend_up` logic is controlled by `weekly_regime.logic`:
- `close_above_fast_and_fast_above_slow`
- `close_above_fast`
- `fast_above_slow`

## 3) No-Lookahead Weekly Semantics
Weekly series are shifted by one completed weekly bar before forward-fill.
- Current bars can only consume prior completed week values.
- This applies to both `weekly_sma_*` and `weekly_trend_up`.

## 4) Timezone Semantics
- Canonical index for indicators is UTC-naive.
- tz-aware indexes are converted to UTC and made tz-naive.
- tz-naive indexes are treated as UTC.
- Weekly resampling boundaries therefore operate on UTC-normalized timestamps.

## 5) Strategy-Internal Series
`CvdDivergenceMSB` computes CVD-derived series internally:
- signed volume delta
- cumulative CVD
- EMA-detrended oscillator
- rolling z-score (`cvd_z`)

These are strategy internals and are not part of the standard feature frame contract.

## 6) Warmup and NaNs
- Rolling indicators naturally emit NaNs in early bars.
- Backtests should run with adequate warmup (`data.warmup_bars`) so entries occur only after required inputs are available.
