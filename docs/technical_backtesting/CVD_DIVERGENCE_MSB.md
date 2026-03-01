# CVD Divergence + MSB Strategy (`CvdDivergenceMSB`)

This strategy module is for research and decision-support only. It is not financial advice.

## 1) What It Does
`CvdDivergenceMSB` is a long-only setup that looks for:
1. Hidden bullish divergence between price swings and a CVD proxy.
2. A close-confirmed market structure break (MSB) before entry.

## 2) Input Requirements
- Candle columns: `Open`, `High`, `Low`, `Close`
- `Volume` is required for CVD logic
- `DatetimeIndex` must be sorted and unique
- Time handling: tz-aware timestamps are converted to UTC-naive; tz-naive timestamps are treated as UTC

## 3) CVD Proxy
The strategy computes CVD internally (it is not precomputed in `compute_features`):
- `delta[t] = Volume[t] * sign(Close[t] - Open[t])`
- doji bars contribute `0`
- `cvd = cumsum(delta)`
- detrend with EMA (`cvd_smooth_span`)
- normalize with rolling z-score (`cvd_z_window`)

Fail-soft behavior:
- Missing `Volume`, non-finite volume, or invalid volume disables entries and emits warnings.
- NaN ATR on an entry bar skips entry.

## 4) Setup + Entry Logic
### Hidden bullish divergence setup
At confirmed swing-low `pivot2`:
- Find prior confirmed swing-low `pivot1` within `divergence_window_bars` and `min_separation_bars`.
- Price condition (higher low):
  - `Low[pivot2] >= Low[pivot1] * (1 + min_price_delta_pct / 100)`
- CVD condition (lower low):
  - `cvd_z[pivot2] <= cvd_z[pivot1] - min_cvd_z_delta`
- Break level:
  - `max(High[pivot1:pivot2])`

### Entry trigger
When no position is open and setup is active:
- Optional weekly filter must pass (`weekly_trend_up == True` when enabled).
- Require `Close[t] > break_level`.
- Require `t - break_level_idx >= msb_min_distance_bars`.
- Place `buy(sl=..., tp=...)`.

With `backtest.trade_on_close: false` (default), fills occur at the next bar open.

## 5) Exit Logic
- Stop: `entry - stop_mult_atr * ATR`
- Target: `entry + take_profit_mult_atr * ATR`
- Optional time stop: close when holding bars `>= max_holding_bars` (`0` disables time stop)

## 6) No-Lookahead Rules
The implementation enforces causal timing:
- Swing pivots are only consumed after confirmation at `pivot_idx + pivot_right`.
- Breakout is evaluated on close of bar `t`; execution occurs on next bar open when `trade_on_close=false`.
- Weekly regime columns are shifted by one completed weekly bar before forward-fill, so only prior completed week data is available to the current bar.

## 7) Timezone + Resampling Semantics
- Canonical internal timestamps are UTC-naive.
- Intraday resampling uses UTC boundaries with `label="left"` and `closed="left"`.
- Weekly regime resampling also operates on the normalized UTC-naive index.

## 8) Config Surface (`config/technical_backtesting.yaml`)
`strategies.CvdDivergenceMSB` includes:
- `enabled` (default `false`)
- `defaults`: `atr_window`, `stop_mult_atr`, `take_profit_mult_atr`, `max_holding_bars`, `use_weekly_filter`, `cvd_smooth_span`, `cvd_z_window`, `pivot_left`, `pivot_right`, `divergence_window_bars`, `min_separation_bars`, `min_price_delta_pct`, `min_cvd_z_delta`, `max_setup_age_bars`, `msb_min_distance_bars`
- `search_space`
- `constraints`

## 9) CLI Usage
Daily-cache example:
```bash
./.venv/bin/options-helper technicals walk-forward \
  --strategy CvdDivergenceMSB \
  --symbol SPY \
  --cache-dir data/candles
```

Intraday example (1Min source resampled to 15m):
```bash
./.venv/bin/options-helper technicals walk-forward \
  --strategy CvdDivergenceMSB \
  --symbol SPY \
  --intraday-dir data/intraday \
  --intraday-timeframe 1Min \
  --intraday-start 2025-01-02 \
  --intraday-end 2025-01-10 \
  --interval 15m
```

## 10) Practical Caveats
- This is a best-effort signal model on imperfect market data.
- Missing intraday partitions are warning/continue, not hard-fail.
- `--interval` must be >= base intraday timeframe and an integer multiple.
- Use results for analysis support only, not execution advice.
