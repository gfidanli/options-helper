# Swing Failure Pattern (SFP) Scan

This feature detects **swing highs/lows** and flags **swing failure pattern (SFP)** candles from OHLC data.

This project is **not financial advice**. Outputs are descriptive signals for research and position-management workflows.

## Signal workflow diagram

![SFP detection workflow](assets/diagrams/generated/sfp_detection_flow.svg)

---

## 1) Definitions

- Swing high: local high using configurable left/right bars.
- Swing low: local low using configurable left/right bars.
  - Defaults: `swing_left_bars=2`, `swing_right_bars=2`
  - Example with defaults: a swing high must be at/above the highs of the prior 2 bars and next 2 bars (same idea for swing low with lows).
- Bearish SFP:
  - Candle wick breaks **above** the latest prior swing high (`High > swing_high_level`)
  - Candle closes **back below** that swing high (`Close < swing_high_level`)
- Bullish SFP:
  - Candle wick breaks **below** the latest prior swing low (`Low < swing_low_level`)
  - Candle closes **back above** that swing low (`Close > swing_low_level`)

---

## 2) CLI command

```bash
./.venv/bin/options-helper technicals sfp-scan --symbol CVX --cache-dir data/candles --out data/reports/technicals/sfp --print
```

You can also run from a local OHLC file:

```bash
./.venv/bin/options-helper technicals sfp-scan --ohlc-path /path/to/ohlc.csv --swing-left-bars 2 --swing-right-bars 2
```

Optional timeframe resampling:

```bash
./.venv/bin/options-helper technicals sfp-scan --symbol CVX --timeframe W-FRI
```

Weekly note:
- Weekly resampled candles are labeled by the **week start (Monday)** in artifacts/output.

Key knobs:
- `--swing-left-bars`, `--swing-right-bars`
- `--min-swing-distance-bars`
- `--timeframe` (`native`, `W-FRI`, `2W-FRI`, `4H`, etc. when datetime-indexed data is available)
- `--rsi-window`, `--rsi-overbought`, `--rsi-oversold`
- `--include-rsi-divergence/--no-rsi-divergence`

---

## 3) Artifacts

By default the command writes:
- `data/reports/technicals/sfp/{SYMBOL}/{ASOF}.json`
- `data/reports/technicals/sfp/{SYMBOL}/{ASOF}.md`

JSON includes:
- scan config
- swing/SFP counts
- latest bullish/bearish SFP events
- full event list with:
  - date-only event/swing timestamps
  - prices rounded to 2 decimals
  - forward returns anchored to the **next candle open** (`1d`, `5d`, `10d`) in percent
  - entry anchor metadata (`entry_anchor_timestamp`, `entry_anchor_price`)
  - extension context (`extension_atr`, `extension_percentile`)
  - optional RSI regime and same-bar RSI divergence tags

---

## 4) Combining with other metrics

SFPs are usually strongest when combined with context, for example:
- extension percentiles/tail states
- RSI extremes (`overbought` / `oversold`)
- RSI divergence alignment
- higher-timeframe structure (weekly trend, weekly levels)

Use this command as a first pass to surface candidate reversal zones, then layer additional evidence.
