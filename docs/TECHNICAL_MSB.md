# Market Structure Break (MSB) Scan

This feature detects **swing highs/lows** and flags **market structure breaks (MSB)** from OHLC data.

This project is **not financial advice**. Outputs are descriptive signals for research and position-management workflows.

## Signal workflow diagram

![MSB detection workflow](assets/diagrams/generated/msb_detection_flow.svg)

---

## 1) Definitions

- Swing high: local high using configurable left/right bars.
- Swing low: local low using configurable left/right bars.
  - Defaults: `swing_left_bars=2`, `swing_right_bars=2`
- Bullish MSB:
  - Candle close breaks **above** the latest prior swing high (`Close > swing_high_level`)
- Bearish MSB:
  - Candle close breaks **below** the latest prior swing low (`Close < swing_low_level`)

MSB events are recorded on the break candle (close-through), not every bar that remains above/below the level.

---

## 2) CLI command

```bash
./.venv/bin/options-helper technicals msb-scan --symbol CVX --cache-dir data/candles --out data/reports/technicals/msb --print
```

You can also run from a local OHLC file:

```bash
./.venv/bin/options-helper technicals msb-scan --ohlc-path /path/to/ohlc.csv --swing-left-bars 2 --swing-right-bars 2
```

Optional timeframe resampling:

```bash
./.venv/bin/options-helper technicals msb-scan --symbol CVX --timeframe W-FRI
```

Weekly note:
- Weekly resampled candles are labeled by the **week start (Monday)** in artifacts/output.

Key knobs:
- `--swing-left-bars`, `--swing-right-bars`
- `--min-swing-distance-bars`
- `--timeframe` (`native`, `W-FRI`, `2W-FRI`, `4H`, etc. when datetime-indexed data is available)
- `--rsi-window`, `--rsi-overbought`, `--rsi-oversold`

---

## 3) Artifacts

By default the command writes:
- `data/reports/technicals/msb/{SYMBOL}/{ASOF}.json`
- `data/reports/technicals/msb/{SYMBOL}/{ASOF}.md`

JSON includes:
- scan config
- swing/MSB counts
- latest bullish/bearish MSB events
- full event list with:
  - date-only event/swing timestamps
  - prices rounded to 2 decimals
  - forward returns anchored to the **next candle open** (`1d`, `5d`, `10d`) in percent
  - entry anchor metadata (`entry_anchor_timestamp`, `entry_anchor_price`)
  - extension context (`extension_atr`, `extension_percentile`)
  - optional RSI regime tags

---

## 4) Combining with other metrics

MSB events are usually stronger when combined with context, for example:
- extension percentiles/tail states
- RSI extremes (`overbought` / `oversold`)
- higher-timeframe structure (weekly trend, weekly levels)

Use this command as a first pass to surface structural break candidates, then layer additional evidence.
