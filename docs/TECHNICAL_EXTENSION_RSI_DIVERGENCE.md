# RSI Divergences During Extension (F-008)

This feature enriches the extension percentile reports with **RSI(14) divergence** detection to help distinguish:
- “Extended but momentum still confirming” vs
- “Extended and momentum weakening” (classic divergence)

This project is **not financial advice**. These outputs are descriptive and intended to support research and
position-management workflows.

---

## 1) What it detects (daily + weekly bars)

We look for divergences over a rolling window (default **14 bars**), anchored on the newer swing point.

Notes:
- On **daily** data, a “bar” is a trading day.
- On **weekly** data, a “bar” is a week (week-end close per your resample rule).

### Bearish divergence (upside extension context)
- Close makes a **higher swing high**.
- RSI(14) makes a **lower high** at those swing highs.

### Bullish divergence (downside extension context)
- Close makes a **lower swing low**.
- RSI(14) makes a **higher low** at those swing lows.

Swing points are defined deterministically:
- swing high at bar `i` if `Close[i] >= Close[i-1]` and `Close[i] >= Close[i+1]`
- swing low at bar `i` if `Close[i] <= Close[i-1]` and `Close[i] <= Close[i+1]`

Because swing points require `i+1`, the newest candle cannot be a “confirmed” swing until the next candle exists.

---

## 2) “Extended” gating (percentile-based)

To avoid tagging every small divergence, divergence detection is gated on extension percentiles:
- bearish divergence requires extension percentile to be **high** for at least `min_extension_days` in the window
- bullish divergence requires extension percentile to be **low** for at least `min_extension_days` in the window

Defaults come from your `extension_percentiles` tail thresholds (e.g. `tail_high_pct` / `tail_low_pct`).

---

## 3) RSI regime tagging (overbought/oversold)

Each divergence is tagged by the RSI value at the newer swing point:
- `overbought` if RSI >= `rsi_overbought` (default 70)
- `oversold` if RSI <= `rsi_oversold` (default 30)
- otherwise `neutral`

Optional gating:
- `--require-rsi-extreme` only emits bearish divergences at `overbought` and bullish divergences at `oversold`.

---

## 4) Where it appears

Currently, divergence enrichment is surfaced in:
- `options-helper technicals extension-stats` artifacts:
  - “RSI Divergence (Daily)” section (current recent-window divergence)
  - “RSI Divergence (Weekly)” section (current recent-window divergence)
  - Tail-event tables include divergence context and RSI regime tags
  - A compact summary table comparing tail outcomes **with** vs **without** divergence (daily)

---

## 5) CLI usage

Example:
```bash
./.venv/bin/options-helper technicals extension-stats --symbol CVX --cache-dir data/candles --out data/reports/technicals/extension --print
```

Key knobs:
- `--divergence-window-days` (default 14; interpreted as bars for each timeframe)
- `--divergence-min-extension-days` (default 5)
- `--divergence-min-extension-percentile` / `--divergence-max-extension-percentile` (defaults to tail thresholds)
- `--divergence-min-price-delta-pct` / `--divergence-min-rsi-delta`
- `--rsi-overbought` / `--rsi-oversold`
- `--require-rsi-extreme`

---

## 6) Caveats

- Divergences are **descriptive patterns**, not predictions.
- Swing-point logic is intentionally simple; choppy markets can produce many small swings.
- Percentiles depend on the configured rolling window and the candle history quality.
