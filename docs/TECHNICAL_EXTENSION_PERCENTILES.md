# Extension Percentiles — Technicals + Drift Tables

This feature adds **per‑ticker extension percentiles** (from `technicals_backtesting`) and
**tail‑event follow‑through** to provide context on whether price is stretched relative to its own history.

This project is **not financial advice**. These outputs are descriptive and intended to support
research/decisioning workflows.

---

## 1) What is `extension_atr`?

`extension_atr = (Close − SMA_N) / ATR_W`

This expresses how far price is from its moving average in **ATR units**. It normalizes for volatility:
an extension of +2.0 means price is ~2 ATRs above the moving average.

## 2) Why percentiles?

Raw ATR units are not always comparable across tickers or regimes. Percentiles answer:
“Is this extension high vs this ticker’s own history?”

Examples:
- 95th percentile → unusually extended (relative to that ticker’s history)
- 5th percentile → unusually depressed

## 3) Rolling windows (1y / 3y / 5y)

Percentiles are computed over rolling windows (trading‑day offsets, default 252 days/year):
- 3y: default window
- If < 3y of data, the percentile window falls back to **all available history**.

The report outputs **p5 / p50 / p95** for each window so you can see if distributions drift.

> Drift flags are **not** automated in v1. We show the table so you can decide which window is most appropriate.

## 4) Tail‑event follow‑through

We identify all days where extension percentile is in the **upper or lower tails** (configurable),
then compute follow-through outcomes at forward offsets:
- Daily: +1/+3/+5/+10/+15 trading days (by default)
- Weekly: +1/+3/+5/+10 weeks (by default)

- **Primary:** extension percentile (mean‑reversion in normalized terms)
- **Secondary:** max favorable move (directional, MFE-style)
  - low tail: bounce using High (max up move)
  - high tail: pullback using Low (max down move)

This helps answer questions like:
“When extension hits the 95th percentile, does it mean‑revert quickly?”

## 5) Where this appears

- **Briefing (Markdown + JSON):** current extension percentiles + rolling quantiles (daily + weekly).
- **Extension stats report (`technicals extension-stats`):**
  - Daily tail‑event table
    - includes weekly context columns (weekly extension percentile, weekly RSI tag, weekly divergence)
    - max ret% is shown for longer horizons (1w/4w/3m/6m/9m/1y)
  - Daily “Max Favorable Move” summary (1w/4w/3m/6m/9m/1y; includes both low-tail and high-tail buckets)
  - Weekly tail‑event table
  - Rolling quantile tables (daily + weekly)

Example:
```
./.venv/bin/options-helper technicals extension-stats --symbol CVX --cache-dir data/candles --out data/reports/technicals/extension --print
```

## 6) Configuration

See `config/technical_backtesting.yaml` → `extension_percentiles`:

- `days_per_year`: trading days/year (default 252)
- `windows_years`: rolling windows
- `tail_high_pct` / `tail_low_pct`: tail thresholds
- `forward_days`: forward windows (base list; used for weekly unless overridden)
- `forward_days_daily`: optional override for daily forward windows
- `forward_days_weekly`: optional override for weekly forward windows

---

## 7) Caveats

- Percentiles depend on the **window**; short windows adapt faster but are noisier.
- Tail outcomes are descriptive, not predictive.
- Gaps or regime shifts can distort percentile behavior.
