# `derived` — Derived Metrics Store (Per-Symbol History)

`derived` persists a compact per-symbol, per-day time series of derived metrics computed from local snapshots.

This enables trend/percentile questions later without re-parsing full chains every time.

This tool is for informational/educational use only and is not financial advice.

## Files

Per symbol:
- `data/derived/{SYMBOL}.csv`

Schema v2 columns:
- `date,spot,pc_oi,pc_vol,call_wall,put_wall,gamma_peak_strike,atm_iv_near,em_near_pct,skew_near_pp,rv_20d,rv_60d,iv_rv_20d,atm_iv_near_percentile,iv_term_slope`

Notes:
- `atm_iv_near` and `em_near_pct` are stored as ratios (e.g., `0.25` for 25%).
- `skew_near_pp` is stored in percentage-points (pp).
- `rv_20d`/`rv_60d` are annualized realized vol (daily log returns, `sqrt(252)`).
- `iv_rv_20d` is a ratio of near ATM IV to 20D RV (null if RV unavailable).
- `atm_iv_near_percentile` is the percentile rank of the near ATM IV vs stored history.
- `iv_term_slope` is `atm_iv_next - atm_iv_near` (absolute IV), null if next expiry missing.

## Commands

Update (idempotent per day):

```bash
options-helper derived update --symbol CVX --as-of 2026-01-30
options-helper derived update --symbol CVX --as-of latest
```

Show last N rows:

```bash
options-helper derived show --symbol CVX --last 30
```

Stats (percentiles + trend flags):

```bash
options-helper derived stats --symbol CVX --as-of latest --window 60 --trend-window 5
options-helper derived stats --symbol CVX --as-of latest --window 60 --trend-window 5 --format json
```

## CLI flags

### `derived update`
- `--symbol TICKER`: symbol to update (required)
- `--as-of YYYY-MM-DD|latest`: snapshot date (default `latest`)
- `--cache-dir PATH`: snapshots root (default `data/options_snapshots`)
- `--derived-dir PATH`: derived store directory (default `data/derived`)
- `--candle-cache-dir PATH`: candle cache root (default `data/candles`)

### `derived show`
- `--symbol TICKER`: symbol to show (required)
- `--derived-dir PATH`: derived store directory (default `data/derived`)
- `--last N`: show last N rows (default `30`)

### `derived stats`
- `--symbol TICKER`: symbol to analyze (required)
- `--as-of YYYY-MM-DD|latest`: derived date to evaluate (default `latest`)
- `--derived-dir PATH`: derived store directory (default `data/derived`)
- `--window N`: percentile lookback window (default `60`)
- `--trend-window N`: trend lookback window (default `5`)
- `--format console|json`: output format (default `console`)
- `--out PATH`: output root for saved artifacts (writes under `{out}/derived/{SYMBOL}/`)

## Caveats (best-effort)

- Derived rows depend on what’s present in the snapshot window; missing strikes/greeks can lead to null fields.
- Re-running `derived update` for the same `{SYMBOL, date}` overwrites that day’s row (no duplicates).
- `derived stats` percentiles and trends are computed from the stored series; missing values reduce the effective sample size.
