# `derived` — Derived Metrics Store (Per-Symbol History)

`derived` persists a compact per-symbol, per-day time series of derived metrics computed from local snapshots.

This enables trend/percentile questions later without re-parsing full chains every time.

This tool is for informational/educational use only and is not financial advice.

## Files

Per symbol:
- `data/derived/{SYMBOL}.csv`

Schema v1 columns:
- `date,spot,pc_oi,pc_vol,call_wall,put_wall,gamma_peak_strike,atm_iv_near,em_near_pct,skew_near_pp`

Notes:
- `atm_iv_near` and `em_near_pct` are stored as ratios (e.g., `0.25` for 25%).
- `skew_near_pp` is stored in percentage-points (pp).

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

## CLI flags

### `derived update`
- `--symbol TICKER`: symbol to update (required)
- `--as-of YYYY-MM-DD|latest`: snapshot date (default `latest`)
- `--cache-dir PATH`: snapshots root (default `data/options_snapshots`)
- `--derived-dir PATH`: derived store directory (default `data/derived`)

### `derived show`
- `--symbol TICKER`: symbol to show (required)
- `--derived-dir PATH`: derived store directory (default `data/derived`)
- `--last N`: show last N rows (default `30`)

## Caveats (best-effort)

- Derived rows depend on what’s present in the snapshot window; missing strikes/greeks can lead to null fields.
- Re-running `derived update` for the same `{SYMBOL, date}` overwrites that day’s row (no duplicates).

