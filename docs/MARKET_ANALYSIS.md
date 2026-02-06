# Market Analysis (Tail Risk + IV Context)

Status: Implemented

This feature adds a Monte Carlo tail-risk view inspired by TeamCinco-style market analysis, implemented natively in this repo.

It is informational/educational tooling and **not financial advice**.

## What it does

- Simulates horizon end outcomes from historical daily log returns (GBM-style Monte Carlo).
- Reports risk context:
  - annualized realized volatility,
  - annualized expected return (sample mean),
  - VaR / CVaR at configurable confidence.
- Shows end-horizon percentile table (price and return).
- Adds IV context from `derived_daily`:
  - IV/RV20 and a regime label: `cheap`, `fair`, `expensive`,
  - ATM IV, RV20/RV60, IV percentile, and IV term slope.

## CLI

Command:

```bash
./.venv/bin/options-helper market-analysis tail-risk --symbol SPY
```

Key options:

- `--lookback-days` (default `1512`)
- `--horizon-days` (default `60`)
- `--num-simulations` (default `25000`)
- `--seed` (default `42`)
- `--var-confidence` (default `0.95`)
- `--refresh/--no-refresh` (default `--no-refresh`)
- `--format console|json` (default `console`)
- `--out <path>` to write artifacts under `{out}/tail_risk/{SYMBOL}/`

Offline-first behavior:

- `--no-refresh` reads local cache/store only.
- `--refresh` updates candles first via provider fetch.

## Streamlit page

The portal page `07_Market_Analysis.py` is a read-only DuckDB-first surface:

- symbol selector from `candles_daily`,
- configurable Monte Carlo controls,
- fan chart and percentile table,
- move percentile calculator,
- IV context panel from `derived_daily`.

If `derived_daily` is missing for a symbol, run:

```bash
./.venv/bin/options-helper snapshot-options --symbol SPY
./.venv/bin/options-helper derived update --symbol SPY
```

## Data assumptions and limitations

- Uses historical daily close returns with normally distributed shocks.
- VaR/CVaR and expected return are model outputs, not predictions.
- Output quality depends on local data quality and coverage in candles/snapshots/derived rows.

