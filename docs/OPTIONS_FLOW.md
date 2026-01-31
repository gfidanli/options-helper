# Options Flow (OI/Volume Deltas) — Design & Usage

## What this is (and isn’t)
This feature builds a **best-effort “positioning” proxy** from:

- **Open interest (OI) day-to-day changes** (ΔOI)
- **Daily volume**

Because Yahoo Finance (via `yfinance`) does **not** provide historical OI/volume, the tool must **collect its own daily
history** by saving daily snapshots of the options chain and comparing snapshots.

Important limitations:

- OI usually updates **once per day** and can lag.
- This is not a true “smart money” feed — it’s a **heuristic signal**.
- Some chains are illiquid; values can be stale/incorrect.

## Data storage layout
Snapshots are stored under a root directory (default `data/options_snapshots`):

- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv`

Each `{EXPIRY}.csv` file includes both calls and puts in one table with an `optionType` column.

**Important:** `{YYYY-MM-DD}` is the **data date** (latest available daily candle date used to estimate spot), not the
wall-clock time you ran the snapshot. This avoids pre-market runs being labeled as “today” when the latest daily candle
is still “yesterday’s close”.

## Greeks (best-effort Black–Scholes)
Snapshots also include **best-effort Black–Scholes Greeks** (computed locally, not sourced from Yahoo):

- `bs_price`
- `bs_delta`
- `bs_gamma`
- `bs_theta_per_day`
- `bs_vega`

Inputs:
- spot from the daily candle close (or a fallback quote)
- `impliedVolatility` from Yahoo
- time to expiry (using the snapshot data date as-of)
- optional `--risk-free-rate` (defaults to `0.0`)

These are approximations and should be treated as a model-based estimate.

## Snapshot scope (window around spot)
To keep snapshots small and focused, the tool saves a **strike window around spot**:

- spot is estimated from cached daily candles (last close)
- strikes saved: `spot * (1 - window_pct)` to `spot * (1 + window_pct)`

Default:
- `window_pct = 1.0` (±100% around spot)

You can override:

```bash
options-helper snapshot-options portfolio.json --window-pct 0.25
```

## Daily snapshot collection (recommended cadence)
Recommended: **once per trading day after market close** (or later in the evening).

CLI:

```bash
options-helper snapshot-options portfolio.json
```

This snapshots:
- only symbols in your `portfolio.json`
- only expirations in your positions for those symbols
- calls + puts for those expiries, filtered by the strike window

### Snapshotting watchlists (optional)
You can also snapshot symbols from your watchlists store:

```bash
options-helper snapshot-options portfolio.json --watchlists-path data/watchlists.json --all-watchlists
```

Or one or more named watchlists (repeatable):

```bash
options-helper snapshot-options portfolio.json --watchlists-path data/watchlists.json --watchlist monitor --watchlist positions
```

By default, watchlist snapshots are capped to the **nearest 2 expiries** per symbol to keep runtime and storage reasonable.
Use `--all-expiries` or `--full-chain` to override.

### Full-chain snapshots (all expiries + raw Yahoo payload)
If you want to snapshot **everything Yahoo returns for the options chain** (extra fields beyond yfinance’s fixed-column
DataFrame) and **every listed expiry**, use `--full-chain`:

```bash
options-helper snapshot-options portfolio.json --watchlists-path data/watchlists.json --all-watchlists --full-chain
```

In `--full-chain` mode the tool writes, per symbol/day/expiry:
- `{EXPIRY}.csv` — calls + puts with **all fields** returned by Yahoo
- `{EXPIRY}.raw.json` — the **raw Yahoo payload** for that expiry

Full-chain snapshots can be large; consider using a separate cache root via `--cache-dir`.

## Flow report (day-to-day)
Once you have at least two snapshots for a symbol, you can view day-to-day deltas:

```bash
options-helper flow portfolio.json
```

You can also report flow for watchlists:

```bash
options-helper flow portfolio.json --watchlists-path data/watchlists.json --all-watchlists
```

The report computes per-contract:
- `ΔOI = OI_today - OI_prev`
- `mark` (best-effort mid/last/ask/bid)
- `premium_notional ≈ ΔOI * mark * 100`
- `volume_notional ≈ volume * mark * 100`
- `delta_notional ≈ ΔOI * bs_delta * spot * 100` (best-effort; spot from `meta.json`)

And classifies activity (heuristic):
- **building**: ΔOI significantly positive
- **unwinding**: ΔOI significantly negative
- **churn**: high volume but small ΔOI
- **unknown**: insufficient prior snapshot coverage

## Flow report (windowed + aggregated “zones”)
Contract-level lists are useful but noisy. You can net multiple snapshot deltas and aggregate by strike/expiry:

```bash
options-helper flow portfolio.json --symbol CVX --window 5 --group-by strike
options-helper flow portfolio.json --symbol CVX --window 5 --group-by expiry
options-helper flow portfolio.json --symbol CVX --window 5 --group-by expiry-strike --top 20
```

Notes:
- `--window N` nets the last **N snapshot-to-snapshot deltas** (requires **N+1 snapshots** for the symbol).
- `--group-by` controls aggregation grain:
  - `strike`: `(optionType, strike)`
  - `expiry`: `(optionType, expiry)`
  - `expiry-strike`: `(optionType, expiry, strike)`

### Saving a JSON artifact
You can persist the aggregated net flow output as JSON:

```bash
options-helper flow portfolio.json --symbol CVX --window 5 --group-by expiry-strike --out data/reports
```

Writes under:
- `data/reports/flow/{SYMBOL}/{FROM}_to_{TO}_w{N}_{group_by}.json`

## Automation (cron)
This repo includes scripts to run the snapshot once per weekday:

- `scripts/cron_daily_options_snapshot.sh`
- `scripts/install_cron_daily_options_snapshot.sh`

Install (adds/updates a tagged entry in your crontab):

```bash
./scripts/install_cron_daily_options_snapshot.sh
```

Defaults:
- Runs at **18:15 local time**, Monday–Friday.
- Logs to `data/logs/options_snapshot.log`.

To automatically install (attempts `crontab` update with a timeout):

```bash
./scripts/install_cron_daily_options_snapshot.sh --install
```

You can edit the schedule in `scripts/install_cron_daily_options_snapshot.sh`.
