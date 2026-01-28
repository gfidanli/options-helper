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

## Snapshot scope (window around spot)
To keep snapshots small and focused, the tool saves a **strike window around spot**:

- spot is estimated from cached daily candles (last close)
- strikes saved: `spot * (1 - window_pct)` to `spot * (1 + window_pct)`

Default:
- `window_pct = 0.30` (±30% around spot)

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

## Flow report (day-to-day)
Once you have at least two snapshots for a symbol, you can view day-to-day deltas:

```bash
options-helper flow portfolio.json
```

The report computes per-contract:
- `ΔOI = OI_today - OI_prev`
- `ΔOI_notional ≈ ΔOI * lastPrice * 100`
- `volume_notional ≈ volume * lastPrice * 100`

And classifies activity (heuristic):
- **building**: ΔOI significantly positive
- **unwinding**: ΔOI significantly negative
- **churn**: high volume but small ΔOI
- **unknown**: insufficient prior snapshot coverage

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
