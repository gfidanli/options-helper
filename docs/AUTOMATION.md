# Automation (cron) — Recurring Jobs, Order, and Dependencies

This repo includes cron-friendly scripts under `scripts/` to keep data caches and reports up to date.

This tool is for informational/educational use only and is not financial advice.

## Key ideas

- **Data first, reports second:** `briefing` (and most offline reports) depend on snapshot files under `data/`.
- **Idempotent by design:** re-running jobs should update/overwrite the latest day rather than creating duplicates.
- **Non-overlap:** cron jobs use a shared lock directory under `data/locks/` to avoid concurrent writes to caches.

## Current recurring jobs (defaults)

All times are **local time**.

### 1) Weekly earnings cache refresh (network)

- **When:** Mondays at **17:50**
- **Script:** `scripts/cron_weekly_refresh_earnings.sh`
- **Installs via:** `scripts/install_cron_weekly_refresh_earnings.sh`
- **What it does:**
  - `options-helper watchlists sync-positions portfolio.json` (keeps the `positions` watchlist current)
  - `options-helper refresh-earnings` (writes `data/earnings/{SYMBOL}.json`)
- **Depends on:**
  - `data/watchlists.json` existing and valid JSON
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/earnings_refresh.log`

### 2) Daily portfolio candles + options snapshots (network)

- **When:** Weekdays at **18:15**
- **Script:** `scripts/cron_daily_options_snapshot.sh`
- **Installs via:** `scripts/install_cron_daily_options_snapshot.sh`
- **What it does:**
  - `options-helper refresh-candles portfolio.json` (portfolio + watchlists)
  - `options-helper snapshot-options portfolio.json` (portfolio symbols, position expiries)
- **Writes:**
  - candles: `data/candles/{SYMBOL}.csv`
  - snapshots: `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv` + `meta.json`
- **Depends on:**
  - `.venv/bin/options-helper` installed
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/options_snapshot.log`

### 3) Daily watchlist (“monitor”) option snapshots (network, optional)

If you maintain a watchlist named `monitor` in `data/watchlists.json`, this job snapshots those symbols so offline
flow/chain reports work for them too.

- **When:** Weekdays at **18:35**
- **Script:** `scripts/cron_daily_monitor_options_snapshot.sh`
- **Installs via:** `scripts/install_cron_daily_monitor_options_snapshot.sh`
- **What it does:** `options-helper snapshot-options portfolio.json --watchlist monitor --max-expiries 2`
- **Depends on:**
  - `data/watchlists.json` existing and containing a non-empty `monitor` list (otherwise it skips)
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/monitor_snapshot.log`

### 4) Daily briefing report (offline-first, depends on snapshots)

- **When:** Weekdays at **19:00**
- **Script:** `scripts/cron_daily_briefing.sh`
- **Installs via:** `scripts/install_cron_daily_briefing.sh`
- **What it does:**
  - `options-helper briefing portfolio.json --as-of latest --compare -1`
  - includes `--watchlist monitor` if `data/watchlists.json` exists
  - updates derived history (`data/derived/`) by default
- **Writes:**
  - daily report: `data/reports/daily/{YYYY-MM-DD}.md`
  - derived history (per symbol): `data/derived/{SYMBOL}.csv`
- **Depends on:**
  - snapshot folders existing for the included symbols (it will emit per-symbol warnings when missing)
- **Logs:** `data/logs/briefing.log`

### 5) Daily full scanner + completion checks (network, heavy)

- **When:** Daily at **17:00 CST**, with checks at **19:00/20:00/21:00 CST**
- **Script:** `scripts/cron_daily_scanner_full.sh`
- **Checks:** `scripts/cron_check_scanner_full.sh`
- **Installs via:** `scripts/install_cron_daily_scanner_full.sh`
- **What it does:**
  - full `options-helper scanner run` over the SEC universe (backfill + options snapshots + liquidity + reports)
  - writes a status marker so the hourly checks only retry if the run didn’t finish
- **Depends on:**
  - `.venv/bin/options-helper` installed
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/scanner_full_YYYY-MM-DD.log`
- **Status:** `data/logs/scanner_full_status.json`

## Dependency order (recommended)

- **Daily:** portfolio snapshot (18:15) → monitor snapshot (18:35) → briefing (19:00)
- **Weekly:** earnings refresh is independent; schedule it whenever (it’s just a cache).
- **Scanner:** full scanner run is independent but heavy; consider running after markets close.

## Installation / management

View current cron entries:

```bash
crontab -l
```

Install/update a job (each installer is idempotent; re-running updates the tagged block):

```bash
./scripts/install_cron_daily_options_snapshot.sh --install
./scripts/install_cron_daily_monitor_options_snapshot.sh --install
./scripts/install_cron_daily_briefing.sh --install
./scripts/install_cron_weekly_refresh_earnings.sh --install
./scripts/install_cron_daily_scanner_full.sh --install
```

## Operational notes (macOS cron)

- `cron` won’t run while the machine is asleep.
- Jobs referencing `/Volumes/...` will fail if the volume isn’t mounted.
- If a job appears “stuck”, check the shared lock dir:
  - `data/locks/options_helper_cron.lock`
  - if the machine crashed mid-run, you may need to remove it manually.
