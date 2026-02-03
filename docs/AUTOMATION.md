# Automation (cron) — Recurring Jobs, Order, and Dependencies

This repo includes cron-friendly scripts under `scripts/` to keep data caches and reports up to date.

This tool is for informational/educational use only and is not financial advice.

## Key ideas

- **Data first, reports second:** `briefing` (and most offline reports) depend on snapshot files under `data/`.
- **Idempotent by design:** re-running jobs should update/overwrite the latest day rather than creating duplicates.
- **Non-overlap:** cron jobs use a shared lock directory under `data/locks/` to avoid concurrent writes to caches.

## Current recurring jobs (defaults)

All times below are **America/Chicago** time (the installers write `CRON_TZ=America/Chicago`).

### 1) Weekly earnings cache refresh (network)

- **When:** Mondays at **15:20**
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

- **When:** Weekdays at **16:00** (market close + 60m)
- **Script:** `scripts/cron_daily_options_snapshot.sh`
- **Installs via:** `scripts/install_cron_daily_options_snapshot.sh`
- **What it does:**
  - `options-helper refresh-candles portfolio.json` (portfolio + watchlists)
  - `options-helper snapshot-options portfolio.json` (portfolio symbols, position expiries)
  - waits for the current day's daily candle to be published (canary check) so snapshots don't get written under the prior date
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

- **When:** Weekdays at **17:30**
- **Script:** `scripts/cron_daily_monitor_options_snapshot.sh`
- **Installs via:** `scripts/install_cron_daily_monitor_options_snapshot.sh`
- **What it does:** `options-helper snapshot-options portfolio.json --watchlist monitor --watchlist positions --max-expiries 2`
- **Depends on:**
  - `data/watchlists.json` existing and containing a non-empty `monitor` and/or `positions` list (otherwise it skips)
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/monitor_snapshot.log`

### 4) Daily briefing report (offline-first, depends on snapshots)

- **When:** Weekdays at **18:00**
- **Script:** `scripts/cron_daily_briefing.sh`
- **Installs via:** `scripts/install_cron_daily_briefing.sh`
- **What it does:**
  - `options-helper briefing portfolio.json --as-of latest --compare -1`
  - includes `--watchlist positions` + `--watchlist monitor` if `data/watchlists.json` exists
  - updates derived history (`data/derived/`) by default
- **Writes:**
  - daily report: `data/reports/daily/{YYYY-MM-DD}.md`
  - derived history (per symbol): `data/derived/{SYMBOL}.csv`
- **Depends on:**
  - snapshot folders existing for the included symbols (it will emit per-symbol warnings when missing)
- **Logs:** `data/logs/briefing.log`

### 5) Daily full scanner + completion checks (network, heavy)

- **When:** Weekdays at **19:00 CST**, with checks at **20:00/21:00 CST**
- **Script:** `scripts/cron_daily_scanner_full.sh`
- **Checks:** `scripts/cron_check_scanner_full.sh`
- **Installs via:** `scripts/install_cron_daily_scanner_full.sh`
- **What it does:**
  - full `options-helper scanner run` over the SEC universe (backfill + options snapshots + liquidity + reports)
  - skips if the current day's daily candle isn't published yet (to avoid mis-dated overwrites)
  - writes a status marker so the hourly checks only retry if the run didn’t finish
- **Depends on:**
  - `.venv/bin/options-helper` installed
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/scanner_full_YYYY-MM-DD.log`
- **Status:** `data/logs/scanner_full_status.json`

### 6) Daily offline report pack (offline-first, depends on snapshots/candles)

Generates per-symbol saved artifacts (JSON/Markdown) for offline review.

- **When:** Weekdays at **21:45 CST**
- **Script:** `scripts/cron_offline_report_pack.sh`
- **Installs via:** `scripts/install_cron_offline_report_pack.sh`
- **What it does:**
  - Runs `options-helper report-pack portfolio.json` over watchlists:
    - `positions`
    - `monitor`
    - `Scanner - Shortlist` (only included when today's scanner run succeeded)
- **Writes:**
  - chain dashboards: `data/reports/chains/{SYMBOL}/{YYYY-MM-DD}.json` + `.md`
  - snapshot diffs: `data/reports/compare/{SYMBOL}/{FROM}_to_{TO}.json`
  - flow deltas: `data/reports/flow/{SYMBOL}/{FROM}_to_{TO}_w1_*.json`
  - derived stats: `data/reports/derived/{SYMBOL}/{ASOF}_w{N}_tw{M}.json`
  - technicals extension-stats: `data/reports/technicals/extension/{SYMBOL}/{YYYY-MM-DD}.json` + `.md`
- **Depends on:**
  - snapshots and candle cache being current for the day
- **Logs:** `data/logs/report_pack.log`

## Dependency order (recommended)

- **Daily:** portfolio snapshot (16:00) → monitor snapshot (17:30) → briefing (18:00) → scanner (19:00) → report pack (21:45)
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
./scripts/install_cron_offline_report_pack.sh --install
```

## Operational notes (macOS cron)

- `cron` won’t run while the machine is asleep.
- Jobs referencing `/Volumes/...` will fail if the volume isn’t mounted.
- If a job appears “stuck”, check the shared lock dir:
  - `data/locks/options_helper_cron.lock`
  - if the machine crashed mid-run, you may need to remove it manually.
