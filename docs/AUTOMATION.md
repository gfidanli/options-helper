# Automation (cron) — Recurring Jobs, Order, and Dependencies

This repo includes cron-friendly scripts under `scripts/` to keep data caches and reports up to date.

This tool is for informational/educational use only and is not financial advice.

## Key ideas

- **Data first, reports second:** `briefing` (and most offline reports) depend on snapshot files under `data/`.
- **Idempotent by design:** re-running jobs should update/overwrite the latest day rather than creating duplicates.
- **Non-overlap:** cron jobs use a shared lock directory under `data/locks/` to avoid concurrent writes to caches.
- **Logs:** every CLI command accepts `--log-dir` (default `data/logs/`) and writes a timestamped log file under
  `data/logs/{YYYY-MM-DD}/` (America/Chicago). Use `--log-path` to write to a specific file (cron uses this to keep one
  log file per job/day).
- **Provider:** cron scripts default to Alpaca (`PROVIDER=alpaca`). Override per-run with `PROVIDER=yahoo` if needed.
- **Adjusted candles:** daily candle caches are fetched using split/dividend-adjusted prices by default (for indicator/backtest continuity).

## Current recurring jobs (defaults)

All times below are **America/Chicago** time (the installers write `CRON_TZ=America/Chicago`).

### Schedule table (source-of-truth = installer scripts)

This table is intended to be the “at-a-glance” reference. If you change any cron schedules, update the
corresponding `scripts/install_cron_*.sh` first, then keep this table in sync (there’s a test that enforces this).

| Job | Installer | Cron schedule (CRON_TZ=America/Chicago) | Script(s) | Primary outputs | Logs / status |
|---|---|---|---|---|---|
| Weekly earnings refresh | `scripts/install_cron_weekly_refresh_earnings.sh` | `0 12 * * 1` | `scripts/cron_weekly_refresh_earnings.sh` | `data/earnings/{SYMBOL}.json` | `data/logs/{YYYY-MM-DD}/earnings_refresh.log` |
| Daily options snapshot | `scripts/install_cron_daily_options_snapshot.sh` | `0 16 * * 1-5` | `scripts/cron_daily_options_snapshot.sh` | `data/candles/{SYMBOL}.csv`<br>`data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv` | `data/logs/{YYYY-MM-DD}/options_snapshot.log` |
| Daily monitor snapshot | `scripts/install_cron_daily_monitor_options_snapshot.sh` | `5 16 * * 1-5` | `scripts/cron_daily_monitor_options_snapshot.sh` | `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv` | `data/logs/{YYYY-MM-DD}/monitor_snapshot.log` |
| Daily briefing | `scripts/install_cron_daily_briefing.sh` | `10 16 * * 1-5` | `scripts/cron_daily_briefing.sh` | `data/reports/daily/{YYYY-MM-DD}.md`<br>`data/derived/{SYMBOL}.csv` | `data/logs/{YYYY-MM-DD}/briefing.log` |
| Daily full scanner (run + check) | `scripts/install_cron_daily_scanner_full.sh` | run: `15 16 * * 1-5`<br>check: `30 16 * * 1-5` | `scripts/cron_daily_scanner_full.sh`<br>`scripts/cron_check_scanner_full.sh` | `data/scanner/runs/...` | `data/logs/{YYYY-MM-DD}/scanner_full.log`<br>`data/logs/{YYYY-MM-DD}/scanner_full_status.json` |
| Daily offline report pack | `scripts/install_cron_offline_report_pack.sh` | `0 17 * * 1-5` | `scripts/cron_offline_report_pack.sh` | `data/reports/**` | `data/logs/{YYYY-MM-DD}/report_pack.log` |
| Intraday capture (optional) | `scripts/install_cron_intraday_capture.sh` | `5 15 * * 1-5` | `scripts/cron_intraday_capture.sh` | `data/intraday/**` | `data/logs/{YYYY-MM-DD}/intraday_capture.log` |
| Stream capture session (optional) | (no installer) | (on-demand) | `scripts/cron_stream_capture.sh` | `data/intraday/**` | `data/logs/{YYYY-MM-DD}/stream_capture.log` |

### 1) Weekly earnings cache refresh (network)

- **When:** Mondays at **12:00**
- **Script:** `scripts/cron_weekly_refresh_earnings.sh`
- **Installs via:** `scripts/install_cron_weekly_refresh_earnings.sh`
- **What it does:**
  - `options-helper watchlists sync-positions portfolio.json` (keeps the `positions` watchlist current)
  - `options-helper refresh-earnings` (writes `data/earnings/{SYMBOL}.json`)
- **Depends on:**
  - `data/watchlists.json` existing and valid JSON
  - network access (Yahoo via `yfinance`)
- **Logs:** `data/logs/{YYYY-MM-DD}/earnings_refresh.log`

### 2) Daily portfolio candles + options snapshots (network)

- **When:** Weekdays at **16:00** (market close + 60m)
- **Script:** `scripts/cron_daily_options_snapshot.sh`
- **Installs via:** `scripts/install_cron_daily_options_snapshot.sh`
- **What it does:**
  - `options-helper refresh-candles portfolio.json` (portfolio + watchlists)
  - `options-helper snapshot-options portfolio.json --windowed --position-expiries` (portfolio symbols, position expiries; flow-focused)
  - waits for the current day's daily candle to be published (canary check) so snapshots don't get written under the prior date
- **Writes:**
  - candles: `data/candles/{SYMBOL}.csv`
  - snapshots: `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv` + `meta.json`
- **Depends on:**
  - `.venv/bin/options-helper` installed
  - network access (Alpaca via `alpaca-py`)
- **Logs:** `data/logs/{YYYY-MM-DD}/options_snapshot.log`

### 3) Daily watchlist (“monitor”) option snapshots (network, optional)

If you maintain a watchlist named `monitor` in `data/watchlists.json`, this job snapshots those symbols so offline
flow/chain reports work for them too.

- **When:** Weekdays at **16:05**
- **Script:** `scripts/cron_daily_monitor_options_snapshot.sh`
- **Installs via:** `scripts/install_cron_daily_monitor_options_snapshot.sh`
- **What it does:** `options-helper snapshot-options portfolio.json --watchlist monitor --watchlist positions --max-expiries 2 --windowed --position-expiries`
- **Depends on:**
  - `data/watchlists.json` existing and containing a non-empty `monitor` and/or `positions` list (otherwise it skips)
  - network access (Alpaca via `alpaca-py`)
- **Logs:** `data/logs/{YYYY-MM-DD}/monitor_snapshot.log`

### 4) Daily briefing report (offline-first, depends on snapshots)

- **When:** Weekdays at **16:10**
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
- **Logs:** `data/logs/{YYYY-MM-DD}/briefing.log`

### 5) Daily full scanner + completion checks (network, heavy)

- **When:** Weekdays at **16:15 CST**, with a check at **16:30 CST**
- **Script:** `scripts/cron_daily_scanner_full.sh`
- **Checks:** `scripts/cron_check_scanner_full.sh`
- **Installs via:** `scripts/install_cron_daily_scanner_full.sh`
- **What it does:**
  - full `options-helper scanner run` over the SEC universe (backfill + options snapshots + liquidity + reports)
  - skips if the current day's daily candle isn't published yet (to avoid mis-dated overwrites)
  - writes a status marker so the check job only retries if the run didn’t finish
- **Depends on:**
  - `.venv/bin/options-helper` installed
  - network access (Alpaca via `alpaca-py`)
- **Logs:** `data/logs/{YYYY-MM-DD}/scanner_full.log`
- **Status:** `data/logs/{YYYY-MM-DD}/scanner_full_status.json`

### 6) Daily offline report pack (offline-first, depends on snapshots/candles)

Generates per-symbol saved artifacts (JSON/Markdown) for offline review.

- **When:** Weekdays at **17:00 CST**
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
- **Logs:** `data/logs/{YYYY-MM-DD}/report_pack.log`

## Other optional jobs

These exist in `scripts/` but are not part of the default “daily stack” above.

- **Intraday capture (optional):** `scripts/cron_intraday_capture.sh` (installer: `scripts/install_cron_intraday_capture.sh`, default **15:05**)
- **Stream capture session (optional):** `scripts/cron_stream_capture.sh` (no installer; run on-demand or add your own cron entry)

## Dependency order (recommended)

- **Daily:** portfolio snapshot (16:00) → monitor snapshot (16:05) → briefing (16:10) → scanner (16:15) → check (16:30) → report pack (17:00)
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
