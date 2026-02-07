# Scanner (Market Opportunity Watchlists)

The scanner builds two watchlists from a broad US universe and writes a run folder for review.
This is **decision support only** and **not financial advice**.

## What it does
- **Universe:** US equities + ETFs (best-effort, cached daily).
- **Scan:** compute current extension percentile (ATR extension) for every symbol.
- **Tail watchlist:** if percentile <= 2.5 or >= 97.5 → add to **Scanner - All** (replaced each run).
- **Backfill:** ensure max candle history for tail symbols.
- **Options snapshot:** full chain + all expiries for tail symbols.
- **Liquidity filter:** DTE >= 60, `volume >= 10`, `openInterest >= 500` → **Scanner - Shortlist** (replaced each run).
- **Scanner rank:** shortlist is ordered by scanner score (multi-factor, best-effort).
- **Confluence score:** still computed for context in the shortlist markdown.
- **Reports:** Extension Percentile Stats reports for shortlist symbols.

## Workflow diagram

![Scanner pipeline](assets/diagrams/generated/scanner_pipeline.svg)

## Command
```bash
./.venv/bin/options-helper scanner run
```

Common flags:
```bash
./.venv/bin/options-helper scanner run \
  --universe file:data/universe/sec_company_tickers.json \
  --prefilter-mode default \
  --exclude-path data/universe/exclude_symbols.txt \
  --scanned-path data/scanner/scanned_symbols.txt \
  --tail-pct 2.5 \
  --derived-dir data/derived \
  --run-id 2026-02-01 \
  --run-dir data/scanner/runs
```

## Outputs
- Run artifacts:
  - `data/scanner/runs/<run_id>/scan.csv`
  - `data/scanner/runs/<run_id>/liquidity.csv`
  - `data/scanner/runs/<run_id>/shortlist.csv`
  - `data/scanner/runs/<run_id>/shortlist.md`
    - includes scanner score + coverage and confluence score + coverage per symbol
- Watchlists:
  - `data/watchlists.json` → **Scanner - All** and **Scanner - Shortlist**
- Options snapshots (full chain, all expiries):
  - `data/options_snapshots/<SYMBOL>/<YYYY-MM-DD>/`
- Extension Percentile Stats reports:
  - `data/reports/technicals/extension/<SYMBOL>/`

## Notes & assumptions
- v1 uses a **local SEC ticker file** at `data/universe/sec_company_tickers.json` (no refresh). A future version can fetch this file periodically to stay up to date.
- Nasdaq Trader symbol directory is still supported and required for the `us-etfs` universe (SEC list has no ETF flag).
- `yfinance` is best-effort: missing fields, stale quotes, or absent option chains can occur.
- Snapshot folders use the **latest candle date**, not the wall-clock run time.
- Scanner ranking uses derived IV/RV when available (`data/derived/{SYMBOL}.csv`); if missing, coverage is reduced.
- This tool is **not** financial advice.

## Performance & API etiquette
- The scan runs in **batches** with limited parallelism and a small pause between batches.
- Tune with `--workers`, `--batch-size`, and `--batch-sleep-seconds` to respect API limits.
- `--prefilter-mode` removes obvious non-standard tickers (warrants/units/rights); use `aggressive` to drop more.
- Symbols that error during the scan are appended to `data/universe/exclude_symbols.txt` (disable with `--no-write-error-excludes`).
- Use `--exclude-statuses` to control which scan outcomes are excluded (default: `error,no_candles`).
- Scanned symbols are persisted to `data/scanner/scanned_symbols.txt` and skipped on future runs (disable with `--no-skip-scanned` or `--no-write-scanned`).
- Shortlist CSV writing can be disabled with `--no-write-shortlist`.

## Automation (cron)
- Full daily scanner (CST) + completion checks:
  - `scripts/cron_daily_scanner_full.sh`
  - `scripts/cron_check_scanner_full.sh`
  - Install: `scripts/install_cron_daily_scanner_full.sh --install`
