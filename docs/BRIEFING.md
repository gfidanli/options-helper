# `briefing` — Daily Markdown Report (Offline-First)

`briefing` generates a cron-friendly daily Markdown report from locally stored snapshot files under
`data/options_snapshots/`.

It is intended to make the workflow **regular**:
- one saved artifact per day,
- per-symbol “state + change + flow” highlights,
- optional derived-metrics history updates.

Related: `analyze` also supports an offline/deterministic mode from the same snapshot + candle caches:

```bash
options-helper analyze portfolio.json --offline --as-of latest
```

This tool is for informational/educational use only and is not financial advice.

## Usage

```bash
options-helper briefing portfolio.json --as-of latest --compare -1
```

Include watchlists (in addition to portfolio symbols):

```bash
options-helper briefing portfolio.json \
  --watchlists-path data/watchlists.json \
  --watchlist positions \
  --watchlist monitor \
  --as-of latest \
  --compare -1
```

Write to a custom path (or directory):

```bash
options-helper briefing portfolio.json --as-of 2026-01-30 --out data/reports/daily/2026-01-30.md
options-helper briefing portfolio.json --as-of 2026-01-30 --out data/reports/daily/
```

Disable derived updates:

```bash
options-helper briefing portfolio.json --as-of latest --no-update-derived
```

## Output

By default it writes:
- `data/reports/daily/{YYYY-MM-DD}.md`
- `data/reports/daily/{YYYY-MM-DD}.json` (LLM-friendly)

The report includes:
- Portfolio table (best-effort marks/PnL from the same snapshot day)
  - Includes `Spr%` when bid/ask quotes are available
- Portfolio exposure + stress summary in the JSON payload (best-effort)
  - Exposure: aggregate delta/theta/vega
  - Stress defaults: spot ±5%, IV ±5pp, time +7 days
- Per symbol:
  - Technical context (from cached candles; canonical indicator source is `technicals_backtesting`)
    - Includes extension percentiles + rolling quantiles (1y/3y/5y when available)
  - Next earnings line (date + days until; or `unknown`)
  - Quote quality summary (missing bid/ask %, median/worst spread, stale count) when `meta.json` includes `quote_quality`
  - Vol regime line (RV20/RV60, IV/RV20, IV percentile, term slope; best-effort)
  - Confluence score (coverage + total; best-effort)
  - Chain highlights (walls, near-term EM/IV, gamma peak)
  - Compare highlights (spot + key deltas), if `--compare` is enabled and snapshots exist
  - Flow zones (net, aggregated by strike), if compare snapshots exist
  - Warnings/errors when data is missing
    - Earnings warnings: `earnings_unknown`, `earnings_within_<N>d`, `expiry_crosses_earnings`

JSON payload note:
- `sections[].confluence` includes the full confluence object (total, coverage, components, warnings).

## Inputs (snapshots)

Requires snapshots from `snapshot-options`:
- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/meta.json` (must include a usable spot)
- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv`

Notes:
- `--as-of latest` is resolved **per symbol**.
- Relative `--compare` offsets (e.g., `-1`) are resolved **per symbol** relative to that symbol’s `--as-of`.

## Inputs (candles)

The technical section uses cached daily candles:
- `data/candles/{SYMBOL}.csv`

## CLI flags

- `--as-of YYYY-MM-DD|latest`: snapshot date (default `latest`)
- `--compare -1|-5|YYYY-MM-DD|none`: include compare/flow sections (default `-1`)
- `--cache-dir PATH`: snapshots root (default `data/options_snapshots`)
- `--candle-cache-dir PATH`: candle cache root (default `data/candles`)
- `--technicals-config PATH`: technical indicator config (default `config/technical_backtesting.yaml`)
- `--watchlists-path PATH`: watchlists store path (default `data/watchlists.json`)
- `--watchlist NAME`: include symbols from a watchlist (repeatable)
- `--symbol TICKER`: only include a single symbol (overrides selection)
- `--out PATH`: output Markdown path or directory (default `data/reports/daily/{ASOF}.md`)
- `--print/--no-print`: print the report to the console (default off; use `--print` for interactive runs)
- `--write-json/--no-write-json`: write the JSON artifact (default on)
- `--update-derived/--no-update-derived`: update `data/derived/{SYMBOL}.csv` per symbol (default on)
- `--derived-dir PATH`: derived store directory (default `data/derived`)
- `--top N`: include top N rows in compare/flow sections (default `3`)

## Caveats (best-effort)

- If a symbol is missing snapshots or `meta.json` spot, its section will include errors and skip compare/flow.
- Portfolio “mark” and “PnL” are best-effort matches against snapshot rows (expiry/type/strike).
- Flow is a heuristic derived from day-to-day ΔOI; treat it as a positioning proxy, not certainty.

## Automation (cron)

This repo includes a helper script to generate a daily briefing after snapshots are captured:

- `scripts/cron_daily_briefing.sh`
- `scripts/install_cron_daily_briefing.sh`

Install (prints the crontab block):

```bash
./scripts/install_cron_daily_briefing.sh
```

Install automatically:

```bash
./scripts/install_cron_daily_briefing.sh --install
```

Defaults:
- Runs at **18:00 America/Chicago time**, Monday–Friday.
- Logs to `data/logs/briefing.log`.
