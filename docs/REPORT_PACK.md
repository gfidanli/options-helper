# `report-pack` — Offline Report Pack (Artifacts for Review)

`report-pack` generates a bundle of **offline** per-symbol artifacts from locally cached data:

- options chain dashboards (levels / walls / IV / gamma proxy)
- snapshot diffs (day-to-day change)
- flow deltas (ΔOI/volume positioning proxy)
- derived stats (percentiles + trend flags)
- IV surface summaries (tenor + delta-bucket structure)
- dealer exposure slices (near/monthly/all)
- actionable levels (daily levels + anchored VWAP/profile when intraday exists)
- technicals extension stats (from candle cache)
- optional per-position scenarios artifacts (portfolio positions)

This tool is for informational/educational use only and is not financial advice.

## Inputs

- Options snapshots: `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/...`
- Candle cache: `data/candles/{SYMBOL}.csv`
- Watchlists: `data/watchlists.json`

## Usage

Default symbol set (watchlists):
- `positions`
- `monitor`
- `Scanner - Shortlist`

Run interactively:

```bash
options-helper report-pack portfolio.json
```

Explicit watchlists:

```bash
options-helper report-pack portfolio.json \
  --watchlists-path data/watchlists.json \
  --watchlist positions \
  --watchlist monitor \
  --watchlist "Scanner - Shortlist"
```

Only generate reports when today's snapshot date exists (recommended for cron):

```bash
options-helper report-pack portfolio.json \
  --require-snapshot-date today \
  --require-snapshot-tz America/Chicago
```

Include per-position scenarios artifacts:

```bash
options-helper report-pack portfolio.json --scenarios
```

## Outputs

Under `--out` (default `data/reports/`):

- Chain dashboard (JSON + Markdown):
  - `chains/{SYMBOL}/{YYYY-MM-DD}.json`
  - `chains/{SYMBOL}/{YYYY-MM-DD}.md`
- Compare (JSON):
  - `compare/{SYMBOL}/{FROM}_to_{TO}.json`
- Flow (JSON):
  - `flow/{SYMBOL}/{FROM}_to_{TO}_w1_contract.json`
  - `flow/{SYMBOL}/{FROM}_to_{TO}_w1_expiry-strike.json`
- Derived stats (JSON):
  - `derived/{SYMBOL}/{ASOF}_w{N}_tw{M}.json`
- IV surface (JSON):
  - `iv_surface/{SYMBOL}/{ASOF}.json`
- Exposure (JSON):
  - `exposure/{SYMBOL}/{ASOF}.json`
- Levels (JSON):
  - `levels/{SYMBOL}/{ASOF}.json`
- Technicals extension stats (JSON + Markdown):
  - `technicals/extension/{SYMBOL}/{YYYY-MM-DD}.json`
  - `technicals/extension/{SYMBOL}/{YYYY-MM-DD}.md`
- Scenarios (JSON, optional):
  - `scenarios/{SYMBOL}/{ASOF}/{POSITION_KEY}.json`

## Notes

- `--as-of latest` is resolved **per symbol** from the snapshot store.
- `--compare-from -1` (default) is resolved **per symbol** relative to the resolved `--as-of`.
- The command is **offline-first**: it reads local snapshot/candle files and writes artifacts; it does not call Yahoo.
- Missing optional data (DuckDB tables, intraday partitions, benchmark candles, prior snapshots) is handled with warnings and skipped panels/artifacts instead of hard failures.
