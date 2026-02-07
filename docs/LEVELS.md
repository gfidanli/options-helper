# Levels (Candles + Optional Intraday)

`options-helper market-analysis levels` builds a deterministic levels artifact from cached daily candles and optional intraday bars.

This is informational tooling and **not financial advice**.

## CLI

```bash
./.venv/bin/options-helper market-analysis levels \
  --symbol SPY \
  --benchmark QQQ \
  --as-of latest \
  --format console
```

JSON output + artifact file:

```bash
./.venv/bin/options-helper market-analysis levels \
  --symbol SPY \
  --as-of 2026-02-07 \
  --format json \
  --out data/reports
```

## Inputs (offline-only)

- Daily candles from `data/candles` for symbol + benchmark.
- Optional intraday stock bars from:
  - `data/intraday/stocks/bars/<timeframe>/<SYMBOL>/<YYYY-MM-DD>.csv.gz`

No provider/network fetch is used on this path.

## Metrics included

- Gap + daily levels (spot, prev close, session open, prior high/low, rolling high/low)
- Relative strength + rolling beta/correlation vs benchmark
- Anchored VWAP (session-open anchor)
- Volume profile bins + POC/HVN/LVN candidates

## Output

- Artifact schema: `LevelsArtifact`
- Optional file output: `{out}/levels/<SYMBOL>/<AS_OF>.json`

## Notes

- If intraday bars are missing, daily-level outputs still render with warnings.
- `--as-of` uses candles up to that date (or latest available when `latest`).
