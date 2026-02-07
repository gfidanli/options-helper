# Intraday Flow (Offline Summarizer)

`options-helper intraday flow` summarizes captured options trades/quotes partitions into contract/day flow rows and UTC time buckets.

This is informational tooling and **not financial advice**.

## CLI

By underlying:

```bash
./.venv/bin/options-helper intraday flow \
  --underlying SPY \
  --day latest \
  --format console
```

By explicit contract:

```bash
./.venv/bin/options-helper intraday flow \
  --contract SPY260320C00450000 \
  --day 2026-02-05 \
  --format json \
  --out data/reports
```

## Inputs

Reads only local partitions from `--out-dir` (default `data/intraday`):

- `options/trades/<timeframe>/<CONTRACT>/<YYYY-MM-DD>.csv.gz`
- `options/quotes/<timeframe>/<CONTRACT>/<YYYY-MM-DD>.csv.gz`

No network calls are made for this command.

## Time handling

- `--day` is the market-date partition key (`YYYY-MM-DD` or `latest`).
- Output `bucket_start_utc` values are UTC timestamps.
- Bucket sizes are deterministic (`--bucket-minutes` supports `5` or `15`).

## Output

- Artifact schema: `IntradayFlowArtifact`
- Console summary tables (contracts + top UTC buckets)
- Optional file output: `{out}/intraday_flow/<SYMBOL>/<MARKET_DATE>.json`

## DuckDB persistence (optional)

When running with `--storage duckdb` and `--persist` (default), contract-flow rows are upserted into:

- `intraday_option_flow`
