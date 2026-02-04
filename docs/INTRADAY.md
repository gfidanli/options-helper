# INTRADAY — Intraday microstructure capture (Alpaca)

This feature captures **intraday bars** for stocks and options using Alpaca and stores them as partitioned
`CSV.gz` + `meta.json` files. It is **not financial advice**.

## Requirements
- Provider: **Alpaca** (`--provider alpaca`)
- Alpaca credentials available via environment or `config/alpaca.env`
- For options: cached contracts from IMP-023 (`data/option_contracts/`)

## CLI
### Stocks
```bash
./.venv/bin/options-helper --provider alpaca intraday pull-stocks-bars \
  --symbol AAPL \
  --day 2026-02-03 \
  --timeframe 1Min
```

### Options
```bash
./.venv/bin/options-helper --provider alpaca intraday pull-options-bars \
  --underlying SPY \
  --contracts-dir data/option_contracts \
  --contracts-as-of latest \
  --day 2026-02-03 \
  --timeframe 1Min \
  --expiries 2026-06-21
```

## Data layout
- Stocks bars:
  - `data/intraday/stocks/bars/1Min/<SYMBOL>/<YYYY-MM-DD>.csv.gz`
  - `data/intraday/stocks/bars/1Min/<SYMBOL>/<YYYY-MM-DD>.meta.json`
- Options bars:
  - `data/intraday/options/bars/1Min/<CONTRACT>/<YYYY-MM-DD>.csv.gz`
  - `data/intraday/options/bars/1Min/<CONTRACT>/<YYYY-MM-DD>.meta.json`

## Schema (bars)
Columns are normalized to:
- `timestamp` (UTC)
- `open`, `high`, `low`, `close`, `volume`
- `trade_count` (optional)
- `vwap` (optional)

## Metadata
Each `meta.json` includes:
- schema version + partition keys
- row counts + coverage timestamps
- provider name/version
- request params (day, timeframe, feed)
- underlying + contracts cache date (options)

## Notes & limitations
- Intraday capture currently supports **Alpaca only**.
- For **today**, bars are requested with a buffer (`OH_ALPACA_RECENT_BARS_BUFFER_MINUTES`, default 16m) to avoid
  plan-based “recent bars” restrictions.
- Options capture depends on the **cached contracts** (IMP-023). Use `--contracts-as-of` to point at a specific cache date.
- Missing bars, partial sessions, or empty partitions are possible (holidays, early closes, provider limits).
