# EVENTS — Corporate actions + news ingestion (Alpaca)

This feature ingests **corporate actions** and **news** from Alpaca for event-aware analytics and journaling.
It is **not financial advice**.

## Requirements
- Provider: **Alpaca** (`--provider alpaca`)
- Alpaca credentials available via environment or `config/alpaca.env`

## CLI
### Corporate actions
```bash
./.venv/bin/options-helper --provider alpaca events refresh-corporate-actions \
  --symbols AAPL,MSFT \
  --start 2026-02-01 \
  --end 2026-02-28
```

### News
```bash
./.venv/bin/options-helper --provider alpaca events refresh-news \
  --symbols AAPL \
  --start 2026-02-01 \
  --end 2026-02-28
```

To include full article content where available:
```bash
./.venv/bin/options-helper --provider alpaca events refresh-news \
  --symbols AAPL \
  --start 2026-02-01 \
  --end 2026-02-28 \
  --include-content
```

## Data layout
- Corporate actions:
  - `data/events/corporate_actions/<SYMBOL>.json`
- News:
  - `data/events/news/<SYMBOL>/<YYYY-MM-DD>.jsonl.gz`
  - `data/events/news/<SYMBOL>/<YYYY-MM-DD>.meta.json`

## Schemas (normalized)
### Corporate actions
- `type`
- `symbol`
- `ex_date` / `record_date` / `pay_date` (when present)
- `ratio` (splits) or `cash_amount` (dividends)
- `raw` (best-effort raw payload)

### News
- `id`
- `created_at`
- `headline`
- `summary`
- `source`
- `symbols`
- `content` (optional, when `--include-content` is set)

## Notes & limitations
- Ingestion is **best-effort** and depends on provider coverage and plan limits.
- News is stored **partitioned by day** for incremental refresh and smaller files.
- Corporate actions are merged per symbol; duplicates are deduped by core fields.
