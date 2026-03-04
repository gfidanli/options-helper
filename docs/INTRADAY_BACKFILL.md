# Intraday Backfill (Stocks, 1Min)

`options-helper intraday-backfill stocks-history` runs a historical 1-minute stock backfill from Alpaca with high-visibility status artifacts. This command is informational tooling only and **not financial advice**.

## What it does
- Loads the Alpaca tradable US equity universe.
- Excludes symbols listed in `data/universe/exclude_symbols.txt`.
- Pulls bars symbol-by-symbol and month-by-month (`YYYY-MM`).
- Writes partitions under `data/intraday/stocks/bars/1Min/<SYMBOL>/<YYYY-MM-DD>.csv.gz`.
- Skips month fetches when all expected market days already exist locally.

## Status visibility
By default, status is written under `data/intraday_backfill_status/<RUN_ID>/`:

- `status/overall.json`: run-level counters and throughput.
- `status/current_symbol.json`: currently active symbol/month.
- `status/symbols/<SYMBOL>.json`: symbol-level month results.
- `results_symbol_month.jsonl`: append-only symbol/month records.
- `failures.csv`: symbol/month errors.

## Checkpoint behavior
Default checkpoint is after 25 processed symbols:
- writes `performance_checkpoint_25_symbols.md`
- writes `performance_checkpoint_25_symbols.json`
- pauses run (`--pause-at-checkpoint`)

Use `--no-pause-at-checkpoint` to continue automatically after report generation.

## Example
```bash
./.venv/bin/options-helper --provider alpaca intraday-backfill stocks-history \
  --exclude-path data/universe/exclude_symbols.txt \
  --out-dir data/intraday \
  --status-dir data/intraday_backfill_status \
  --feed sip \
  --checkpoint-symbols 25
```

## Common flags
- `--start-date YYYY-MM-DD` (default: `2000-01-01`)
- `--end-date YYYY-MM-DD` (default: last completed market day)
- `--max-symbols N` (for staged runs)
- `--run-id TEXT` (stable status folder name)
- `--checkpoint-symbols N` (`0` disables checkpoint)
- `--pause-at-checkpoint/--no-pause-at-checkpoint`

## Notes
- Provider must be Alpaca (`--provider alpaca`).
- The run can be storage-heavy (many symbol/day files).
- Existing partitions are treated as authoritative and are not overwritten.
