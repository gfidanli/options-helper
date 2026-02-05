# options_helper/data/ — Data Access, Caches, Snapshots

## Responsibilities
- All external I/O lives here:
  - `yfinance` calls
  - candle cache reads/writes
  - options snapshot reads/writes

## yfinance hardening
- Assume fields can be missing, stale, or zero (`bid/ask=0`, `IV=0`, etc.).
- Wrap errors in project exceptions (e.g. `DataFetchError`) with actionable messages.
- Keep `yfinance` usage behind `YFinanceClient` where possible.

## Candle cache rules
- Always normalize indexes to timezone-naive `DatetimeIndex`.
- Update incrementally (backfill + refresh tail); avoid re-downloading multi-year windows per run.

## Snapshot rules
- Snapshot folders should use the **data date** (latest daily candle date) rather than wall-clock run date.
- Keep snapshots windowed around spot by default (configurable).
- Store metadata in `meta.json` (spot, strike window, snapshot_date).
- When the DuckDB backend is enabled, keep filesystem artifacts in sync for compatibility:
  - snapshots: write/read legacy day-folder CSV + `meta.json` alongside header/parquet rows
  - journal: mirror writes to `signal_events.jsonl` for filesystem readers
  - derived: mirror per-symbol CSV writes so `root_dir`-scoped workflows stay deterministic
- In DuckDB snapshot reads, scope header rows to paths under the active `lake_root` to avoid cross-root date contamination.

## Testing
- Do not hit the network in tests.
- Prefer dependency injection (e.g. candle fetcher function) or monkeypatching client methods.
