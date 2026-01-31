# Earnings dates (best-effort)

This repo can cache the **next earnings date** per symbol for quick reference in workflows.

Notes / caveats:
- Data is fetched from **Yahoo Finance via `yfinance`**, which can be missing, stale, or timezone-shifted.
- Yahoo sometimes provides an **earnings window** (start/end dates) instead of a single timestamp.
- Treat this as a convenience signal, not a source of truth.

## CLI

Fetch and cache the next earnings date:
```bash
./.venv/bin/options-helper earnings IREN --refresh
```

Show the cached record (without hitting the network):
```bash
./.venv/bin/options-helper earnings IREN
```

Refresh earnings for all symbols in your watchlists:
```bash
./.venv/bin/options-helper refresh-earnings
```

Refresh earnings for a specific watchlist (repeatable):
```bash
./.venv/bin/options-helper refresh-earnings --watchlist positions --watchlist watchlist
```

Manually set/override a date:
```bash
./.venv/bin/options-helper earnings IREN --set 2026-02-06
```

Delete the cached record:
```bash
./.venv/bin/options-helper earnings IREN --clear
```

## Storage layout

By default, earnings records are stored under:
- `data/earnings/{SYMBOL}.json`

You can override the directory with `--cache-dir`.
