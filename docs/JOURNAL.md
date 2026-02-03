# Journal (Signal Logging + Outcome Tracking)

This repo provides a **signal journal** to measure how daily signals performed over time.  
It is **informational only — not financial advice**.

## What gets logged
Signals are stored as **append-only JSONL** under:

- `data/journal/signal_events.jsonl`

Each entry includes:
- `date`, `symbol`, `context` (`position`, `research`, `scanner`)
- `payload` (action, reasons, key metrics)
- `snapshot_date` + `contract_symbol` when available (used for option mark outcomes)

## Commands

### Log signals
Positions are logged by default. Research/scanner are optional.

```bash
./.venv/bin/options-helper journal log portfolio.json --as-of latest
./.venv/bin/options-helper journal log portfolio.json --research --scanner
```

**Notes**
- `--offline` uses cached candles + snapshots (best-effort, deterministic).
- Research uses `yfinance` options chains (online) unless you skip `--research`.
- Scanner logging reads the latest `shortlist.csv` from `data/scanner/runs/`.

### Evaluate outcomes
Creates JSON + Markdown reports under `data/reports/journal/`.

```bash
./.venv/bin/options-helper journal evaluate --window 252
```

Output:
- `data/reports/journal/YYYY-MM-DD.json`
- `data/reports/journal/YYYY-MM-DD.md`

## How outcomes are computed
- **Underlying return:** uses cached daily closes (trading-day horizons).
- **Option return:** uses snapshot marks for the same `contractSymbol` on the horizon date.
- Missing data → `None` (no guesswork).

## Data quality caveats
- `yfinance` data is **best-effort** (stale/zero quotes happen).
- Snapshots may be missing for some symbols or dates.
- Outcome windows are trading-day offsets, not calendar days.

If you need stricter guarantees, run with:
- `journal log --offline --offline-strict`
