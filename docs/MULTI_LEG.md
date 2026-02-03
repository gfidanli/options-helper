# Multi-leg positions (spreads & calendars)

This repo now supports **multi-leg option positions** in the portfolio model and core CLI flows.
Outputs are **not financial advice** — treat them as best-effort diagnostics.

## Portfolio schema (multi-leg)

Multi-leg positions use a `legs` array under a position item:

```json
{
  "id": "aapl-ml-lc100@2026-04-17-sc105@2026-04-17",
  "symbol": "AAPL",
  "net_debit": 150.0,
  "legs": [
    {"side": "long", "option_type": "call", "expiry": "2026-04-17", "strike": 100, "contracts": 1},
    {"side": "short", "option_type": "call", "expiry": "2026-04-17", "strike": 105, "contracts": 1}
  ]
}
```

Notes:
- `net_debit` is **total dollars** for the structure (used for net PnL).
- `legs` must include at least two entries.
- `ratio` is accepted but currently informational only (not yet applied in math).

## CLI: add a spread

Use `add-spread` with repeatable `--leg` flags:

```bash
options-helper add-spread portfolio.json \
  --symbol AAPL \
  --leg "long,call,2026-04-17,100,1" \
  --leg "short,call,2026-04-17,105,1" \
  --net-debit 150 \
  --id aapl-vertical-1
```

Leg format:
```
side,type,expiry,strike,contracts[,ratio]
```

## Analyze output

`analyze` now shows:
- a **Multi-leg Positions** table (net mark + net PnL)
- a per-leg breakdown (mark, IV, OI, volume, spreads)

Best-effort caveats:
- If any leg is missing a mark, the net mark is blank.
- Per-leg PnL is not computed unless you provide per-leg cost basis (not yet supported).

## Roll planning

`roll-plan` supports **2-leg same-expiry verticals** (v1).

Current behavior:
- Keeps strikes (preserves width)
- Rolls to expiries closest to `--horizon-months`
- Requires both strikes to exist in the snapshot day

Limitations (v1):
- Calendars/diagonals are not yet supported.
- Net roll cost is derived from snapshot marks (best-effort).

## Data quality notes

- Quotes/IV can be stale or missing in Yahoo snapshots.
- Wide spreads or low OI/volume will be flagged at the structure level.
- Always validate any action in your broker platform.
