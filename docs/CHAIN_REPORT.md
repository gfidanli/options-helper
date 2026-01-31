# `chain-report` — Offline Options Chain Dashboard

`chain-report` reads locally stored snapshot files under `data/options_snapshots/` and produces a standardized “what levels matter?” dashboard:

- P/C ratios (OI + volume)
- “Walls” (top OI strikes)
- Expected move (ATM straddle proxy)
- ATM IV term structure (best-effort)
- 25Δ skew (best-effort, requires `bs_delta`)
- Gamma concentration by strike (gross proxy, requires `bs_gamma`)

This command is **offline-first**: it does not call Yahoo.

## Usage

```bash
options-helper chain-report --symbol CVX --as-of 2026-01-30
options-helper chain-report --symbol CVX --as-of latest
```

### Output formats

```bash
options-helper chain-report --symbol CVX --as-of latest --format console
options-helper chain-report --symbol CVX --as-of latest --format md
options-helper chain-report --symbol CVX --as-of latest --format json
```

### Save artifacts

```bash
options-helper chain-report --symbol CVX --as-of latest --out data/reports
```

Writes:
- `data/reports/chains/CVX/{YYYY-MM-DD}.json`
- `data/reports/chains/CVX/{YYYY-MM-DD}.md`

## Snapshot inputs

The snapshot layout must exist (usually via `snapshot-options`):

- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/meta.json`
- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv`

Notes:
- `{YYYY-MM-DD}` is the **data date** (latest available daily candle close), not necessarily the wall-clock run date.
- `meta.json` must include a usable spot (typically `spot`).

## CLI flags

- `--cache-dir PATH`: snapshot root (default `data/options_snapshots`)
- `--as-of YYYY-MM-DD|latest`: snapshot date (default `latest`)
- `--expiries near|monthly|all`: expiry selection mode (default `near`)
- `--include-expiry YYYY-MM-DD`: include specific expiries (repeatable; overrides `--expiries`)
- `--top N`: top strikes to show for walls/gamma (default `10`)
- `--format console|md|json`: output format (default `console`)
- `--out PATH`: write artifacts under `{out}/chains/{SYMBOL}/`
- `--best-effort`: emit warnings and partial output instead of failing on missing fields

## Caveats (best-effort)

- Yahoo chains can be stale/illiquid; bid/ask can be missing or zero.
- Expected move and IV/skew require the relevant strikes/contracts to exist in the snapshot (windowed snapshots can miss them).
- Greeks (`bs_delta`, `bs_gamma`) are locally computed Black–Scholes estimates.

