# `compare` — Snapshot Diff Engine (Offline)

`compare` diffs two locally stored snapshot dates for a symbol and surfaces changes in:

- spot
- P/C ratios
- key expiries’ ATM IV and expected move (best-effort)
- wall changes (ΔOI at key strikes)
- contract-level ΔOI (reuses `flow` computation)

This command is **offline-first**: it does not call Yahoo.

## Usage

Explicit dates:

```bash
options-helper compare --symbol CVX --from 2026-01-29 --to 2026-01-30
```

Relative date (previous snapshot):

```bash
options-helper compare --symbol CVX --to latest --from -1
```

## Save artifacts

```bash
options-helper compare --symbol CVX --to latest --from -1 --out data/reports
```

Writes:
- `data/reports/compare/CVX/{FROM}_to_{TO}.json`

The JSON includes:
- `from` metrics (same schema as `chain-report`)
- `to` metrics
- `diff` summary

## CLI flags

- `--cache-dir PATH`: snapshot root (default `data/options_snapshots`)
- `--from YYYY-MM-DD|-N`: from snapshot (default `-1`)
- `--to YYYY-MM-DD|latest`: to snapshot (default `latest`)
- `--top N`: cap rows per section (default `10`)
- `--out PATH`: write artifacts under `{out}/compare/{SYMBOL}/`

## Caveats (best-effort)

- Requires `meta.json` spot for both dates.
- If expiries differ between dates, only common expiries are compared (uses a “near” subset by default).

