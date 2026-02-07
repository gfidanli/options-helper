# Dealer Exposure (Strike Ladder)

`options-helper market-analysis exposure` computes strike-level gamma exposure from cached option snapshots.

This is informational tooling and **not financial advice**.

## CLI

```bash
./.venv/bin/options-helper market-analysis exposure \
  --symbol SPY \
  --as-of latest \
  --format console
```

JSON output + artifact file:

```bash
./.venv/bin/options-helper market-analysis exposure \
  --symbol SPY \
  --as-of 2026-02-07 \
  --format json \
  --out data/reports
```

## Inputs (offline-first)

- Option snapshot chain from `data/options_snapshots/<SYMBOL>/<DATE>/`.
- Spot from snapshot `meta.json` (fallback: cached daily candles).

No network calls are required on this path.

## What it returns

`ExposureArtifact` with slices:

- `near`
- `monthly`
- `all`

Each slice includes:

- strike rows (`call_oi`, `put_oi`, `call_gex`, `put_gex`, `net_gex`)
- summary (`flip_strike`, totals, warnings)
- top absolute net levels

Signed convention is fixed to `calls_positive_puts_negative`.

## Output

- Console summary tables per slice.
- JSON artifact under `{out}/exposure/<SYMBOL>/<AS_OF>.json` when `--out` is provided.

## DuckDB persistence (optional)

When running with `--storage duckdb` and `--persist` (default), the `all` strike rows are upserted into:

- `dealer_exposure_strikes`
