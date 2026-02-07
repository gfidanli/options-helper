# IV Surface (Snapshot-Derived)

`options-helper market-analysis iv-surface` builds an IV surface artifact from cached option snapshots.

This is informational tooling and **not financial advice**.

## CLI

```bash
./.venv/bin/options-helper market-analysis iv-surface \
  --symbol SPY \
  --as-of latest \
  --format console
```

JSON output + artifact file:

```bash
./.venv/bin/options-helper market-analysis iv-surface \
  --symbol SPY \
  --as-of 2026-02-07 \
  --format json \
  --out data/reports
```

## Inputs (offline-first)

- Option snapshot chain from `data/options_snapshots/<SYMBOL>/<DATE>/`.
- Snapshot `meta.json` spot (fallback: cached daily candles from `data/candles/`).
- Optional prior snapshot date for day-over-day change tables.

No network calls are required on this path.

## Output

- Console: tenor + delta-bucket tables.
- JSON artifact schema: `IvSurfaceArtifact`.
- If `--out` is set: `{out}/iv_surface/<SYMBOL>/<AS_OF>.json`.

## DuckDB persistence (optional)

When running with `--storage duckdb` and `--persist` (default), rows are upserted into:

- `iv_surface_tenor`
- `iv_surface_delta_buckets`

with provider tags from the active CLI provider context.

## Notes

- Missing or stale fields in snapshots are tolerated with warning codes.
- If spot cannot be resolved from snapshot meta/candles, spot-dependent metrics may be null.
