# EPIC — DuckDB warehouse + Parquet lake storage backend (local-first)

- **Status:** draft
- **Effort:** L
- **Alpha potential:** Very High

## Summary
Replace the current “many CSV/JSON files” storage approach with a **local-first analytics storage stack**:

- **DuckDB** as an embedded “warehouse” for metadata, indexing, and small/medium tables.
- **Parquet** files as a “lake” for large, append-heavy data like **full option-chain snapshots** and potentially intraday data later.

The goal is to keep the project **single-trader / single-machine friendly** while making it scalable and professional:
fast reads, idempotent writes, schema evolution via migrations, and a clean path to multi-user later.

## Storage stack
- **Warehouse:** DuckDB file (default path: `data/warehouse/options.duckdb`)
- **Lake:** Parquet partitions (default path: `data/lake/…` under existing cache dirs)
- **Python integration:** `duckdb` Python package + pandas

## Plan inventory (implementation order + deps)

| Order | Plan | Status | Effort | Depends on | Key outputs |
|---:|---|---|:---:|---|---|
| 1 | IMP-040 — DuckDB scaffold + migrations + CLI toggles | draft | S–M | — | DuckDB module, schema migrations, `--storage` runtime |
| 2 | IMP-041 — Derived metrics + Journal → DuckDB | draft | S–M | IMP-040 | `DerivedStore`/`JournalStore` work in DuckDB mode |
| 3 | IMP-042 — Candle cache → DuckDB | draft | M | IMP-040 | Daily OHLCV cached in DuckDB with settings-aware meta |
| 4 | IMP-043 — Options snapshot lake (Parquet) + DuckDB index | draft | M–L | IMP-040 | Full-chain snapshots: Parquet facts + DuckDB header/index |

## Critical design decisions (make these once)
1. **Local-first, single file DB**
   - Use a single DuckDB file per environment (dev/prod) for fast iteration.
   - Keep defaults under `data/warehouse/` and allow overrides.

2. **Warehouse vs lake split**
   - DuckDB is great for analytics, but very large fact tables (e.g., full-chain snapshots across many symbols/dates)
     are best stored as **partitioned Parquet**.
   - Store **paths + metadata + “what exists” indexes** in DuckDB, while large snapshot rows live in Parquet.

3. **Provider-aware storage**
   - Key snapshot headers by `(symbol, snapshot_date, provider)` so you can store Yahoo + Alpaca side-by-side.
   - Provider name comes from the existing provider runtime (`--provider`).

4. **Backwards compatible by default**
   - Filesystem stores remain the default to avoid breaking existing workflows.
   - DuckDB is enabled via a new top-level flag `--storage duckdb`.
   - Existing CLI commands should work unchanged (just faster / more scalable).

5. **Idempotent writes**
   - “Upsert” semantics for candles, derived, journal, snapshots.
   - Avoid duplicate rows and allow re-runs without manual cleanup.

## Execution guidelines
- Keep tests **offline and deterministic** (no network calls).
- Prefer **additive** schema changes; version via migrations.
- Keep storage runtime configuration via `contextvars` so the CLI can set it once.
- Avoid heavy deps (e.g., no mandatory `pyarrow`); use DuckDB `COPY` for Parquet writes.

## Definition of done
- `options-helper db init` creates the DuckDB file + schema.
- Running any existing command with `--storage duckdb` works end-to-end (no CSV explosion).
- Options full-chain snapshots are stored as Parquet partitions and discoverable/queryable via DuckDB metadata.
- Clear docs: setup, file locations, migration strategy, and operational notes (backup/restore).
