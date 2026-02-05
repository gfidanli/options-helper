from __future__ import annotations

from pathlib import Path
from typing import Any

from options_helper.data.candles import CandleStore
from options_helper.data.derived import DerivedStore
from options_helper.data.journal import JournalStore
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.data.storage_runtime import get_storage_runtime_config

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


_WAREHOUSE_CACHE: dict[Path, DuckDBWarehouse] = {}


def get_warehouse(path: Path | None = None) -> DuckDBWarehouse:
    cfg = get_storage_runtime_config()
    resolved = path or cfg.duckdb_path
    resolved = Path(resolved)
    wh = _WAREHOUSE_CACHE.get(resolved)
    if wh is None:
        wh = DuckDBWarehouse(resolved)
        ensure_schema(wh)
        _WAREHOUSE_CACHE[resolved] = wh
    return wh


def close_warehouses() -> None:
    # Currently we don't keep long-lived connections, but this hook lets the CLI
    # clean up cached warehouses and makes future refactors safer.
    _WAREHOUSE_CACHE.clear()


def get_candle_store(root_dir: Path, **kwargs: Any) -> CandleStore:
    cfg = get_storage_runtime_config()
    if cfg.backend == "duckdb":
        from options_helper.data.stores_duckdb import DuckDBCandleStore

        return DuckDBCandleStore(root_dir=root_dir, warehouse=get_warehouse(), **kwargs)
    return CandleStore(root_dir, **kwargs)


def get_derived_store(root_dir: Path) -> DerivedStore:
    cfg = get_storage_runtime_config()
    if cfg.backend == "duckdb":
        from options_helper.data.stores_duckdb import DuckDBDerivedStore

        return DuckDBDerivedStore(root_dir=root_dir, warehouse=get_warehouse())
    return DerivedStore(root_dir)


def get_journal_store(root_dir: Path) -> JournalStore:
    cfg = get_storage_runtime_config()
    if cfg.backend == "duckdb":
        from options_helper.data.stores_duckdb import DuckDBJournalStore

        return DuckDBJournalStore(root_dir=root_dir, warehouse=get_warehouse())
    return JournalStore(root_dir)


def get_options_snapshot_store(root_dir: Path) -> OptionsSnapshotStore:
    cfg = get_storage_runtime_config()
    if cfg.backend == "duckdb":
        from options_helper.data.stores_duckdb import DuckDBOptionsSnapshotStore

        return DuckDBOptionsSnapshotStore(lake_root=root_dir, warehouse=get_warehouse())
    return OptionsSnapshotStore(root_dir)
