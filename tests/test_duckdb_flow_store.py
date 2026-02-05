from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from options_helper import cli_deps
from options_helper.data.flow_store import DuckDBFlowStore, NoopFlowStore
from options_helper.data.storage_runtime import (
    reset_default_duckdb_path,
    reset_default_storage_backend,
    set_default_duckdb_path,
    set_default_storage_backend,
)
from options_helper.data.store_factory import close_warehouses
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.flow import FlowArtifact


def _sample_artifact(*, delta_oi_notional: float = 120_000.0) -> FlowArtifact:
    return FlowArtifact(
        schema_version=1,
        generated_at=datetime(2026, 2, 5, 14, 30, tzinfo=timezone.utc),
        as_of="2026-02-05",
        symbol="aapl",
        from_date="2026-02-04",
        to_date="2026-02-05",
        window=2,
        group_by="expiry-strike",
        snapshot_dates=["2026-02-03", "2026-02-04", "2026-02-05"],
        net=[
            {
                "expiry": "2026-03-20",
                "option_type": "call",
                "strike": 100.0,
                "delta_oi": 15.0,
                "delta_oi_notional": delta_oi_notional,
                "volume_notional": 240_000.0,
                "delta_notional": 95_000.0,
                "n_pairs": 2,
            },
            {
                "expiry": "2026-03-20",
                "option_type": "put",
                "strike": 95.0,
                "delta_oi": -8.0,
                "delta_oi_notional": -80_000.0,
                "volume_notional": 130_000.0,
                "delta_notional": -60_000.0,
                "n_pairs": 2,
            },
        ],
    )


def test_duckdb_flow_store_upsert_and_load_artifact(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(warehouse)
    store = DuckDBFlowStore(root_dir=tmp_path / "flow", warehouse=warehouse)

    artifact = _sample_artifact()
    written = store.upsert_artifact(artifact)
    assert written == 2

    loaded = store.load_artifact(
        symbol="AAPL",
        from_date="2026-02-04",
        to_date="2026-02-05",
        window=2,
        group_by="expiry-strike",
    )
    assert loaded is not None
    assert loaded.symbol == "AAPL"
    assert loaded.from_date == "2026-02-04"
    assert loaded.to_date == "2026-02-05"
    assert loaded.group_by == "expiry-strike"
    assert loaded.window == 2
    assert loaded.snapshot_dates == ["2026-02-03", "2026-02-04", "2026-02-05"]
    assert len(loaded.net) == 2
    assert loaded.net[0].option_type == "call"
    assert loaded.net[1].option_type == "put"

    conn = warehouse.connect(read_only=True)
    try:
        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM options_flow
            WHERE symbol = 'AAPL'
              AND from_date = DATE '2026-02-04'
              AND to_date = DATE '2026-02-05'
              AND window_size = 2
              AND group_by = 'expiry-strike'
            """
        ).fetchone()
        assert count is not None
        assert count[0] == 2
    finally:
        conn.close()


def test_duckdb_flow_store_partition_upsert_replaces_rows(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(warehouse)
    store = DuckDBFlowStore(root_dir=tmp_path / "flow", warehouse=warehouse)

    store.upsert_artifact(_sample_artifact())
    store.upsert_artifact(_sample_artifact(delta_oi_notional=333_000.0))

    rows = store.load_rows(
        symbol="AAPL",
        from_date="2026-02-04",
        to_date="2026-02-05",
        window=2,
        group_by="expiry-strike",
    )
    assert len(rows) == 2
    assert float(rows.iloc[0]["delta_oi_notional"]) == 333_000.0

    conn = warehouse.connect(read_only=True)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*), MIN(delta_oi_notional), MAX(delta_oi_notional)
            FROM options_flow
            WHERE symbol = 'AAPL'
              AND from_date = DATE '2026-02-04'
              AND to_date = DATE '2026-02-05'
              AND window_size = 2
              AND group_by = 'expiry-strike'
            """
        ).fetchone()
        assert row is not None
        assert row[0] == 2
        assert float(row[1]) == -80_000.0
        assert float(row[2]) == 333_000.0
    finally:
        conn.close()


def test_cli_deps_build_flow_store_duckdb_and_filesystem(tmp_path: Path) -> None:
    duckdb_backend_token = set_default_storage_backend("duckdb")
    duckdb_path_token = set_default_duckdb_path(tmp_path / "options.duckdb")
    try:
        duckdb_store = cli_deps.build_flow_store(tmp_path / "flow")
        assert isinstance(duckdb_store, DuckDBFlowStore)
        assert duckdb_store.upsert_artifact(_sample_artifact()) == 2

        rows = duckdb_store.load_rows(
            symbol="AAPL",
            from_date="2026-02-04",
            to_date="2026-02-05",
            window=2,
            group_by="expiry-strike",
        )
        assert len(rows) == 2
    finally:
        close_warehouses()
        reset_default_duckdb_path(duckdb_path_token)
        reset_default_storage_backend(duckdb_backend_token)

    filesystem_backend_token = set_default_storage_backend("filesystem")
    try:
        filesystem_store = cli_deps.build_flow_store(tmp_path / "flow")
        assert isinstance(filesystem_store, NoopFlowStore)
        assert filesystem_store.upsert_artifact(_sample_artifact()) == 0
        assert filesystem_store.load_artifact(
            symbol="AAPL",
            from_date="2026-02-04",
            to_date="2026-02-05",
            window=2,
            group_by="expiry-strike",
        ) is None
        assert filesystem_store.load_rows(
            symbol="AAPL",
            from_date="2026-02-04",
            to_date="2026-02-05",
            window=2,
            group_by="expiry-strike",
        ).empty
    finally:
        close_warehouses()
        reset_default_storage_backend(filesystem_backend_token)
