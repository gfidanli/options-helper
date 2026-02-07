from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.data.providers.runtime import reset_default_provider_name, set_default_provider_name
from options_helper.data.stores_duckdb import DuckDBOptionsSnapshotStore


def test_duckdb_options_snapshot_store_save_and_load(tmp_path):
    token = set_default_provider_name("yahoo")
    try:
        wh = DuckDBWarehouse(tmp_path / "options.duckdb")
        ensure_schema(wh)

        store = DuckDBOptionsSnapshotStore(lake_root=tmp_path / "options_snapshots", warehouse=wh)

        snapshot_date = date(2026, 2, 1)
        exp = date(2026, 3, 15)
        chain = pd.DataFrame(
            [
                {
                    "contractSymbol": "AAPL260315C00100000",
                    "strike": 100.0,
                    "expiry": exp.isoformat(),
                    "optionType": "call",
                    "bid": 1.0,
                    "ask": 1.1,
                },
                {
                    "contractSymbol": "AAPL260315P00100000",
                    "strike": 100.0,
                    "expiry": exp.isoformat(),
                    "optionType": "put",
                    "bid": 1.2,
                    "ask": 1.3,
                },
            ]
        )

        raw_by_expiry = {exp: {"calls": [], "puts": [], "underlying": {"symbol": "AAPL"}}}
        meta = {"spot": 101.23, "risk_free_rate": 0.01, "full_chain": True}

        out_path = store.save_day_snapshot(
            "AAPL",
            snapshot_date,
            chain=chain,
            expiries=[exp],
            raw_by_expiry=raw_by_expiry,
            meta=meta,
        )
        assert out_path.exists()

        dates = store.list_dates("AAPL")
        assert dates == [snapshot_date]

        loaded = store.load_day("AAPL", snapshot_date)
        assert len(loaded) == 2
        assert set(loaded["optionType"].astype(str)) == {"call", "put"}
    finally:
        reset_default_provider_name(token)


def test_duckdb_options_snapshot_store_can_skip_legacy_files(tmp_path):
    token = set_default_provider_name("yahoo")
    try:
        wh = DuckDBWarehouse(tmp_path / "options.duckdb")
        ensure_schema(wh)

        store = DuckDBOptionsSnapshotStore(
            lake_root=tmp_path / "options_snapshots",
            warehouse=wh,
            sync_legacy_files=False,
        )

        snapshot_date = date(2026, 2, 1)
        exp = date(2026, 3, 15)
        chain = pd.DataFrame(
            [
                {
                    "contractSymbol": "AAPL260315C00100000",
                    "strike": 100.0,
                    "expiry": exp.isoformat(),
                    "optionType": "call",
                    "bid": 1.0,
                    "ask": 1.1,
                }
            ]
        )

        out_path = store.save_day_snapshot(
            "AAPL",
            snapshot_date,
            chain=chain,
            expiries=[exp],
            raw_by_expiry={exp: {"calls": [], "puts": []}},
            meta={"spot": 101.23},
        )
        assert out_path.exists()

        day_dir = tmp_path / "options_snapshots" / "AAPL" / snapshot_date.isoformat()
        assert (day_dir / "chain.parquet").exists()
        assert not (day_dir / f"{exp.isoformat()}.csv").exists()
        assert not (day_dir / "meta.json").exists()
        assert (day_dir / "meta.json.gz").exists()

        loaded = store.load_day("AAPL", snapshot_date)
        assert len(loaded) == 1
        assert loaded.iloc[0]["contractSymbol"] == "AAPL260315C00100000"
    finally:
        reset_default_provider_name(token)
