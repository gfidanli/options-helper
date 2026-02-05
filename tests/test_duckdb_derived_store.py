from __future__ import annotations

from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.data.derived import DERIVED_COLUMNS
from options_helper.data.stores_duckdb import DuckDBDerivedStore


def test_duckdb_derived_store_upsert_and_load(tmp_path):
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)

    store = DuckDBDerivedStore(root_dir=tmp_path / "derived", warehouse=wh)

    row = DerivedRow(date="2026-02-01", spot=100.0, pc_oi=1.1)
    store.upsert("AAPL", row)

    df = store.load("AAPL")
    assert list(df.columns) == DERIVED_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["date"] == "2026-02-01"
    assert float(df.iloc[0]["spot"]) == 100.0

    # Upsert same date overwrites.
    row2 = DerivedRow(date="2026-02-01", spot=101.0, pc_oi=1.2)
    store.upsert("AAPL", row2)

    df2 = store.load("AAPL")
    assert len(df2) == 1
    assert float(df2.iloc[0]["spot"]) == 101.0
    assert float(df2.iloc[0]["pc_oi"]) == 1.2
