from __future__ import annotations

import pandas as pd

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.data.stores_duckdb import DuckDBCandleStore


def test_duckdb_candle_store_save_and_load(tmp_path):
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)

    store = DuckDBCandleStore(root_dir=tmp_path / "candles", warehouse=wh)

    idx = pd.to_datetime(["2026-02-01", "2026-02-02"])
    history = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.0, 102.5],
            "Volume": [10_000, 12_000],
        },
        index=idx,
    )

    store.save("AAPL", history)

    loaded = store.load("AAPL")
    assert not loaded.empty
    assert list(loaded.columns)[:5] == ["Open", "High", "Low", "Close", "Volume"]
    assert len(loaded) == 2
    assert float(loaded.iloc[-1]["Close"]) == 102.5

    meta = store.load_meta("AAPL")
    assert meta is not None
    assert meta.get("symbol") == "AAPL"
    assert int(meta.get("rows")) == 2
