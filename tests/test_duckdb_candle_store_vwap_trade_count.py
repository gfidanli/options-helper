from __future__ import annotations

import pandas as pd

from options_helper.data.stores_duckdb import DuckDBCandleStore
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def test_duckdb_candle_store_preserves_vwap_trade_count(tmp_path):
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
            "VWAP": [100.5, None],
            "Trade Count": [12, None],
        },
        index=idx,
    )

    store.save("AAPL", history)

    loaded = store.load("AAPL")
    assert "VWAP" in loaded.columns
    assert "Trade Count" in loaded.columns
    assert float(loaded.iloc[0]["VWAP"]) == 100.5
    assert int(loaded.iloc[0]["Trade Count"]) == 12
    assert pd.isna(loaded.iloc[1]["VWAP"])
    assert pd.isna(loaded.iloc[1]["Trade Count"])
