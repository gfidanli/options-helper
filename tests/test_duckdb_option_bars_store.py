from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from options_helper.data.stores_duckdb import DuckDBOptionBarsStore
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def test_duckdb_option_bars_store_upsert_and_meta(tmp_path):
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)

    store = DuckDBOptionBarsStore(root_dir=tmp_path / "option_bars", warehouse=wh)

    df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL260315C00100000",
                "ts": "2026-02-01",
                "open": 1.0,
                "high": 1.2,
                "low": 0.9,
                "close": 1.1,
                "volume": 100,
                "vwap": 1.05,
                "trade_count": 10,
            },
            {
                "contractSymbol": "AAPL260315C00100000",
                "ts": "2026-02-01",
                "open": 1.0,
                "high": 1.3,
                "low": 0.8,
                "close": 1.15,
                "volume": 120,
                "vwap": 1.07,
                "trade_count": 11,
            },
            {
                "contractSymbol": "AAPL260315C00100000",
                "ts": "2026-02-02",
                "open": 1.2,
                "high": 1.4,
                "low": 1.1,
                "close": 1.3,
                "volume": 140,
                "vwap": 1.25,
                "trade_count": 12,
            },
            {
                "contractSymbol": "AAPL260221P00095000",
                "ts": "2026-02-01",
                "open": 2.0,
                "high": 2.2,
                "low": 1.9,
                "close": 2.1,
                "volume": 80,
                "vwap": 2.05,
                "trade_count": 8,
            },
        ]
    )

    store.upsert_bars(df, provider="alpaca", interval="1d")

    bars = wh.fetch_df(
        """
        SELECT contract_symbol, ts, close, volume
        FROM option_bars
        WHERE provider = ? AND interval = ?
        ORDER BY contract_symbol ASC, ts ASC
        """,
        ["alpaca", "1d"],
    )
    assert len(bars) == 3
    first = bars[(bars["contract_symbol"] == "AAPL260315C00100000")].iloc[0]
    assert float(first["close"]) == 1.15
    assert float(first["volume"]) == 120

    df2 = df.copy()
    df2.loc[df2["contractSymbol"] == "AAPL260221P00095000", "close"] = 2.2
    store.upsert_bars(df2, provider="alpaca", interval="1d")

    bars2 = wh.fetch_df(
        """
        SELECT contract_symbol, ts, close
        FROM option_bars
        WHERE provider = ? AND interval = ?
        ORDER BY contract_symbol ASC, ts ASC
        """,
        ["alpaca", "1d"],
    )
    assert len(bars2) == 3
    updated = bars2[bars2["contract_symbol"] == "AAPL260221P00095000"].iloc[0]
    assert float(updated["close"]) == 2.2

    symbols = ["AAPL260315C00100000", "AAPL260221P00095000"]
    rows_by_symbol = {"AAPL260315C00100000": 2, "AAPL260221P00095000": 1}
    start_by_symbol = {
        "AAPL260315C00100000": datetime(2026, 2, 1, tzinfo=timezone.utc),
        "AAPL260221P00095000": datetime(2026, 2, 1, tzinfo=timezone.utc),
    }
    end_by_symbol = {
        "AAPL260315C00100000": datetime(2026, 2, 2, tzinfo=timezone.utc),
        "AAPL260221P00095000": datetime(2026, 2, 1, tzinfo=timezone.utc),
    }
    store.mark_meta_success(
        symbols,
        interval="1d",
        provider="alpaca",
        rows=rows_by_symbol,
        start_ts=start_by_symbol,
        end_ts=end_by_symbol,
    )

    meta = wh.fetch_df(
        """
        SELECT contract_symbol, status, rows, start_ts, end_ts, error_count
        FROM option_bars_meta
        WHERE provider = ? AND interval = ?
        ORDER BY contract_symbol ASC
        """,
        ["alpaca", "1d"],
    )
    assert list(meta["status"].astype(str)) == ["ok", "ok"]
    assert list(meta["rows"].astype(int)) == [1, 2]

    store.mark_meta_error(
        ["AAPL260221P00095000"],
        interval="1d",
        provider="alpaca",
        error="boom",
        status="error",
    )
    meta2 = wh.fetch_df(
        """
        SELECT contract_symbol, status, error_count, last_error
        FROM option_bars_meta
        WHERE provider = ? AND interval = ?
        ORDER BY contract_symbol ASC
        """,
        ["alpaca", "1d"],
    )
    row = meta2[meta2["contract_symbol"] == "AAPL260221P00095000"].iloc[0]
    assert str(row["status"]) == "error"
    assert int(row["error_count"]) == 1
    assert "boom" in str(row["last_error"])


def test_duckdb_option_bars_meta_success_merges_ranges(tmp_path):
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)

    store = DuckDBOptionBarsStore(root_dir=tmp_path / "option_bars", warehouse=wh)

    sym = "AAPL260315C00100000"
    store.mark_meta_success(
        [sym],
        interval="1d",
        provider="alpaca",
        rows=10,
        start_ts=datetime(2026, 2, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 2, 2, tzinfo=timezone.utc),
    )
    store.mark_meta_success(
        [sym],
        interval="1d",
        provider="alpaca",
        rows=0,
        start_ts=None,
        end_ts=None,
    )
    store.mark_meta_success(
        [sym],
        interval="1d",
        provider="alpaca",
        rows=1,
        start_ts=datetime(2026, 1, 31, tzinfo=timezone.utc),
        end_ts=datetime(2026, 2, 3, tzinfo=timezone.utc),
    )

    meta = wh.fetch_df(
        """
        SELECT rows, start_ts, end_ts
        FROM option_bars_meta
        WHERE contract_symbol = ? AND interval = ? AND provider = ?
        """,
        [sym, "1d", "alpaca"],
    )
    assert len(meta) == 1
    row = meta.iloc[0].to_dict()
    assert int(row["rows"]) == 10
    assert row["start_ts"] == datetime(2026, 1, 31, 0, 0)
    assert row["end_ts"] == datetime(2026, 2, 3, 0, 0)
