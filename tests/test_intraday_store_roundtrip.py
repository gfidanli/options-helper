from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.intraday_store import IntradayStore


def test_intraday_store_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path)
    day = date(2026, 2, 3)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-02-03T14:30:00Z", "2026-02-03T15:30:00Z"], utc=True
            ),
            "open": [10.0, 10.5],
            "close": [10.2, 10.7],
        }
    )

    out_path = store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "AAPL",
        day,
        df,
        meta={"provider": "alpaca"},
    )

    assert out_path.exists()
    loaded = store.load_partition("stocks", "bars", "1Min", "AAPL", day)
    assert not loaded.empty
    assert list(loaded.columns) == ["timestamp", "open", "close"]

    meta = store.load_meta("stocks", "bars", "1Min", "AAPL", day)
    assert meta["rows"] == 2
    assert meta["provider"] == "alpaca"
    assert meta["symbol"] == "AAPL"
    assert meta["coverage_start"].startswith("2026-02-03")


def test_intraday_store_missing_partition(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path)
    day = date(2026, 2, 1)

    loaded = store.load_partition("stocks", "bars", "1Min", "AAPL", day)
    assert loaded.empty

    meta = store.load_meta("stocks", "bars", "1Min", "AAPL", day)
    assert meta == {}
