from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.streaming.intraday_writer import BufferedIntradayWriter, PartitionSpec


def test_intraday_writer_appends_and_dedupes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path)
    spec = PartitionSpec(kind="stocks", dataset="quotes", timeframe="tick", symbol="AAPL", day=date(2026, 2, 3))
    writer = BufferedIntradayWriter(store, spec, meta={"provider": "alpaca"}, dedupe_on=("timestamp",))

    writer.add({"timestamp": "2026-02-03T14:30:00Z", "bid_price": 1.0})
    first = writer.flush()
    assert first is not None and first.exists()

    writer.add({"timestamp": "2026-02-03T14:30:00Z", "bid_price": 2.0})
    writer.add({"timestamp": datetime(2026, 2, 3, 14, 31, tzinfo=timezone.utc), "bid_price": 3.0})
    writer.flush()

    loaded = store.load_partition("stocks", "quotes", "tick", "AAPL", date(2026, 2, 3))
    assert not loaded.empty
    # Timestamp dedupe should keep the latest bid_price=2.0 for 14:30.
    loaded["bid_price"] = pd.to_numeric(loaded["bid_price"], errors="coerce")
    row = loaded[loaded["timestamp"].astype(str).str.startswith("2026-02-03T14:30:00")].iloc[0]
    assert float(row["bid_price"]) == 2.0

    meta = store.load_meta("stocks", "quotes", "tick", "AAPL", date(2026, 2, 3))
    assert meta["provider"] == "alpaca"
    assert meta["flushes"] == 2


def test_intraday_writer_injects_contract_symbol(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path)
    contract = "AAPL260320C00150000"
    spec = PartitionSpec(
        kind="options",
        dataset="trades",
        timeframe="tick",
        symbol=contract,
        day=date(2026, 2, 3),
    )
    writer = BufferedIntradayWriter(store, spec)
    writer.add({"timestamp": "2026-02-03T14:30:00Z", "price": 1.25, "size": 2})
    writer.flush()

    loaded = store.load_partition("options", "trades", "tick", contract, date(2026, 2, 3))
    assert "contractSymbol" in loaded.columns
    assert loaded["contractSymbol"].iloc[0] == contract
