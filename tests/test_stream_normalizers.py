from __future__ import annotations

from datetime import datetime, timezone

from options_helper.data.streaming.normalizers import (
    normalize_option_quote,
    normalize_option_trade,
    normalize_stock_bar,
)


class Obj:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_normalize_stock_bar_from_dict() -> None:
    ts = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
    event = {
        "S": "BRK.B",
        "t": int(ts.timestamp() * 1e9),
        "o": 1.0,
        "h": 2.0,
        "l": 0.5,
        "c": 1.5,
        "v": 1200,
        "n": 12,
        "vw": 1.4,
    }
    normalized = normalize_stock_bar(event)
    assert normalized is not None
    assert normalized.dataset == "stock_bars"
    assert normalized.symbol == "BRK-B"
    row = normalized.row
    assert row["timestamp"].timestamp() == ts.timestamp()
    assert row["open"] == 1.0
    assert row["trade_count"] == 12


def test_normalize_option_quote_from_object() -> None:
    event = Obj(
        symbol="AAPL260320C00150000",
        timestamp="2026-02-03T14:31:00Z",
        bp=1.2,
        ap=1.4,
        bs=10,
        bx="Q",
    )
    setattr(event, "as", 12)
    normalized = normalize_option_quote(event)
    assert normalized is not None
    assert normalized.dataset == "option_quotes"
    assert normalized.symbol == "AAPL260320C00150000"
    row = normalized.row
    assert row["bid_price"] == 1.2
    assert row["ask_price"] == 1.4
    assert row["exchange"] == "Q"


def test_normalize_trade_conditions_list() -> None:
    event = Obj(
        symbol="AAPL260320C00150000",
        timestamp="2026-02-03T14:32:00Z",
        p=2.5,
        s=4,
        x="P",
        c=[1, "A"],
    )
    normalized = normalize_option_trade(event)
    assert normalized is not None
    assert normalized.row["conditions"] == "1,A"


def test_normalize_missing_symbol_or_timestamp_returns_none() -> None:
    assert normalize_stock_bar({"t": 123}) is None
    assert normalize_stock_bar({"S": "AAPL"}) is None
