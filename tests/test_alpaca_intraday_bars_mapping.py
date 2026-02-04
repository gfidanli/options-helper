from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from options_helper.data import alpaca_client
from options_helper.data.alpaca_client import AlpacaClient


class _StubTimeFrame:
    Minute = "1Min"

    def __init__(self, amount: int, unit: object) -> None:
        self.amount = amount
        self.unit = unit


class _StubTimeFrameUnit:
    Minute = "Minute"


class _StubBars:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


class _StubStockClient:
    def __init__(self, payload: _StubBars) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def get_stock_bars(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class _StubOptionClient:
    def __init__(self, payload: _StubBars) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def get_option_bars(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def _setup_timeframe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "TimeFrameUnit", _StubTimeFrameUnit)


def test_get_stock_bars_intraday_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_timeframe(monkeypatch)
    monkeypatch.setattr(alpaca_client, "_load_stock_bars_request", lambda: None)
    monkeypatch.setenv("OH_ALPACA_MARKET_TZ", "America/New_York")

    idx = pd.MultiIndex.from_product(
        [["AAPL"], pd.to_datetime(["2026-02-03T14:30:00Z", "2026-02-03T14:31:00Z"], utc=True)],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {
            "o": [10.0, 10.5],
            "h": [10.6, 10.8],
            "l": [9.9, 10.2],
            "c": [10.2, 10.7],
            "v": [100, 150],
        },
        index=idx,
    )
    stub = _StubStockClient(_StubBars(df))
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._stock_client = stub

    out = client.get_stock_bars_intraday("AAPL", day=date(2026, 2, 3), timeframe="1Min")

    assert len(stub.calls) == 1
    call = stub.calls[0]
    market_tz = ZoneInfo("America/New_York")
    start_local = datetime.combine(date(2026, 2, 3), datetime.min.time()).replace(tzinfo=market_tz)
    end_local = datetime.combine(date(2026, 2, 3), datetime.max.time()).replace(tzinfo=market_tz)
    assert call["start"] == start_local.astimezone(timezone.utc)
    assert call["end"] == end_local.astimezone(timezone.utc)
    assert list(out.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    assert out["volume"].iloc[0] == 100


def test_get_option_bars_intraday_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_timeframe(monkeypatch)
    monkeypatch.setattr(alpaca_client, "_load_option_bars_request", lambda: None)
    monkeypatch.setenv("OH_ALPACA_MARKET_TZ", "America/New_York")

    idx = pd.MultiIndex.from_product(
        [
            ["SPY260621C00100000", "SPY260621P00100000"],
            pd.to_datetime(["2026-02-03T14:30:00Z", "2026-02-03T14:31:00Z"], utc=True),
        ],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {
            "o": [1.0, 1.1, 2.0, 2.1],
            "h": [1.2, 1.3, 2.2, 2.3],
            "l": [0.9, 1.0, 1.9, 2.0],
            "c": [1.1, 1.2, 2.1, 2.2],
            "v": [10, 11, 20, 21],
        },
        index=idx,
    )
    stub = _StubOptionClient(_StubBars(df))
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._option_client = stub

    out = client.get_option_bars_intraday(
        ["SPY260621C00100000", "SPY260621P00100000"],
        day=date(2026, 2, 3),
        timeframe="1Min",
    )

    assert len(stub.calls) == 1
    call = stub.calls[0]
    market_tz = ZoneInfo("America/New_York")
    start_local = datetime.combine(date(2026, 2, 3), datetime.min.time()).replace(tzinfo=market_tz)
    end_local = datetime.combine(date(2026, 2, 3), datetime.max.time()).replace(tzinfo=market_tz)
    assert call["start"] == start_local.astimezone(timezone.utc)
    assert call["end"] == end_local.astimezone(timezone.utc)
    assert set(out["contractSymbol"]) == {"SPY260621C00100000", "SPY260621P00100000"}
    assert out["volume"].max() == 21
