from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.data import alpaca_client
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.market_types import DataFetchError
from options_helper.data.providers.alpaca import AlpacaProvider


class _StubTimeFrame:
    Day = "1Day"
    Hour = "1Hour"
    Minute = "1Minute"


class _StubBars:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


class _StubStockClient:
    def __init__(self, bars: _StubBars) -> None:
        self._bars = bars
        self.last_args = ()
        self.last_kwargs: dict[str, object] = {}

    def get_stock_bars(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return self._bars


class _StubSnapshotClient:
    def __init__(self, snapshot=None, trade=None) -> None:
        self.snapshot = snapshot
        self.trade = trade
        self.last_symbol: str | None = None

    def get_stock_snapshot(self, symbol: str):
        self.last_symbol = symbol
        return self.snapshot

    def get_stock_latest_trade(self, symbol: str):
        self.last_symbol = symbol
        return self.trade


class _StubAlpacaClient:
    def __init__(self, bars_df: pd.DataFrame, snapshot=None, trade=None) -> None:
        self.stock_client = _StubSnapshotClient(snapshot=snapshot, trade=trade)
        self._bars_df = bars_df
        self.last_bars_kwargs: dict[str, object] = {}

    def get_stock_bars(
        self,
        symbol: str,
        *,
        start,
        end,
        interval: str,
        adjustment: str,
    ) -> pd.DataFrame:
        self.last_bars_kwargs = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "interval": interval,
            "adjustment": adjustment,
        }
        return self._bars_df.copy()


def _make_bars_df(symbol: str) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            [symbol],
            pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True),
        ],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        },
        index=idx,
    )


def test_get_stock_bars_normalizes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_bars_df("SPY")
    stub = _StubStockClient(_StubBars(df))
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._stock_client = stub

    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "_load_stock_bars_request", lambda: None)

    out = client.get_stock_bars(
        "SPY",
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        interval="1d",
        adjustment="raw",
    )

    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in out.columns
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None


def test_get_stock_bars_passes_timeframe_and_adjustment(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_bars_df("BRK.B")
    stub = _StubStockClient(_StubBars(df))
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._stock_client = stub

    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "_load_stock_bars_request", lambda: None)

    client.get_stock_bars(
        "BRK-B",
        start=date(2026, 1, 1),
        end=date(2026, 1, 10),
        interval="1d",
        adjustment="all",
    )

    assert stub.last_kwargs["symbol_or_symbols"] == "BRK.B"
    assert stub.last_kwargs["timeframe"] == _StubTimeFrame.Day
    assert stub.last_kwargs["adjustment"] == "all"
    assert stub.last_kwargs["start"].tzinfo is not None
    assert stub.last_kwargs["end"].tzinfo is not None


def test_get_stock_bars_rejects_unsupported_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_bars_df("SPY")
    stub = _StubStockClient(_StubBars(df))
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._stock_client = stub

    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "_load_stock_bars_request", lambda: None)

    with pytest.raises(DataFetchError):
        client.get_stock_bars(
            "SPY",
            start=date(2026, 1, 1),
            end=date(2026, 1, 10),
            interval="2d",
            adjustment="raw",
        )


def test_alpaca_provider_history_adjustment_mapping() -> None:
    idx = pd.to_datetime(["2026-01-02", "2026-01-03"])
    bars = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0], "Volume": [10, 20]},
        index=idx,
    )
    stub = _StubAlpacaClient(bars)
    provider = AlpacaProvider(client=stub)

    out = provider.get_history(
        "SPY",
        start=date(2026, 1, 1),
        end=None,
        interval="1d",
        auto_adjust=True,
        back_adjust=False,
    )

    assert stub.last_bars_kwargs["adjustment"] == "all"
    assert not out.empty

    provider.get_history(
        "SPY",
        start=date(2026, 1, 1),
        end=None,
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
    )
    assert stub.last_bars_kwargs["adjustment"] == "raw"


def test_alpaca_provider_quote_uses_snapshot_price() -> None:
    bars = pd.DataFrame({"Close": [99.0]}, index=pd.to_datetime(["2026-01-02"]))
    snapshot = {"latest_trade": {"price": 123.45}}
    stub = _StubAlpacaClient(bars, snapshot=snapshot)
    provider = AlpacaProvider(client=stub)

    price = provider.get_quote("BRK-B")

    assert price == 123.45
    assert stub.stock_client.last_symbol == "BRK.B"


def test_alpaca_provider_quote_falls_back_to_bars() -> None:
    bars = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.to_datetime(["2026-01-02", "2026-01-03"]))
    stub = _StubAlpacaClient(bars, snapshot=None, trade=None)
    provider = AlpacaProvider(client=stub)

    price = provider.get_quote("SPY")

    assert price == 101.0


def test_alpaca_provider_underlying_uses_history() -> None:
    bars = pd.DataFrame({"Close": [200.0, 201.0]}, index=pd.to_datetime(["2026-01-02", "2026-01-03"]))
    stub = _StubAlpacaClient(bars)
    provider = AlpacaProvider(client=stub)

    underlying = provider.get_underlying("brk.b", period="5d", interval="1d")

    assert underlying.symbol == "BRK-B"
    assert underlying.last_price == 201.0
    assert not underlying.history.empty
    assert stub.last_bars_kwargs["adjustment"] == "all"
