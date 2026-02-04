from __future__ import annotations

import pytest

from options_helper.data.market_types import DataFetchError
from options_helper.data.streaming.alpaca_stream import AlpacaOptionStreamer, AlpacaStockStreamer


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.subscribed = {"bars": [], "trades": [], "quotes": []}
        self.unsubscribed = {"bars": [], "trades": [], "quotes": []}

    def subscribe_bars(self, handler, *symbols) -> None:
        self.subscribed["bars"].append((handler, symbols))

    def subscribe_trades(self, handler, *symbols) -> None:
        self.subscribed["trades"].append((handler, symbols))

    def subscribe_quotes(self, handler, *symbols) -> None:
        self.subscribed["quotes"].append((handler, symbols))

    def unsubscribe_bars(self, *symbols) -> None:
        self.unsubscribed["bars"].append(symbols)

    def unsubscribe_trades(self, *symbols) -> None:
        self.unsubscribed["trades"].append(symbols)

    def unsubscribe_quotes(self, *symbols) -> None:
        self.unsubscribed["quotes"].append(symbols)


class MissingQuotesStream:
    def subscribe_bars(self, handler, *symbols) -> None:
        pass

    def subscribe_trades(self, handler, *symbols) -> None:
        pass


class DummyStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _handler(event) -> None:
    return None


def test_stock_stream_subscribe_maps_symbols() -> None:
    stream = FakeStream()
    streamer = AlpacaStockStreamer(
        stream=stream,
        on_bars=_handler,
        on_trades=_handler,
        on_quotes=_handler,
    )
    mapped = streamer.subscribe(["brk-b", "AAPL", "aapl", ""])
    assert mapped == ["BRK.B", "AAPL"]
    for event in ("bars", "trades", "quotes"):
        assert stream.subscribed[event][0][1] == ("BRK.B", "AAPL")


def test_stock_stream_skips_missing_handlers() -> None:
    stream = FakeStream()
    streamer = AlpacaStockStreamer(stream=stream, on_trades=_handler)
    streamer.subscribe(["AAPL"])
    assert stream.subscribed["bars"] == []
    assert stream.subscribed["quotes"] == []
    assert stream.subscribed["trades"][0][1] == ("AAPL",)


def test_subscribe_raises_when_method_missing() -> None:
    streamer = AlpacaStockStreamer(stream=MissingQuotesStream(), on_quotes=_handler)
    with pytest.raises(DataFetchError, match="quotes"):
        streamer.subscribe(["AAPL"])


def test_missing_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", "/tmp/does-not-exist")
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(DataFetchError, match="Missing Alpaca credentials"):
        AlpacaOptionStreamer(stream_cls=DummyStream)


def test_stock_stream_coerces_feed_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", "/tmp/does-not-exist")
    monkeypatch.setenv("APCA_API_KEY_ID", "test")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test")

    from alpaca.data.live import StockDataStream

    streamer = AlpacaStockStreamer(stream_cls=StockDataStream, feed="sip")
    assert "/sip" in getattr(streamer.stream, "_endpoint", "")


def test_option_stream_coerces_feed_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", "/tmp/does-not-exist")
    monkeypatch.setenv("APCA_API_KEY_ID", "test")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test")

    from alpaca.data.live import OptionDataStream

    streamer = AlpacaOptionStreamer(stream_cls=OptionDataStream, feed="opra")
    assert "/opra" in getattr(streamer.stream, "_endpoint", "")
