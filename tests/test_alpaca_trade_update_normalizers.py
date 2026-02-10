from __future__ import annotations

from datetime import datetime, timezone

import pytest

from options_helper.data.market_types import DataFetchError
from options_helper.data.streaming.alpaca_trading_stream import AlpacaTradingStreamer
from options_helper.data.streaming.trading_normalizers import normalize_trade_update


class Obj:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTradingStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.subscribed: list = []
        self.run_calls = 0
        self.stop_calls = 0

    def subscribe_trade_updates(self, handler) -> None:  # type: ignore[no-untyped-def]
        self.subscribed.append(handler)

    def run(self) -> None:
        self.run_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class MissingSubscribeStream:
    def run(self) -> None:
        return None


def test_normalize_trade_update_from_dict_includes_optional_extras() -> None:
    payload = {
        "event": "fill",
        "timestamp": "2026-02-10T14:30:00Z",
        "order": {
            "id": "order-123",
            "symbol": "aapl",
            "side": "buy",
            "qty": "2",
            "filled_qty": "2",
            "filled_avg_price": "1.35",
            "status": "filled",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": "1.50",
            "stop_price": "1.00",
        },
    }
    normalized = normalize_trade_update(payload)
    assert normalized is not None
    assert normalized["event"] == "fill"
    assert normalized["order_id"] == "order-123"
    assert normalized["symbol"] == "AAPL"
    assert normalized["side"] == "buy"
    assert normalized["qty"] == 2.0
    assert normalized["filled_qty"] == 2.0
    assert normalized["filled_avg_price"] == 1.35
    assert normalized["status"] == "filled"
    assert normalized["type"] == "limit"
    assert normalized["tif"] == "day"
    assert normalized["limit_price"] == 1.5
    assert normalized["stop_price"] == 1.0


def test_normalize_trade_update_from_object_keeps_stable_base_keys() -> None:
    payload = Obj(
        event="partial_fill",
        at=datetime(2026, 2, 10, 14, 31),
        order=Obj(
            id="abc-1",
            symbol="SPY260320C00600000",
            side="sell",
            qty=5,
            filled_qty=2,
            filled_avg_price=0.95,
            status="partially_filled",
        ),
    )
    normalized = normalize_trade_update(payload)
    assert normalized is not None
    expected_keys = {
        "timestamp",
        "event",
        "order_id",
        "symbol",
        "side",
        "qty",
        "filled_qty",
        "filled_avg_price",
        "status",
    }
    assert set(normalized) == expected_keys
    assert normalized["timestamp"].tzinfo == timezone.utc
    assert normalized["event"] == "partial_fill"
    assert normalized["status"] == "partially_filled"


def test_normalize_trade_update_returns_none_for_empty_payload() -> None:
    assert normalize_trade_update({}) is None
    assert normalize_trade_update(Obj()) is None


def test_trading_stream_wrapper_uses_injected_stream_instance() -> None:
    stream = FakeTradingStream()
    updates: list[str] = []

    def _handler(event) -> None:  # type: ignore[no-untyped-def]
        updates.append(str(event))

    streamer = AlpacaTradingStreamer(stream=stream, on_trade_updates=_handler)
    streamer.subscribe_trade_updates()
    streamer.run()
    streamer.stop()
    assert stream.subscribed == [_handler]
    assert stream.run_calls == 1
    assert stream.stop_calls == 1


def test_trading_stream_wrapper_uses_injected_stream_cls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", "/tmp/does-not-exist")
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    streamer = AlpacaTradingStreamer(
        stream_cls=FakeTradingStream,
        on_trade_updates=lambda _: None,
    )
    assert streamer.stream.kwargs.get("paper") is True
    assert streamer.stream.kwargs.get("api_key") == "key"
    assert streamer.stream.kwargs.get("secret_key") == "secret"
    assert streamer.stream.kwargs.get("url_override") is None


def test_trading_stream_wrapper_uses_ws_url_override_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", "/tmp/does-not-exist")
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "wss://paper-api.alpaca.markets/stream")

    streamer = AlpacaTradingStreamer(
        stream_cls=FakeTradingStream,
        on_trade_updates=lambda _: None,
    )
    assert streamer.stream.kwargs.get("paper") is True
    assert streamer.stream.kwargs.get("url_override") == "wss://paper-api.alpaca.markets/stream"


def test_trading_stream_wrapper_infers_live_from_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", "/tmp/does-not-exist")
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")

    streamer = AlpacaTradingStreamer(
        stream_cls=FakeTradingStream,
        on_trade_updates=lambda _: None,
    )
    assert streamer.stream.kwargs.get("paper") is False
    assert streamer.stream.kwargs.get("url_override") is None


def test_trading_stream_wrapper_requires_handler_for_subscription() -> None:
    streamer = AlpacaTradingStreamer(stream=FakeTradingStream())
    with pytest.raises(DataFetchError, match="on_trade_updates"):
        streamer.subscribe_trade_updates()


def test_trading_stream_wrapper_raises_if_subscribe_method_missing() -> None:
    streamer = AlpacaTradingStreamer(
        stream=MissingSubscribeStream(),
        on_trade_updates=lambda _: None,
    )
    with pytest.raises(DataFetchError, match="trade update subscriptions"):
        streamer.subscribe_trade_updates()
