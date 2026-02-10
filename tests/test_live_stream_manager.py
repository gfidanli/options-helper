from __future__ import annotations

import asyncio
import inspect
import threading
import time
from typing import Any

import pytest

import options_helper.data.streaming.live_manager as live_manager_module
from options_helper.data.streaming.live_manager import LiveStreamConfig, LiveStreamManager


def _invoke_handler(handler: Any, payload: Any) -> None:
    if handler is None:
        return
    result = handler(payload)
    if inspect.isawaitable(result):
        asyncio.run(result)


def _wait_until(predicate, *, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for condition.")


class _LifecycleStockStream:
    instances: list["_LifecycleStockStream"] = []

    def __init__(
        self,
        *,
        feed: str | None = None,
        on_bars: Any | None = None,
        on_quotes: Any | None = None,
        on_trades: Any | None = None,
    ) -> None:
        self.feed = feed
        self.on_bars = on_bars
        self.on_quotes = on_quotes
        self.on_trades = on_trades
        self.stop_event = threading.Event()
        self.subscriptions: list[str] = []
        self.run_calls = 0
        self.stop_calls = 0
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def subscribe(self, symbols: list[str]) -> None:
        self.subscriptions = list(symbols)

    def run(self) -> None:
        self.run_calls += 1
        symbol = self.subscriptions[0] if self.subscriptions else "AAPL"
        _invoke_handler(
            self.on_quotes,
            {
                "S": symbol,
                "t": "2026-02-10T15:00:00Z",
                "bp": 100.0,
                "ap": 100.2,
            },
        )
        _invoke_handler(
            self.on_trades,
            {
                "S": symbol,
                "t": "2026-02-10T15:00:01Z",
                "p": 100.1,
                "s": 10,
            },
        )
        self.stop_event.wait()

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_event.set()


class _LifecycleOptionStream:
    instances: list["_LifecycleOptionStream"] = []

    def __init__(
        self,
        *,
        feed: str | None = None,
        on_bars: Any | None = None,
        on_quotes: Any | None = None,
        on_trades: Any | None = None,
    ) -> None:
        self.feed = feed
        self.on_bars = on_bars
        self.on_quotes = on_quotes
        self.on_trades = on_trades
        self.stop_event = threading.Event()
        self.subscriptions: list[str] = []
        self.run_calls = 0
        self.stop_calls = 0
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def subscribe(self, symbols: list[str]) -> None:
        self.subscriptions = list(symbols)

    def run(self) -> None:
        self.run_calls += 1
        symbol = self.subscriptions[0] if self.subscriptions else "AAPL260320C00180000"
        _invoke_handler(
            self.on_quotes,
            {
                "S": symbol,
                "t": "2026-02-10T15:00:00Z",
                "bp": 2.0,
                "ap": 2.4,
            },
        )
        _invoke_handler(
            self.on_trades,
            {
                "S": symbol,
                "t": "2026-02-10T15:00:01Z",
                "p": 2.2,
                "s": 5,
            },
        )
        self.stop_event.wait()

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_event.set()


class _LifecycleTradingStream:
    instances: list["_LifecycleTradingStream"] = []

    def __init__(self, *, on_trade_updates: Any | None = None) -> None:
        self.on_trade_updates = on_trade_updates
        self.stop_event = threading.Event()
        self.subscribed = False
        self.run_calls = 0
        self.stop_calls = 0
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def subscribe_trade_updates(self) -> None:
        self.subscribed = True

    def run(self) -> None:
        self.run_calls += 1
        _invoke_handler(
            self.on_trade_updates,
            {
                "event": "fill",
                "timestamp": "2026-02-10T15:00:03Z",
                "order": {
                    "id": "order-1",
                    "symbol": "AAPL260320C00180000",
                    "side": "buy",
                    "qty": "1",
                    "filled_qty": "1",
                    "filled_avg_price": "2.2",
                    "status": "filled",
                },
            },
        )
        self.stop_event.wait()

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_event.set()


class _FlakyStockStream(_LifecycleStockStream):
    run_attempts = 0

    @classmethod
    def reset(cls) -> None:
        super().reset()
        cls.run_attempts = 0

    def run(self) -> None:
        type(self).run_attempts += 1
        if type(self).run_attempts == 1:
            raise RuntimeError("simulated stock stream failure")
        super().run()


class _SpamStockStream(_LifecycleStockStream):
    def run(self) -> None:
        self.run_calls += 1
        symbol = self.subscriptions[0] if self.subscriptions else "AAPL"
        for i in range(4000):
            _invoke_handler(
                self.on_quotes,
                {
                    "S": symbol,
                    "t": "2026-02-10T15:00:00Z",
                    "bp": 100.0 + (i * 0.001),
                    "ap": 100.2 + (i * 0.001),
                },
            )
        self.stop_event.wait()


def test_live_stream_manager_lifecycle_idempotent_start_and_restart() -> None:
    _LifecycleStockStream.reset()
    _LifecycleOptionStream.reset()
    _LifecycleTradingStream.reset()

    manager = LiveStreamManager(
        stock_streamer_factory=_LifecycleStockStream,
        option_streamer_factory=_LifecycleOptionStream,
        trading_streamer_factory=_LifecycleTradingStream,
    )

    config = LiveStreamConfig(
        stocks=["aapl"],
        option_contracts=["aapl260320c00180000"],
        stream_stocks=True,
        stream_options=True,
        stream_fills=True,
        queue_maxsize=64,
        reconnect_base_seconds=0.0,
        reconnect_cap_seconds=0.0,
    )
    manager.start(config)

    _wait_until(
        lambda: (
            "AAPL" in manager.snapshot().stock_quotes
            and "AAPL260320C00180000" in manager.snapshot().option_quotes
            and len(manager.snapshot().fills) >= 1
        )
    )
    snapshot = manager.snapshot()
    assert manager.is_running() is True
    assert snapshot.running is True
    assert snapshot.alive["stocks"] is True
    assert snapshot.alive["options"] is True
    assert snapshot.alive["fills"] is True
    assert snapshot.last_event_ts_by_stream["stocks"] is not None
    assert snapshot.last_event_ts_by_stream["options"] is not None
    assert snapshot.last_event_ts_by_stream["fills"] is not None

    stock_instances = len(_LifecycleStockStream.instances)
    option_instances = len(_LifecycleOptionStream.instances)
    fill_instances = len(_LifecycleTradingStream.instances)
    manager.start(config)
    assert len(_LifecycleStockStream.instances) == stock_instances
    assert len(_LifecycleOptionStream.instances) == option_instances
    assert len(_LifecycleTradingStream.instances) == fill_instances

    updated = LiveStreamConfig(
        stocks=["msft"],
        option_contracts=["aapl260320c00180000"],
        stream_stocks=True,
        stream_options=False,
        stream_fills=False,
        queue_maxsize=64,
        reconnect_base_seconds=0.0,
        reconnect_cap_seconds=0.0,
    )
    manager.start(updated)
    _wait_until(lambda: "MSFT" in manager.snapshot().stock_quotes)

    restarted = manager.snapshot()
    assert len(_LifecycleStockStream.instances) > stock_instances
    assert restarted.option_quotes == {}
    assert restarted.option_trades == {}
    assert restarted.fills == []
    assert restarted.alive["options"] is False
    assert restarted.alive["fills"] is False

    manager.stop()
    manager.stop()
    stopped = manager.snapshot()
    assert manager.is_running() is False
    assert stopped.running is False
    assert stopped.alive == {"stocks": False, "options": False, "fills": False}


def test_live_stream_manager_tracks_reconnect_attempts() -> None:
    _FlakyStockStream.reset()

    manager = LiveStreamManager(
        stock_streamer_factory=_FlakyStockStream,
        option_streamer_factory=_LifecycleOptionStream,
        trading_streamer_factory=_LifecycleTradingStream,
    )
    config = LiveStreamConfig(
        stocks=["AAPL"],
        option_contracts=[],
        stream_stocks=True,
        stream_options=False,
        stream_fills=False,
        max_reconnects=2,
        reconnect_base_seconds=0.0,
        reconnect_cap_seconds=0.0,
    )
    manager.start(config)

    _wait_until(lambda: manager.snapshot().reconnect_attempts["stocks"] >= 1)
    _wait_until(lambda: "AAPL" in manager.snapshot().stock_quotes)

    snapshot = manager.snapshot()
    assert _FlakyStockStream.run_attempts >= 2
    assert snapshot.reconnect_attempts["stocks"] >= 1
    assert snapshot.last_event_ts_by_stream["stocks"] is not None
    assert snapshot.errors_by_stream["stocks"] is None
    assert snapshot.alive["stocks"] is True

    manager.stop()


def test_live_stream_manager_drops_events_when_queue_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _SpamStockStream.reset()

    original_normalize_stock_quote = live_manager_module.normalize_stock_quote

    def _slow_normalize_stock_quote(payload: Any):  # type: ignore[no-untyped-def]
        time.sleep(0.001)
        return original_normalize_stock_quote(payload)

    monkeypatch.setattr(
        live_manager_module,
        "normalize_stock_quote",
        _slow_normalize_stock_quote,
    )

    manager = LiveStreamManager(
        stock_streamer_factory=_SpamStockStream,
        option_streamer_factory=_LifecycleOptionStream,
        trading_streamer_factory=_LifecycleTradingStream,
    )
    config = LiveStreamConfig(
        stocks=["AAPL"],
        option_contracts=[],
        stream_stocks=True,
        stream_options=False,
        stream_fills=False,
        queue_maxsize=1,
        queue_poll_seconds=0.2,
        reconnect_base_seconds=0.0,
        reconnect_cap_seconds=0.0,
    )
    manager.start(config)

    _wait_until(lambda: manager.snapshot().dropped_events > 0)
    snapshot = manager.snapshot()
    assert snapshot.dropped_events > 0
    assert snapshot.dropped_events_by_stream["stocks"] > 0
    assert snapshot.queue_depth <= 1

    manager.stop()
