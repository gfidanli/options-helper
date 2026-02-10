from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from options_helper.data.market_types import DataFetchError
from options_helper.data.streaming.alpaca_stream import AlpacaOptionStreamer, AlpacaStockStreamer
from options_helper.data.streaming.alpaca_trading_stream import AlpacaTradingStreamer
from options_helper.data.streaming.normalizers import (
    normalize_option_quote,
    normalize_option_trade,
    normalize_stock_quote,
    normalize_stock_trade,
)
from options_helper.data.streaming.runner import compute_backoff_seconds
from options_helper.data.streaming.trading_normalizers import normalize_trade_update

logger = logging.getLogger(__name__)

_STREAM_KEYS = ("stocks", "options", "fills")
_EVENT_STOCK_QUOTE = "stock_quote"
_EVENT_STOCK_TRADE = "stock_trade"
_EVENT_OPTION_QUOTE = "option_quote"
_EVENT_OPTION_TRADE = "option_trade"
_EVENT_FILL_UPDATE = "fill_update"


def _clean_token(value: str | None) -> str | None:
    token = (value or "").strip()
    return token or None


def _normalize_symbols(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _stream_bool_map(default: bool) -> dict[str, bool]:
    return {key: default for key in _STREAM_KEYS}


def _stream_count_map() -> dict[str, int]:
    return {key: 0 for key in _STREAM_KEYS}


def _stream_error_map() -> dict[str, str | None]:
    return {key: None for key in _STREAM_KEYS}


def _stream_ts_map() -> dict[str, datetime | None]:
    return {key: None for key in _STREAM_KEYS}


@dataclass(frozen=True)
class LiveStreamConfig:
    stocks: list[str] = field(default_factory=list)
    option_contracts: list[str] = field(default_factory=list)
    stream_stocks: bool = True
    stream_options: bool = True
    stream_fills: bool = True
    stock_feed: str | None = None
    options_feed: str | None = None
    max_reconnects: int = 5
    reconnect_base_seconds: float = 0.5
    reconnect_cap_seconds: float = 30.0
    queue_maxsize: int = 2048
    queue_poll_seconds: float = 0.1
    fills_cache_size: int = 200

    def normalized(self) -> LiveStreamConfig:
        stream_stocks = bool(self.stream_stocks)
        stream_options = bool(self.stream_options)
        stream_fills = bool(self.stream_fills)

        stocks = _normalize_symbols(self.stocks) if stream_stocks else []
        option_contracts = (
            _normalize_symbols(self.option_contracts) if stream_options else []
        )

        max_reconnects = max(0, int(self.max_reconnects))
        reconnect_base = max(0.0, float(self.reconnect_base_seconds))
        reconnect_cap = max(0.0, float(self.reconnect_cap_seconds))
        if reconnect_cap < reconnect_base:
            reconnect_cap = reconnect_base

        queue_maxsize = max(1, int(self.queue_maxsize))
        queue_poll_seconds = max(0.01, float(self.queue_poll_seconds))
        fills_cache_size = max(1, int(self.fills_cache_size))

        return LiveStreamConfig(
            stocks=stocks,
            option_contracts=option_contracts,
            stream_stocks=stream_stocks,
            stream_options=stream_options,
            stream_fills=stream_fills,
            stock_feed=_clean_token(self.stock_feed),
            options_feed=_clean_token(self.options_feed),
            max_reconnects=max_reconnects,
            reconnect_base_seconds=reconnect_base,
            reconnect_cap_seconds=reconnect_cap,
            queue_maxsize=queue_maxsize,
            queue_poll_seconds=queue_poll_seconds,
            fills_cache_size=fills_cache_size,
        )


@dataclass(frozen=True)
class LiveSnapshot:
    as_of: datetime | None = None
    stock_quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    stock_trades: dict[str, dict[str, Any]] = field(default_factory=dict)
    option_quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    option_trades: dict[str, dict[str, Any]] = field(default_factory=dict)
    fills: list[dict[str, Any]] = field(default_factory=list)
    alive: dict[str, bool] = field(default_factory=lambda: _stream_bool_map(False))
    reconnect_attempts: dict[str, int] = field(default_factory=_stream_count_map)
    last_event_ts_by_stream: dict[str, datetime | None] = field(default_factory=_stream_ts_map)
    queue_depth: int = 0
    dropped_events: int = 0
    dropped_events_by_stream: dict[str, int] = field(default_factory=_stream_count_map)
    last_error: str | None = None
    errors_by_stream: dict[str, str | None] = field(default_factory=_stream_error_map)
    running: bool = False


@dataclass(frozen=True)
class _QueuedEvent:
    stream_key: str
    event_kind: str
    payload: Any


class _StreamWorker:
    def __init__(
        self,
        *,
        stream_key: str,
        create_streamer: Callable[[], Any],
        subscribe: Callable[[Any], None],
        stop_event: threading.Event,
        max_reconnects: int,
        reconnect_base_seconds: float,
        reconnect_cap_seconds: float,
        on_connected: Callable[[str], None],
        on_alive: Callable[[str, bool], None],
        on_reconnect: Callable[[str, Exception], None],
        on_error: Callable[[str, Exception], None],
        backoff_fn: Callable[..., float],
    ) -> None:
        self._stream_key = stream_key
        self._create_streamer = create_streamer
        self._subscribe = subscribe
        self._stop_event = stop_event
        self._max_reconnects = max_reconnects
        self._reconnect_base_seconds = reconnect_base_seconds
        self._reconnect_cap_seconds = reconnect_cap_seconds
        self._on_connected = on_connected
        self._on_alive = on_alive
        self._on_reconnect = on_reconnect
        self._on_error = on_error
        self._backoff_fn = backoff_fn
        self._thread: threading.Thread | None = None
        self._active_stream_lock = threading.Lock()
        self._active_streamer: Any | None = None

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"live-stream-{self._stream_key}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._active_stream_lock:
            active = self._active_streamer
        if active is not None:
            try:
                if hasattr(active, "stop"):
                    active.stop()
            except Exception:  # noqa: BLE001
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            streamer = None
            try:
                streamer = self._create_streamer()
                with self._active_stream_lock:
                    self._active_streamer = streamer
                self._subscribe(streamer)
                self._on_connected(self._stream_key)
                self._on_alive(self._stream_key, True)
                consecutive_failures = 0
                if hasattr(streamer, "run"):
                    streamer.run()
                else:
                    raise DataFetchError(f"{self._stream_key} streamer does not implement run().")

                if self._stop_event.is_set():
                    return
                raise DataFetchError(f"{self._stream_key} stream stopped unexpectedly.")
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    return
                consecutive_failures += 1
                if consecutive_failures > self._max_reconnects:
                    self._on_error(self._stream_key, exc)
                    return
                self._on_reconnect(self._stream_key, exc)
                delay = float(
                    self._backoff_fn(
                        consecutive_failures,
                        base_seconds=self._reconnect_base_seconds,
                        cap_seconds=self._reconnect_cap_seconds,
                    )
                )
                if delay < 0.0:
                    delay = 0.0
                logger.warning(
                    "%s stream error (%s). reconnecting in %.2fs (%s/%s).",
                    self._stream_key,
                    exc,
                    delay,
                    consecutive_failures,
                    self._max_reconnects,
                )
                if self._stop_event.wait(delay):
                    return
            finally:
                self._on_alive(self._stream_key, False)
                with self._active_stream_lock:
                    active = self._active_streamer
                    self._active_streamer = None
                if active is not None:
                    try:
                        if hasattr(active, "stop"):
                            active.stop()
                    except Exception:  # noqa: BLE001
                        pass


class LiveStreamManager:
    def __init__(
        self,
        *,
        stock_streamer_factory: Callable[..., Any] = AlpacaStockStreamer,
        option_streamer_factory: Callable[..., Any] = AlpacaOptionStreamer,
        trading_streamer_factory: Callable[..., Any] = AlpacaTradingStreamer,
        backoff_fn: Callable[..., float] = compute_backoff_seconds,
    ) -> None:
        self._stock_streamer_factory = stock_streamer_factory
        self._option_streamer_factory = option_streamer_factory
        self._trading_streamer_factory = trading_streamer_factory
        self._backoff_fn = backoff_fn

        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._running = False
        self._run_id = 0
        self._active_config: LiveStreamConfig | None = None
        self._queue: queue.Queue[_QueuedEvent] | None = None
        self._stop_event: threading.Event | None = None
        self._consumer_thread: threading.Thread | None = None
        self._workers: dict[str, _StreamWorker] = {}

        self._as_of: datetime | None = None
        self._stock_quotes: dict[str, dict[str, Any]] = {}
        self._stock_trades: dict[str, dict[str, Any]] = {}
        self._option_quotes: dict[str, dict[str, Any]] = {}
        self._option_trades: dict[str, dict[str, Any]] = {}
        self._fills: deque[dict[str, Any]] = deque(maxlen=200)

        self._alive: dict[str, bool] = _stream_bool_map(False)
        self._reconnect_attempts: dict[str, int] = _stream_count_map()
        self._last_event_ts_by_stream: dict[str, datetime | None] = _stream_ts_map()
        self._dropped_events = 0
        self._dropped_events_by_stream: dict[str, int] = _stream_count_map()
        self._errors_by_stream: dict[str, str | None] = _stream_error_map()
        self._last_error: str | None = None

    def start(self, config: LiveStreamConfig) -> None:
        desired = config.normalized()
        with self._lifecycle_lock:
            if self._running and self._active_config == desired:
                return
            if self._running:
                self._stop_locked()
            self._start_locked(desired)

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_locked()

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    def snapshot(self) -> LiveSnapshot:
        with self._state_lock:
            queue_depth = self._queue.qsize() if self._queue is not None else 0
            return LiveSnapshot(
                as_of=self._as_of,
                stock_quotes={key: dict(row) for key, row in self._stock_quotes.items()},
                stock_trades={key: dict(row) for key, row in self._stock_trades.items()},
                option_quotes={key: dict(row) for key, row in self._option_quotes.items()},
                option_trades={key: dict(row) for key, row in self._option_trades.items()},
                fills=[dict(row) for row in self._fills],
                alive=dict(self._alive),
                reconnect_attempts=dict(self._reconnect_attempts),
                last_event_ts_by_stream=dict(self._last_event_ts_by_stream),
                queue_depth=queue_depth,
                dropped_events=self._dropped_events,
                dropped_events_by_stream=dict(self._dropped_events_by_stream),
                last_error=self._last_error,
                errors_by_stream=dict(self._errors_by_stream),
                running=self._running,
            )

    def _start_locked(self, config: LiveStreamConfig) -> None:
        stop_event = threading.Event()
        event_queue: queue.Queue[_QueuedEvent] = queue.Queue(maxsize=config.queue_maxsize)

        with self._state_lock:
            self._run_id += 1
            run_id = self._run_id

        workers = self._build_workers(
            config=config,
            run_id=run_id,
            stop_event=stop_event,
        )
        if not workers:
            raise DataFetchError("LiveStreamConfig enables no active streams.")

        consumer_thread = threading.Thread(
            target=self._consumer_loop,
            args=(run_id, stop_event, event_queue, config.queue_poll_seconds),
            name="live-stream-consumer",
            daemon=True,
        )

        with self._state_lock:
            self._running = True
            self._active_config = config
            self._queue = event_queue
            self._stop_event = stop_event
            self._consumer_thread = consumer_thread
            self._workers = workers

            self._as_of = None
            self._stock_quotes = {}
            self._stock_trades = {}
            self._option_quotes = {}
            self._option_trades = {}
            self._fills = deque(maxlen=config.fills_cache_size)

            self._alive = _stream_bool_map(False)
            self._reconnect_attempts = _stream_count_map()
            self._last_event_ts_by_stream = _stream_ts_map()
            self._dropped_events = 0
            self._dropped_events_by_stream = _stream_count_map()
            self._errors_by_stream = _stream_error_map()
            self._last_error = None

        consumer_thread.start()
        for worker in workers.values():
            worker.start()

    def _stop_locked(self) -> None:
        with self._state_lock:
            if not self._running:
                return
            stop_event = self._stop_event
            consumer_thread = self._consumer_thread
            workers = list(self._workers.values())

            self._running = False
            self._active_config = None
            self._queue = None
            self._stop_event = None
            self._consumer_thread = None
            self._workers = {}
            self._run_id += 1
            for key in _STREAM_KEYS:
                self._alive[key] = False

        if stop_event is not None:
            stop_event.set()
        for worker in workers:
            worker.stop(timeout=5.0)
        if consumer_thread is not None:
            consumer_thread.join(timeout=5.0)

    def _build_workers(
        self,
        *,
        config: LiveStreamConfig,
        run_id: int,
        stop_event: threading.Event,
    ) -> dict[str, _StreamWorker]:
        workers: dict[str, _StreamWorker] = {}

        if config.stream_stocks and config.stocks:

            async def _on_stock_quote(payload: Any) -> None:
                self._enqueue_event(
                    run_id=run_id,
                    stream_key="stocks",
                    event_kind=_EVENT_STOCK_QUOTE,
                    payload=payload,
                )

            async def _on_stock_trade(payload: Any) -> None:
                self._enqueue_event(
                    run_id=run_id,
                    stream_key="stocks",
                    event_kind=_EVENT_STOCK_TRADE,
                    payload=payload,
                )

            def _create_stock_streamer() -> Any:
                return self._stock_streamer_factory(
                    feed=config.stock_feed,
                    on_quotes=_on_stock_quote,
                    on_trades=_on_stock_trade,
                    on_bars=None,
                )

            def _subscribe_stock(streamer: Any) -> None:
                streamer.subscribe(config.stocks)

            workers["stocks"] = self._new_worker(
                run_id=run_id,
                stream_key="stocks",
                create_streamer=_create_stock_streamer,
                subscribe=_subscribe_stock,
                stop_event=stop_event,
                config=config,
            )

        if config.stream_options and config.option_contracts:

            async def _on_option_quote(payload: Any) -> None:
                self._enqueue_event(
                    run_id=run_id,
                    stream_key="options",
                    event_kind=_EVENT_OPTION_QUOTE,
                    payload=payload,
                )

            async def _on_option_trade(payload: Any) -> None:
                self._enqueue_event(
                    run_id=run_id,
                    stream_key="options",
                    event_kind=_EVENT_OPTION_TRADE,
                    payload=payload,
                )

            def _create_option_streamer() -> Any:
                return self._option_streamer_factory(
                    feed=config.options_feed,
                    on_quotes=_on_option_quote,
                    on_trades=_on_option_trade,
                    on_bars=None,
                )

            def _subscribe_options(streamer: Any) -> None:
                streamer.subscribe(config.option_contracts)

            workers["options"] = self._new_worker(
                run_id=run_id,
                stream_key="options",
                create_streamer=_create_option_streamer,
                subscribe=_subscribe_options,
                stop_event=stop_event,
                config=config,
            )

        if config.stream_fills:

            async def _on_fill(payload: Any) -> None:
                self._enqueue_event(
                    run_id=run_id,
                    stream_key="fills",
                    event_kind=_EVENT_FILL_UPDATE,
                    payload=payload,
                )

            def _create_fill_streamer() -> Any:
                return self._trading_streamer_factory(on_trade_updates=_on_fill)

            def _subscribe_fills(streamer: Any) -> None:
                streamer.subscribe_trade_updates()

            workers["fills"] = self._new_worker(
                run_id=run_id,
                stream_key="fills",
                create_streamer=_create_fill_streamer,
                subscribe=_subscribe_fills,
                stop_event=stop_event,
                config=config,
            )

        return workers

    def _new_worker(
        self,
        *,
        run_id: int,
        stream_key: str,
        create_streamer: Callable[[], Any],
        subscribe: Callable[[Any], None],
        stop_event: threading.Event,
        config: LiveStreamConfig,
    ) -> _StreamWorker:
        return _StreamWorker(
            stream_key=stream_key,
            create_streamer=create_streamer,
            subscribe=subscribe,
            stop_event=stop_event,
            max_reconnects=config.max_reconnects,
            reconnect_base_seconds=config.reconnect_base_seconds,
            reconnect_cap_seconds=config.reconnect_cap_seconds,
            on_connected=lambda name: self._on_stream_connected(run_id=run_id, stream_key=name),
            on_alive=lambda name, alive: self._set_stream_alive(
                run_id=run_id, stream_key=name, alive=alive
            ),
            on_reconnect=lambda name, exc: self._record_reconnect(
                run_id=run_id, stream_key=name, exc=exc
            ),
            on_error=lambda name, exc: self._record_stream_error(
                run_id=run_id, stream_key=name, exc=exc
            ),
            backoff_fn=self._backoff_fn,
        )

    def _enqueue_event(
        self,
        *,
        run_id: int,
        stream_key: str,
        event_kind: str,
        payload: Any,
    ) -> None:
        with self._state_lock:
            if run_id != self._run_id:
                return
            event_queue = self._queue
        if event_queue is None:
            return
        try:
            event_queue.put_nowait(
                _QueuedEvent(
                    stream_key=stream_key,
                    event_kind=event_kind,
                    payload=payload,
                )
            )
        except queue.Full:
            with self._state_lock:
                if run_id != self._run_id:
                    return
                self._dropped_events += 1
                self._dropped_events_by_stream[stream_key] += 1
        except Exception as exc:  # noqa: BLE001
            with self._state_lock:
                if run_id != self._run_id:
                    return
                self._last_error = str(exc)
                self._errors_by_stream[stream_key] = str(exc)

    def _consumer_loop(
        self,
        run_id: int,
        stop_event: threading.Event,
        event_queue: queue.Queue[_QueuedEvent],
        queue_poll_seconds: float,
    ) -> None:
        timeout = max(0.01, float(queue_poll_seconds))
        while True:
            if stop_event.is_set():
                try:
                    queued = event_queue.get_nowait()
                except queue.Empty:
                    break
            else:
                try:
                    queued = event_queue.get(timeout=timeout)
                except queue.Empty:
                    continue
            self._consume_event(run_id=run_id, queued=queued)

    def _consume_event(self, *, run_id: int, queued: _QueuedEvent) -> None:
        try:
            if queued.event_kind == _EVENT_STOCK_QUOTE:
                normalized = normalize_stock_quote(queued.payload)
                if normalized is None:
                    return
                self._update_symbol_cache(
                    run_id=run_id,
                    stream_key="stocks",
                    cache=self._stock_quotes,
                    symbol=normalized.symbol,
                    row=normalized.row,
                )
                return
            if queued.event_kind == _EVENT_STOCK_TRADE:
                normalized = normalize_stock_trade(queued.payload)
                if normalized is None:
                    return
                self._update_symbol_cache(
                    run_id=run_id,
                    stream_key="stocks",
                    cache=self._stock_trades,
                    symbol=normalized.symbol,
                    row=normalized.row,
                )
                return
            if queued.event_kind == _EVENT_OPTION_QUOTE:
                normalized = normalize_option_quote(queued.payload)
                if normalized is None:
                    return
                self._update_symbol_cache(
                    run_id=run_id,
                    stream_key="options",
                    cache=self._option_quotes,
                    symbol=normalized.symbol,
                    row=normalized.row,
                )
                return
            if queued.event_kind == _EVENT_OPTION_TRADE:
                normalized = normalize_option_trade(queued.payload)
                if normalized is None:
                    return
                self._update_symbol_cache(
                    run_id=run_id,
                    stream_key="options",
                    cache=self._option_trades,
                    symbol=normalized.symbol,
                    row=normalized.row,
                )
                return
            if queued.event_kind == _EVENT_FILL_UPDATE:
                row = normalize_trade_update(queued.payload)
                if row is None:
                    return
                self._update_fills(run_id=run_id, stream_key="fills", row=row)
                return
        except Exception as exc:  # noqa: BLE001
            self._record_stream_error(run_id=run_id, stream_key=queued.stream_key, exc=exc)

    def _update_symbol_cache(
        self,
        *,
        run_id: int,
        stream_key: str,
        cache: dict[str, dict[str, Any]],
        symbol: str,
        row: dict[str, Any],
    ) -> None:
        timestamp = _coerce_timestamp(row.get("timestamp"))
        with self._state_lock:
            if run_id != self._run_id:
                return
            cache[symbol] = dict(row)
            self._record_event_ts_locked(stream_key=stream_key, timestamp=timestamp)

    def _update_fills(self, *, run_id: int, stream_key: str, row: dict[str, Any]) -> None:
        timestamp = _coerce_timestamp(row.get("timestamp"))
        with self._state_lock:
            if run_id != self._run_id:
                return
            self._fills.append(dict(row))
            self._record_event_ts_locked(stream_key=stream_key, timestamp=timestamp)

    def _record_event_ts_locked(self, *, stream_key: str, timestamp: datetime | None) -> None:
        event_ts = timestamp or datetime.now(timezone.utc)
        current = self._last_event_ts_by_stream.get(stream_key)
        if current is None or event_ts > current:
            self._last_event_ts_by_stream[stream_key] = event_ts
        if self._as_of is None or event_ts > self._as_of:
            self._as_of = event_ts

    def _on_stream_connected(self, *, run_id: int, stream_key: str) -> None:
        with self._state_lock:
            if run_id != self._run_id:
                return
            self._errors_by_stream[stream_key] = None

    def _set_stream_alive(self, *, run_id: int, stream_key: str, alive: bool) -> None:
        with self._state_lock:
            if run_id != self._run_id:
                return
            self._alive[stream_key] = bool(alive)

    def _record_reconnect(self, *, run_id: int, stream_key: str, exc: Exception) -> None:
        message = str(exc)
        with self._state_lock:
            if run_id != self._run_id:
                return
            self._reconnect_attempts[stream_key] += 1
            self._errors_by_stream[stream_key] = message
            self._last_error = message

    def _record_stream_error(self, *, run_id: int, stream_key: str, exc: Exception) -> None:
        message = str(exc)
        with self._state_lock:
            if run_id != self._run_id:
                return
            self._errors_by_stream[stream_key] = message
            self._last_error = message


__all__ = ["LiveSnapshot", "LiveStreamConfig", "LiveStreamManager"]
