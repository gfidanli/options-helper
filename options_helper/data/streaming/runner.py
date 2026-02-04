from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from options_helper.data.alpaca_client import _load_market_tz
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.market_types import DataFetchError
from options_helper.data.streaming.alpaca_stream import AlpacaOptionStreamer, AlpacaStockStreamer
from options_helper.data.streaming.intraday_writer import BufferedIntradayWriter, PartitionSpec
from options_helper.data.streaming.normalizers import (
    NormalizedEvent,
    normalize_option_quote,
    normalize_option_trade,
    normalize_stock_bar,
    normalize_stock_quote,
    normalize_stock_trade,
)

logger = logging.getLogger(__name__)

DatasetSpec = tuple[str, str, str]

_DATASET_SPECS: dict[str, DatasetSpec] = {
    "stock_bars": ("stocks", "bars", "1Min"),
    "stock_quotes": ("stocks", "quotes", "tick"),
    "stock_trades": ("stocks", "trades", "tick"),
    "option_quotes": ("options", "quotes", "tick"),
    "option_trades": ("options", "trades", "tick"),
    "option_bars": ("options", "bars", "1Min"),
}


def compute_backoff_seconds(
    attempt: int,
    *,
    base_seconds: float = 0.5,
    cap_seconds: float = 30.0,
) -> float:
    if attempt <= 0:
        return 0.0
    delay = base_seconds * (2 ** (attempt - 1))
    return min(float(delay), float(cap_seconds))


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
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _market_day(ts: datetime, market_tz: ZoneInfo) -> date:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(market_tz).date()


@dataclass
class CaptureBuffer:
    store: IntradayStore
    market_tz: ZoneInfo = field(default_factory=_load_market_tz)
    base_meta: dict[str, Any] = field(default_factory=dict)

    _writers: dict[tuple[str, str, str, str, date], BufferedIntradayWriter] = field(
        default_factory=dict, init=False
    )

    def ingest(self, event: NormalizedEvent) -> None:
        if event is None:
            return
        spec = _DATASET_SPECS.get(event.dataset)
        if spec is None:
            return
        kind, dataset, timeframe = spec
        ts = _coerce_timestamp(event.row.get("timestamp"))
        if ts is None:
            return
        day = _market_day(ts, self.market_tz)
        key = (kind, dataset, timeframe, event.symbol, day)
        writer = self._writers.get(key)
        if writer is None:
            part = PartitionSpec(kind=kind, dataset=dataset, timeframe=timeframe, symbol=event.symbol, day=day)
            meta = dict(self.base_meta)
            meta.setdefault("kind", kind)
            meta.setdefault("dataset", dataset)
            meta.setdefault("timeframe", timeframe)
            writer = BufferedIntradayWriter(self.store, part, meta=meta)
            self._writers[key] = writer
        writer.add(event.row)

    def flush_all(self) -> list[Path]:
        written: list[Path] = []
        for writer in list(self._writers.values()):
            try:
                out = writer.flush()
            except Exception as exc:  # noqa: BLE001
                raise DataFetchError("Failed to flush streaming capture partitions.") from exc
            if out is not None:
                written.append(out)
        return written

    @property
    def writer_count(self) -> int:
        return len(self._writers)


class _StreamWorker:
    def __init__(
        self,
        *,
        name: str,
        create_streamer: Callable[[], Any],
        subscribe_symbols: list[str],
        stop_event: threading.Event,
        max_reconnects: int,
    ) -> None:
        self._name = name
        self._create_streamer = create_streamer
        self._symbols = subscribe_symbols
        self._stop_event = stop_event
        self._max_reconnects = max_reconnects
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._active_lock = threading.Lock()
        self._active_streamer: Any | None = None

    @property
    def error(self) -> Exception | None:
        return self._error

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._active_lock:
            streamer = self._active_streamer
        if streamer is not None:
            try:
                if hasattr(streamer, "stop"):
                    streamer.stop()
            except Exception:  # noqa: BLE001
                pass
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)

    def _run(self) -> None:
        reconnect_attempts = 0
        while not self._stop_event.is_set():
            streamer = None
            try:
                streamer = self._create_streamer()
                with self._active_lock:
                    self._active_streamer = streamer
                if hasattr(streamer, "subscribe"):
                    streamer.subscribe(self._symbols)
                if hasattr(streamer, "run"):
                    streamer.run()
                else:
                    raise DataFetchError(f"{self._name} streamer missing run()")
                if self._stop_event.is_set():
                    return
                raise DataFetchError(f"{self._name} stream stopped unexpectedly.")
            except Exception as exc:  # noqa: BLE001
                if self._stop_event.is_set():
                    return
                reconnect_attempts += 1
                if reconnect_attempts > self._max_reconnects:
                    self._error = exc
                    self._stop_event.set()
                    return
                delay = compute_backoff_seconds(reconnect_attempts)
                logger.warning(
                    "%s stream error (%s). Reconnecting in %.1fs (attempt %s/%s).",
                    self._name,
                    exc,
                    delay,
                    reconnect_attempts,
                    self._max_reconnects,
                )
                time.sleep(delay)
            finally:
                with self._active_lock:
                    self._active_streamer = None
                if streamer is not None:
                    try:
                        if hasattr(streamer, "stop"):
                            streamer.stop()
                    except Exception:  # noqa: BLE001
                        pass


@dataclass
class StreamRunner:
    out_dir: Path
    stocks: list[str] = field(default_factory=list)
    option_contracts: list[str] = field(default_factory=list)
    capture_bars: bool = True
    capture_quotes: bool = False
    capture_trades: bool = False
    flush_interval_seconds: float = 10.0
    flush_every_events: int = 250
    max_reconnects: int = 5
    stock_feed: str | None = None
    options_feed: str | None = None
    provider: str = "alpaca"

    _queue: queue.Queue[NormalizedEvent] = field(default_factory=queue.Queue, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _buffer: CaptureBuffer | None = field(default=None, init=False)
    _workers: list[_StreamWorker] = field(default_factory=list, init=False)

    def _base_meta(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "stock_feed": self.stock_feed,
            "options_feed": self.options_feed,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def _build_buffer(self) -> CaptureBuffer:
        store = IntradayStore(self.out_dir)
        return CaptureBuffer(store, base_meta=self._base_meta())

    def _on_event(self, event: NormalizedEvent | None) -> None:
        if event is None:
            return
        try:
            self._queue.put_nowait(event)
        except Exception:  # noqa: BLE001
            return

    def _build_workers(self) -> list[_StreamWorker]:
        workers: list[_StreamWorker] = []

        if self.stocks:
            async def _on_bar(payload: Any) -> None:
                self._on_event(normalize_stock_bar(payload))

            async def _on_quote(payload: Any) -> None:
                self._on_event(normalize_stock_quote(payload))

            async def _on_trade(payload: Any) -> None:
                self._on_event(normalize_stock_trade(payload))

            stock_streamer = lambda: AlpacaStockStreamer(  # noqa: E731 - factory
                feed=self.stock_feed,
                on_bars=_on_bar if self.capture_bars else None,
                on_quotes=_on_quote if self.capture_quotes else None,
                on_trades=_on_trade if self.capture_trades else None,
            )
            workers.append(
                _StreamWorker(
                    name="alpaca-stocks",
                    create_streamer=stock_streamer,
                    subscribe_symbols=self.stocks,
                    stop_event=self._stop_event,
                    max_reconnects=self.max_reconnects,
                )
            )

        if self.option_contracts:
            async def _on_opt_quote(payload: Any) -> None:
                self._on_event(normalize_option_quote(payload))

            async def _on_opt_trade(payload: Any) -> None:
                self._on_event(normalize_option_trade(payload))

            option_streamer = lambda: AlpacaOptionStreamer(  # noqa: E731 - factory
                feed=self.options_feed,
                on_quotes=_on_opt_quote if self.capture_quotes else None,
                on_trades=_on_opt_trade if self.capture_trades else None,
                on_bars=None,
            )
            workers.append(
                _StreamWorker(
                    name="alpaca-options",
                    create_streamer=option_streamer,
                    subscribe_symbols=self.option_contracts,
                    stop_event=self._stop_event,
                    max_reconnects=self.max_reconnects,
                )
            )

        return workers

    def stop(self) -> None:
        self._stop_event.set()

    def run(self, *, duration_seconds: float | None = None) -> list[Path]:
        if not (self.stocks or self.option_contracts):
            raise DataFetchError("Provide at least one stock symbol or option contract to stream.")

        if not (self.capture_bars or self.capture_quotes or self.capture_trades):
            raise DataFetchError("Enable at least one capture type: bars, quotes, or trades.")

        self._buffer = self._build_buffer()
        self._workers = self._build_workers()
        for worker in self._workers:
            worker.start()

        start = time.monotonic()
        last_flush = start
        events_seen = 0
        written: list[Path] = []

        try:
            while not self._stop_event.is_set():
                if duration_seconds is not None and (time.monotonic() - start) >= duration_seconds:
                    break

                timeout = 0.25
                if self.flush_interval_seconds > 0:
                    until = max(0.0, self.flush_interval_seconds - (time.monotonic() - last_flush))
                    timeout = min(timeout, until) if until else 0.0

                try:
                    event = self._queue.get(timeout=timeout)
                except queue.Empty:
                    event = None

                if event is not None and self._buffer is not None:
                    self._buffer.ingest(event)
                    events_seen += 1

                if (
                    self.flush_every_events > 0
                    and events_seen
                    and (events_seen % self.flush_every_events == 0)
                    and self._buffer is not None
                ):
                    written.extend(self._buffer.flush_all())
                    last_flush = time.monotonic()

                if (
                    self.flush_interval_seconds > 0
                    and (time.monotonic() - last_flush) >= self.flush_interval_seconds
                    and self._buffer is not None
                ):
                    written.extend(self._buffer.flush_all())
                    last_flush = time.monotonic()

                for worker in self._workers:
                    if worker.error is not None:
                        raise DataFetchError(f"{worker.error}")

        except KeyboardInterrupt:
            logger.info("Streaming capture interrupted; flushing buffers.")
        finally:
            self._stop_event.set()
            for worker in self._workers:
                worker.stop()

            if self._buffer is not None:
                # Drain remaining events before the final flush.
                while True:
                    try:
                        event = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._buffer.ingest(event)
                written.extend(self._buffer.flush_all())

        return written


__all__ = ["CaptureBuffer", "StreamRunner", "compute_backoff_seconds"]
