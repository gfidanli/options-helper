from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

from options_helper.data.alpaca_client import _maybe_load_alpaca_env
from options_helper.data.alpaca_symbols import to_alpaca_symbol
from options_helper.data.market_types import DataFetchError

logger = logging.getLogger(__name__)

StreamHandler = Callable[[Any], Any]

_SUBSCRIBE_METHODS = {
    "bars": ("subscribe_bars", "subscribe_option_bars", "subscribe_options_bars"),
    "trades": ("subscribe_trades", "subscribe_option_trades", "subscribe_options_trades"),
    "quotes": ("subscribe_quotes", "subscribe_option_quotes", "subscribe_options_quotes"),
}

_UNSUBSCRIBE_METHODS = {
    "bars": ("unsubscribe_bars", "unsubscribe_option_bars", "unsubscribe_options_bars"),
    "trades": ("unsubscribe_trades", "unsubscribe_option_trades", "unsubscribe_options_trades"),
    "quotes": ("unsubscribe_quotes", "unsubscribe_option_quotes", "unsubscribe_options_quotes"),
}


def _clean_env(value: str | None) -> str:
    return (value or "").strip()


def _resolve_feed(value: str | None, env_key: str) -> str | None:
    resolved = _clean_env(value) or _clean_env(os.getenv(env_key))
    return resolved or None


def _resolve_credentials(
    api_key_id: str | None,
    api_secret_key: str | None,
    api_base_url: str | None,
) -> tuple[str, str, str | None]:
    _maybe_load_alpaca_env()
    key_id = _clean_env(api_key_id) or _clean_env(os.getenv("APCA_API_KEY_ID"))
    secret = _clean_env(api_secret_key) or _clean_env(os.getenv("APCA_API_SECRET_KEY"))
    base_url = _clean_env(api_base_url) or _clean_env(os.getenv("APCA_API_BASE_URL"))
    if not key_id or not secret:
        raise DataFetchError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )
    return key_id, secret, base_url or None


def _load_stream_class(class_name: str) -> tuple[type[Any] | None, Exception | None]:
    last_exc: Exception | None = None
    for module_name in ("alpaca.data.live", "alpaca.data.stream"):
        try:
            module = import_module(module_name)
            stream_cls = getattr(module, class_name)
            return stream_cls, None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    return None, last_exc


def _construct_stream(stream_cls: type[Any], **kwargs: Any) -> Any:
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    try:
        sig = inspect.signature(stream_cls)
        allowed = set(sig.parameters)
        filtered = {k: v for k, v in filtered.items() if k in allowed}
    except (TypeError, ValueError):
        pass
    return stream_cls(**filtered)


def _method_for(stream: Any, names: tuple[str, ...], *, event: str, label: str) -> Callable[..., Any]:
    for name in names:
        method = getattr(stream, name, None)
        if callable(method):
            return method
    raise DataFetchError(f"Alpaca {label} stream does not support {event} subscriptions.")


class _AlpacaStreamerBase:
    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        api_base_url: str | None = None,
        feed: str | None = None,
        on_bars: StreamHandler | None = None,
        on_trades: StreamHandler | None = None,
        on_quotes: StreamHandler | None = None,
        stream: Any | None = None,
        stream_cls: type[Any] | None = None,
        symbol_mapper: Callable[[str], str] | None = to_alpaca_symbol,
        label: str = "data",
    ) -> None:
        self._label = label
        self._symbol_mapper = symbol_mapper
        self._handlers: dict[str, StreamHandler | None] = {
            "bars": on_bars,
            "trades": on_trades,
            "quotes": on_quotes,
        }
        if stream is not None:
            self._stream = stream
            return

        if stream_cls is None:
            stream_cls = self._default_stream_cls()

        api_key, api_secret, base_url = _resolve_credentials(
            api_key_id, api_secret_key, api_base_url
        )
        stream_kwargs = {
            "api_key": api_key,
            "api_key_id": api_key,
            "key_id": api_key,
            "secret_key": api_secret,
            "secret": api_secret,
            "api_secret_key": api_secret,
            "base_url": base_url,
            "url": base_url,
            "feed": feed,
            "data_feed": feed,
        }
        self._stream = _construct_stream(stream_cls, **stream_kwargs)

    @property
    def stream(self) -> Any:
        return self._stream

    def _default_stream_cls(self) -> type[Any]:
        raise NotImplementedError

    def _normalize_symbols(self, symbols: Iterable[str] | str) -> list[str]:
        if isinstance(symbols, str):
            symbols = [symbols]
        seen: set[str] = set()
        output: list[str] = []
        for symbol in symbols:
            if symbol is None:
                continue
            raw = str(symbol).strip()
            if not raw:
                continue
            mapped = self._symbol_mapper(raw) if self._symbol_mapper else raw.upper()
            if not mapped:
                continue
            if mapped in seen:
                continue
            seen.add(mapped)
            output.append(mapped)
        return output

    def subscribe(self, symbols: Iterable[str] | str) -> list[str]:
        normalized = self._normalize_symbols(symbols)
        if not normalized:
            return []
        for event, handler in self._handlers.items():
            if handler is None:
                continue
            method = _method_for(
                self._stream, _SUBSCRIBE_METHODS[event], event=event, label=self._label
            )
            method(handler, *normalized)
        return normalized

    def unsubscribe(self, symbols: Iterable[str] | str) -> list[str]:
        normalized = self._normalize_symbols(symbols)
        if not normalized:
            return []
        for event, handler in self._handlers.items():
            if handler is None:
                continue
            method = _method_for(
                self._stream, _UNSUBSCRIBE_METHODS[event], event=event, label=self._label
            )
            method(*normalized)
        return normalized

    def run(self) -> None:
        method = getattr(self._stream, "run", None) or getattr(self._stream, "start", None)
        if not callable(method):
            raise DataFetchError(f"Alpaca {self._label} stream does not support run/start.")
        method()

    def stop(self) -> None:
        for name in ("stop", "close", "disconnect"):
            method = getattr(self._stream, name, None)
            if callable(method):
                method()
                return
        logger.warning("Alpaca %s stream has no stop/close method.", self._label)


class AlpacaStockStreamer(_AlpacaStreamerBase):
    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        api_base_url: str | None = None,
        feed: str | None = None,
        on_bars: StreamHandler | None = None,
        on_trades: StreamHandler | None = None,
        on_quotes: StreamHandler | None = None,
        stream: Any | None = None,
        stream_cls: type[Any] | None = None,
        symbol_mapper: Callable[[str], str] | None = to_alpaca_symbol,
    ) -> None:
        resolved_feed = _resolve_feed(feed, "OH_ALPACA_STOCK_FEED")
        super().__init__(
            api_key_id=api_key_id,
            api_secret_key=api_secret_key,
            api_base_url=api_base_url,
            feed=resolved_feed,
            on_bars=on_bars,
            on_trades=on_trades,
            on_quotes=on_quotes,
            stream=stream,
            stream_cls=stream_cls,
            symbol_mapper=symbol_mapper,
            label="stock",
        )

    def _default_stream_cls(self) -> type[Any]:
        stream_cls, exc = _load_stream_class("StockDataStream")
        if stream_cls is None:
            message = "Alpaca stock streaming requires alpaca-py. Install with `pip install -e \".[alpaca]\"`."
            if exc is not None:
                message = f"{message} (import error: {exc})"
            raise DataFetchError(message)
        return stream_cls


class AlpacaOptionStreamer(_AlpacaStreamerBase):
    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        api_base_url: str | None = None,
        feed: str | None = None,
        on_bars: StreamHandler | None = None,
        on_trades: StreamHandler | None = None,
        on_quotes: StreamHandler | None = None,
        stream: Any | None = None,
        stream_cls: type[Any] | None = None,
        symbol_mapper: Callable[[str], str] | None = None,
    ) -> None:
        resolved_feed = _resolve_feed(feed, "OH_ALPACA_OPTIONS_FEED")
        super().__init__(
            api_key_id=api_key_id,
            api_secret_key=api_secret_key,
            api_base_url=api_base_url,
            feed=resolved_feed,
            on_bars=on_bars,
            on_trades=on_trades,
            on_quotes=on_quotes,
            stream=stream,
            stream_cls=stream_cls,
            symbol_mapper=symbol_mapper,
            label="option",
        )

    def _default_stream_cls(self) -> type[Any]:
        stream_cls, exc = _load_stream_class("OptionDataStream")
        if stream_cls is None:
            message = "Alpaca option streaming requires alpaca-py. Install with `pip install -e \".[alpaca]\"`."
            if exc is not None:
                message = f"{message} (import error: {exc})"
            raise DataFetchError(message)
        return stream_cls


__all__ = ["AlpacaOptionStreamer", "AlpacaStockStreamer", "StreamHandler"]
