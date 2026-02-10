from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from importlib import import_module
from typing import Any

from options_helper.data.alpaca_client import _maybe_load_alpaca_env
from options_helper.data.market_types import DataFetchError

logger = logging.getLogger(__name__)

StreamHandler = Callable[[Any], Any]


def _clean_env(value: str | None) -> str:
    return (value or "").strip()


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


def _resolve_paper_flag(paper: bool | None, api_base_url: str | None) -> bool | None:
    if paper is not None:
        return bool(paper)
    base_url = _clean_env(api_base_url).lower()
    if not base_url:
        return None
    if "paper" in base_url:
        return True
    if "sandbox" in base_url:
        return True
    if "api.alpaca.markets" in base_url:
        return False
    if "live" in base_url:
        return False
    return None


def _resolve_url_override(api_base_url: str | None) -> str | None:
    base_url = _clean_env(api_base_url)
    if not base_url:
        return None
    normalized = base_url.lower()
    if normalized.startswith("wss://") or normalized.startswith("ws://"):
        return base_url
    return None


def _load_trading_stream_class() -> tuple[type[Any] | None, Exception | None]:
    try:
        module = import_module("alpaca.trading.stream")
        stream_cls = getattr(module, "TradingStream")
        return stream_cls, None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def _construct_stream(stream_cls: type[Any], **kwargs: Any) -> Any:
    filtered = {key: value for key, value in kwargs.items() if value is not None}
    try:
        sig = inspect.signature(stream_cls)
        has_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
        )
        if not has_var_kwargs:
            allowed = set(sig.parameters)
            filtered = {key: value for key, value in filtered.items() if key in allowed}
    except (TypeError, ValueError):
        pass
    return stream_cls(**filtered)


class AlpacaTradingStreamer:
    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        api_base_url: str | None = None,
        paper: bool | None = None,
        stream: Any | None = None,
        stream_cls: type[Any] | None = None,
        on_trade_updates: StreamHandler | None = None,
    ) -> None:
        self._on_trade_updates = on_trade_updates
        if stream is not None:
            self._stream = stream
            return

        if stream_cls is None:
            stream_cls = self._default_stream_cls()

        api_key, api_secret, base_url = _resolve_credentials(
            api_key_id, api_secret_key, api_base_url
        )
        resolved_paper = _resolve_paper_flag(paper, base_url)
        resolved_url_override = _resolve_url_override(base_url)
        stream_kwargs = {
            "api_key": api_key,
            "api_key_id": api_key,
            "key_id": api_key,
            "secret_key": api_secret,
            "secret": api_secret,
            "api_secret_key": api_secret,
            "paper": resolved_paper,
            "url_override": resolved_url_override,
            "base_url": resolved_url_override,
            "url": resolved_url_override,
        }
        self._stream = _construct_stream(stream_cls, **stream_kwargs)

    @property
    def stream(self) -> Any:
        return self._stream

    def _default_stream_cls(self) -> type[Any]:
        stream_cls, exc = _load_trading_stream_class()
        if stream_cls is None:
            message = (
                "Alpaca trading streaming requires alpaca-py. Install with `pip install -e "
                "\".[alpaca]\"`."
            )
            if exc is not None:
                message = f"{message} (import error: {exc})"
            raise DataFetchError(message)
        return stream_cls

    def subscribe_trade_updates(self, handler: StreamHandler | None = None) -> None:
        if handler is not None:
            self._on_trade_updates = handler
        if self._on_trade_updates is None:
            raise DataFetchError("Alpaca trading stream requires an on_trade_updates handler.")
        method = getattr(self._stream, "subscribe_trade_updates", None)
        if not callable(method):
            raise DataFetchError(
                "Alpaca trading stream does not support trade update subscriptions."
            )
        method(self._on_trade_updates)

    def run(self) -> None:
        method = getattr(self._stream, "run", None) or getattr(self._stream, "start", None)
        if not callable(method):
            raise DataFetchError("Alpaca trading stream does not support run/start.")
        method()

    def stop(self) -> None:
        for name in ("stop", "close", "disconnect"):
            method = getattr(self._stream, name, None)
            if callable(method):
                method()
                return
        logger.warning("Alpaca trading stream has no stop/close method.")


__all__ = ["AlpacaTradingStreamer", "StreamHandler"]
