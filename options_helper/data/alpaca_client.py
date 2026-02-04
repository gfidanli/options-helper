from __future__ import annotations

import inspect
import os
from importlib import metadata

from options_helper.data.alpaca_symbols import to_alpaca_symbol, to_repo_symbol
from options_helper.data.market_types import DataFetchError

try:  # pragma: no cover - import guard
    from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    _ALPACA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - optional dependency
    StockHistoricalDataClient = None
    OptionHistoricalDataClient = None
    TradingClient = None
    _ALPACA_IMPORT_ERROR = exc


def _clean_env(value: str | None) -> str:
    return (value or "").strip()


def _coerce_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


class AlpacaClient:
    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        api_base_url: str | None = None,
        stock_feed: str | None = None,
        options_feed: str | None = None,
        recent_bars_buffer_minutes: int | None = None,
    ) -> None:
        self._api_key_id = _clean_env(api_key_id) or _clean_env(os.getenv("APCA_API_KEY_ID"))
        self._api_secret_key = _clean_env(api_secret_key) or _clean_env(os.getenv("APCA_API_SECRET_KEY"))
        self._api_base_url = _clean_env(api_base_url) or _clean_env(os.getenv("APCA_API_BASE_URL"))
        self._stock_feed = _clean_env(stock_feed) or _clean_env(os.getenv("OH_ALPACA_STOCK_FEED"))
        self._options_feed = _clean_env(options_feed) or _clean_env(os.getenv("OH_ALPACA_OPTIONS_FEED"))
        if recent_bars_buffer_minutes is None:
            self._recent_bars_buffer_minutes = _coerce_int(
                os.getenv("OH_ALPACA_RECENT_BARS_BUFFER_MINUTES"),
                default=16,
            )
        else:
            self._recent_bars_buffer_minutes = recent_bars_buffer_minutes

        self._stock_client = None
        self._option_client = None
        self._trading_client = None

    @property
    def provider_version(self) -> str | None:
        try:
            return metadata.version("alpaca-py")
        except metadata.PackageNotFoundError:
            return None

    @property
    def recent_bars_buffer_minutes(self) -> int:
        return self._recent_bars_buffer_minutes

    @property
    def stock_feed(self) -> str | None:
        return self._stock_feed or None

    @property
    def options_feed(self) -> str | None:
        return self._options_feed or None

    @property
    def api_base_url(self) -> str | None:
        return self._api_base_url or None

    @property
    def stock_client(self):
        if self._stock_client is None:
            self._require_sdk()
            self._ensure_credentials()
            self._stock_client = self._construct_client(
                StockHistoricalDataClient,
                **self._credential_kwargs(),
                **self._feed_kwargs(self._stock_feed),
            )
        return self._stock_client

    @property
    def option_client(self):
        if self._option_client is None:
            self._require_sdk()
            self._ensure_credentials()
            self._option_client = self._construct_client(
                OptionHistoricalDataClient,
                **self._credential_kwargs(),
                **self._feed_kwargs(self._options_feed),
            )
        return self._option_client

    @property
    def trading_client(self):
        if self._trading_client is None:
            self._require_sdk()
            self._ensure_credentials()
            kwargs = self._credential_kwargs()
            if self._api_base_url:
                kwargs.update({"base_url": self._api_base_url, "url_override": self._api_base_url})
            self._trading_client = self._construct_client(TradingClient, **kwargs)
        return self._trading_client

    def _ensure_credentials(self) -> None:
        if not self._api_key_id or not self._api_secret_key:
            raise DataFetchError(
                "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY."
            )

    def _require_sdk(self) -> None:
        if StockHistoricalDataClient is None or OptionHistoricalDataClient is None or TradingClient is None:
            message = (
                "Alpaca provider requires alpaca-py. Install with `pip install -e \".[alpaca]\"`."
            )
            if _ALPACA_IMPORT_ERROR is not None:
                message = f"{message} (import error: {_ALPACA_IMPORT_ERROR})"
            raise DataFetchError(message)

    def _credential_kwargs(self) -> dict[str, str]:
        return {
            "api_key": self._api_key_id,
            "api_key_id": self._api_key_id,
            "key_id": self._api_key_id,
            "secret_key": self._api_secret_key,
            "secret": self._api_secret_key,
            "api_secret_key": self._api_secret_key,
        }

    def _feed_kwargs(self, feed: str | None) -> dict[str, str]:
        if not feed:
            return {}
        return {
            "feed": feed,
            "data_feed": feed,
        }

    @staticmethod
    def _construct_client(client_cls, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        try:
            sig = inspect.signature(client_cls)
            allowed = set(sig.parameters)
            filtered = {k: v for k, v in filtered.items() if k in allowed}
        except (TypeError, ValueError):
            pass
        return client_cls(**filtered)


__all__ = [
    "AlpacaClient",
    "to_alpaca_symbol",
    "to_repo_symbol",
]
