from __future__ import annotations

import inspect
import logging
import os
import time
from urllib.parse import urlparse
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from typing import Any

import pandas as pd
try:
    from requests.adapters import HTTPAdapter
except Exception:  # noqa: BLE001 - optional dependency at runtime
    HTTPAdapter = None

from options_helper.data.alpaca_symbols import to_alpaca_symbol, to_repo_symbol
from options_helper.data.alpaca_rate_limits import (
    AlpacaRateLimitSnapshot,
    parse_alpaca_rate_limit_headers,
)
from options_helper.data.alpaca_client_legacy_support.chain_contracts import (
    contracts_to_df,
    option_chain_to_rows,
)
from options_helper.data.alpaca_client_legacy_support import (
    _bars_to_dataframe,
    _clean_env,
    _coerce_datetime,
    _coerce_int,
    _coerce_optional_int,
    _contract_to_dict,
    _env_truthy,
    _extract_contracts_page,
    _extract_corporate_actions_page,
    _extract_news_page,
    _extract_option_bars_page_token,
    _extract_retry_after_seconds,
    _extract_status_code,
    _filter_kwargs,
    _is_rate_limit_error,
    _is_timeout_error,
    _load_market_tz,
    _market_day_bounds,
    _maybe_load_alpaca_env,
    _normalize_corporate_action,
    _normalize_intraday_option_bars,
    _normalize_intraday_stock_bars,
    _normalize_news_item,
    _normalize_option_bars,
    _normalize_option_bars_daily_full,
    _normalize_stock_bars,
    _option_bars_to_dataframe,
    get_option_bars_daily_full_impl,
    get_option_bars_intraday_impl,
    get_stock_bars_impl,
    get_stock_bars_intraday_impl,
    list_option_contracts_impl,
)
from options_helper.data.market_types import DataFetchError

logger = logging.getLogger("options_helper.cli")

try:  # pragma: no cover - import guard
    from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    _ALPACA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - optional dependency
    StockHistoricalDataClient = None
    OptionHistoricalDataClient = None
    TradingClient = None
    _ALPACA_IMPORT_ERROR = exc

TimeFrame = None
TimeFrameUnit = None
StockBarsRequest = None
OptionContractsRequest = None
OptionChainRequest = None
OptionBarsRequest = None
CorporateActionsClient = None
CorporateActionsRequest = None
NewsClient = None
NewsRequest = None


def _load_timeframe():
    global TimeFrame, TimeFrameUnit  # noqa: PLW0603 - intentional lazy import
    if TimeFrame is not None:
        return TimeFrame, TimeFrameUnit
    try:
        from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit as AlpacaTimeFrameUnit

        TimeFrame = AlpacaTimeFrame
        TimeFrameUnit = AlpacaTimeFrameUnit
    except Exception:  # noqa: BLE001
        return None, None
    return TimeFrame, TimeFrameUnit


def _load_stock_bars_request():
    global StockBarsRequest  # noqa: PLW0603 - intentional lazy import
    if StockBarsRequest is not None:
        return StockBarsRequest
    try:
        from alpaca.data.requests import StockBarsRequest as AlpacaStockBarsRequest

        StockBarsRequest = AlpacaStockBarsRequest
    except Exception:  # noqa: BLE001
        return None
    return StockBarsRequest


def _load_option_contracts_request():
    global OptionContractsRequest  # noqa: PLW0603 - intentional lazy import
    if OptionContractsRequest is not None:
        return OptionContractsRequest
    try:
        from alpaca.trading.requests import GetOptionContractsRequest as AlpacaGetOptionContractsRequest

        OptionContractsRequest = AlpacaGetOptionContractsRequest
    except Exception:  # noqa: BLE001
        try:
            from alpaca.trading.requests import OptionContractsRequest as AlpacaOptionContractsRequest

            OptionContractsRequest = AlpacaOptionContractsRequest
        except Exception:  # noqa: BLE001
            return None
    return OptionContractsRequest


def _load_option_chain_request():
    global OptionChainRequest  # noqa: PLW0603 - intentional lazy import
    if OptionChainRequest is not None:
        return OptionChainRequest
    try:
        from alpaca.data.requests import OptionChainRequest as AlpacaOptionChainRequest

        OptionChainRequest = AlpacaOptionChainRequest
    except Exception:  # noqa: BLE001
        return None
    return OptionChainRequest


def _load_option_bars_request():
    global OptionBarsRequest  # noqa: PLW0603 - intentional lazy import
    if OptionBarsRequest is not None:
        return OptionBarsRequest
    try:
        from alpaca.data.requests import OptionBarsRequest as AlpacaOptionBarsRequest

        OptionBarsRequest = AlpacaOptionBarsRequest
    except Exception:  # noqa: BLE001
        return None
    return OptionBarsRequest


def _load_corporate_actions_client():
    global CorporateActionsClient  # noqa: PLW0603 - intentional lazy import
    if CorporateActionsClient is not None:
        return CorporateActionsClient
    try:
        from alpaca.data.historical.corporate_actions import (
            CorporateActionsClient as AlpacaCorporateActionsClient,
        )

        CorporateActionsClient = AlpacaCorporateActionsClient
    except Exception:  # noqa: BLE001
        return None
    return CorporateActionsClient


def _load_corporate_actions_request():
    global CorporateActionsRequest  # noqa: PLW0603 - intentional lazy import
    if CorporateActionsRequest is not None:
        return CorporateActionsRequest
    try:
        from alpaca.data.requests import CorporateActionsRequest as AlpacaCorporateActionsRequest

        CorporateActionsRequest = AlpacaCorporateActionsRequest
    except Exception:  # noqa: BLE001
        return None
    return CorporateActionsRequest


def _load_news_client():
    global NewsClient  # noqa: PLW0603 - intentional lazy import
    if NewsClient is not None:
        return NewsClient
    try:
        from alpaca.data.historical.news import NewsClient as AlpacaNewsClient

        NewsClient = AlpacaNewsClient
    except Exception:  # noqa: BLE001
        return None
    return NewsClient


def _load_news_request():
    global NewsRequest  # noqa: PLW0603 - intentional lazy import
    if NewsRequest is not None:
        return NewsRequest
    try:
        from alpaca.data.requests import NewsRequest as AlpacaNewsRequest

        NewsRequest = AlpacaNewsRequest
    except Exception:  # noqa: BLE001
        return None
    return NewsRequest

def _resolve_timeframe(interval: str) -> Any:
    interval = (interval or "").strip().lower()
    mapping = {
        "1d": ("day", 1, "Day"),
        "1h": ("hour", 1, "Hour"),
        "1m": ("minute", 1, "Minute"),
        "1min": ("minute", 1, "Minute"),
        "5m": ("minute", 5, "Minute"),
        "5min": ("minute", 5, "Minute"),
    }
    if interval not in mapping:
        raise DataFetchError(
            f"Unsupported Alpaca interval '{interval}'. Supported: {', '.join(sorted(mapping))}."
        )
    unit_name, amount, attr = mapping[interval]
    timeframe_cls, timeframe_unit = _load_timeframe()
    if timeframe_cls is None:
        raise DataFetchError("Alpaca TimeFrame unavailable. Install with `pip install -e \".[alpaca]\"`.")
    if amount == 1 and hasattr(timeframe_cls, attr):
        return getattr(timeframe_cls, attr)
    if timeframe_unit is not None:
        unit_val = getattr(timeframe_unit, unit_name.capitalize(), None) or getattr(
            timeframe_unit, unit_name.upper(), None
        )
        if unit_val is not None:
            try:
                return timeframe_cls(amount, unit_val)
            except Exception:  # noqa: BLE001
                pass
    try:
        return timeframe_cls(amount, unit_name)
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError(f"Unable to build Alpaca TimeFrame for interval '{interval}'.") from exc


def _raise_missing_optional_client(feature: str) -> None:
    message = f"Alpaca {feature} requires alpaca-py. Install with `pip install -e \".[alpaca]\"`."
    if _ALPACA_IMPORT_ERROR is not None:
        message = f"{message} (import error: {_ALPACA_IMPORT_ERROR})"
    raise DataFetchError(message)


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
        log_rate_limits: bool | None = None,
        http_pool_maxsize: int | None = None,
        http_pool_connections: int | None = None,
        http_pool_block: bool | None = None,
    ) -> None:
        _maybe_load_alpaca_env()
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
        self._corporate_actions_client = None
        self._news_client = None
        self._log_rate_limits = _env_truthy("OH_ALPACA_LOG_RATE_LIMITS") if log_rate_limits is None else bool(
            log_rate_limits
        )
        if http_pool_maxsize is None:
            self._http_pool_maxsize = _coerce_optional_int(os.getenv("OH_ALPACA_HTTP_POOL_MAXSIZE"))
        else:
            self._http_pool_maxsize = max(1, int(http_pool_maxsize))
        if http_pool_connections is None:
            self._http_pool_connections = _coerce_optional_int(os.getenv("OH_ALPACA_HTTP_POOL_CONNECTIONS"))
        else:
            self._http_pool_connections = max(1, int(http_pool_connections))
        self._http_pool_block = _env_truthy("OH_ALPACA_HTTP_POOL_BLOCK") if http_pool_block is None else bool(
            http_pool_block
        )
        self._last_rate_limit: AlpacaRateLimitSnapshot | None = None

    @property
    def last_rate_limit(self) -> AlpacaRateLimitSnapshot | None:
        return self._last_rate_limit

    def _install_rate_limit_hook(self, client: Any, *, client_name: str) -> None:
        if not self._log_rate_limits or client is None:
            return
        session = getattr(client, "_session", None)
        hooks = getattr(session, "hooks", None)
        if hooks is None:
            return
        if getattr(session, "_oh_rate_limit_hook_installed", False):
            return

        def _hook(response, *args, **kwargs):  # noqa: ANN001
            try:
                snapshot = parse_alpaca_rate_limit_headers(getattr(response, "headers", None))
                if snapshot is None:
                    return response
                self._last_rate_limit = snapshot
                request = getattr(response, "request", None)
                method = getattr(request, "method", None) or "?"
                url = getattr(response, "url", None) or getattr(request, "url", None) or ""
                parsed = urlparse(str(url))
                path = parsed.path or "/"
                status = getattr(response, "status_code", None)
                reset_in = snapshot.reset_in_seconds()
                logger.info(
                    "ALPACA_RATELIMIT client=%s method=%s path=%s status=%s limit=%s remaining=%s reset_epoch=%s reset_in_s=%.3f",
                    client_name,
                    method,
                    path,
                    status,
                    snapshot.limit,
                    snapshot.remaining,
                    snapshot.reset_epoch,
                    -1.0 if reset_in is None else reset_in,
                )
            except Exception:  # noqa: BLE001
                return response
            return response

        try:
            existing = hooks.get("response")
            if existing is None:
                hooks["response"] = [_hook]
            elif isinstance(existing, list):
                hooks["response"] = [*existing, _hook]
            else:
                hooks["response"] = [existing, _hook]
            setattr(session, "_oh_rate_limit_hook_installed", True)
        except Exception:  # noqa: BLE001
            return

    def _configure_http_session(self, client: Any, *, client_name: str) -> None:
        if client is None or HTTPAdapter is None:
            return

        session = getattr(client, "_session", None)
        if session is None:
            return
        if getattr(session, "_oh_http_pool_configured", False):
            return

        pool_connections = self._http_pool_connections
        pool_maxsize = self._http_pool_maxsize
        if pool_connections is None and pool_maxsize is None:
            return
        if pool_connections is None:
            pool_connections = pool_maxsize
        if pool_maxsize is None:
            pool_maxsize = pool_connections
        if pool_connections is None or pool_maxsize is None:
            return

        try:
            adapter = HTTPAdapter(
                pool_connections=max(1, int(pool_connections)),
                pool_maxsize=max(1, int(pool_maxsize)),
                pool_block=self._http_pool_block,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            setattr(session, "_oh_http_pool_configured", True)
            logger.info(
                "ALPACA_HTTP_POOL client=%s pool_connections=%s pool_maxsize=%s pool_block=%s",
                client_name,
                pool_connections,
                pool_maxsize,
                self._http_pool_block,
            )
        except Exception:  # noqa: BLE001
            return

    @property
    def provider_version(self) -> str | None:
        try:
            return metadata.version("alpaca-py")
        except metadata.PackageNotFoundError:
            return None

    @property
    def recent_bars_buffer_minutes(self) -> int:
        return self._recent_bars_buffer_minutes

    def effective_end(self, end: date | datetime | None, *, end_of_day: bool = True) -> datetime | None:
        end_dt = _coerce_datetime(end, end_of_day=end_of_day)
        if end_dt is None:
            return datetime.now(timezone.utc) - timedelta(minutes=self._recent_bars_buffer_minutes)
        return end_dt

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
            self._configure_http_session(self._stock_client, client_name="stock")
            self._install_rate_limit_hook(self._stock_client, client_name="stock")
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
            self._configure_http_session(self._option_client, client_name="option")
            self._install_rate_limit_hook(self._option_client, client_name="option")
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
            self._configure_http_session(self._trading_client, client_name="trading")
            self._install_rate_limit_hook(self._trading_client, client_name="trading")
        return self._trading_client

    @property
    def corporate_actions_client(self):
        if self._corporate_actions_client is None:
            self._ensure_credentials()
            client_cls = _load_corporate_actions_client()
            if client_cls is None:
                _raise_missing_optional_client("corporate actions")
            self._corporate_actions_client = self._construct_client(client_cls, **self._credential_kwargs())
            self._configure_http_session(self._corporate_actions_client, client_name="corporate_actions")
            self._install_rate_limit_hook(self._corporate_actions_client, client_name="corporate_actions")
        return self._corporate_actions_client

    @property
    def news_client(self):
        if self._news_client is None:
            self._ensure_credentials()
            client_cls = _load_news_client()
            if client_cls is None:
                _raise_missing_optional_client("news")
            self._news_client = self._construct_client(client_cls, **self._credential_kwargs())
            self._configure_http_session(self._news_client, client_name="news")
            self._install_rate_limit_hook(self._news_client, client_name="news")
        return self._news_client

    def get_stock_bars(
        self,
        symbol: str,
        *,
        start: date | datetime | None,
        end: date | datetime | None,
        interval: str,
        adjustment: str,
    ) -> pd.DataFrame:
        deps = {
            "to_alpaca_symbol": to_alpaca_symbol,
            "resolve_timeframe": _resolve_timeframe,
            "coerce_datetime": _coerce_datetime,
            "load_stock_bars_request": _load_stock_bars_request,
            "coerce_int": _coerce_int,
            "extract_status_code": _extract_status_code,
            "extract_retry_after_seconds": _extract_retry_after_seconds,
            "is_rate_limit_error": _is_rate_limit_error,
            "is_timeout_error": _is_timeout_error,
            "bars_to_dataframe": _bars_to_dataframe,
            "normalize_stock_bars": _normalize_stock_bars,
        }
        return get_stock_bars_impl(
            self,
            symbol,
            start=start,
            end=end,
            interval=interval,
            adjustment=adjustment,
            deps=deps,
        )

    def get_stock_bars_intraday(
        self,
        symbol: str,
        *,
        day: date,
        timeframe: str = "1Min",
        feed: str | None = None,
        adjustment: str = "raw",
    ) -> pd.DataFrame:
        deps = {
            "to_alpaca_symbol": to_alpaca_symbol,
            "load_market_tz": _load_market_tz,
            "resolve_timeframe": _resolve_timeframe,
            "market_day_bounds": _market_day_bounds,
            "load_stock_bars_request": _load_stock_bars_request,
            "filter_kwargs": _filter_kwargs,
            "bars_to_dataframe": _bars_to_dataframe,
            "normalize_intraday_stock_bars": _normalize_intraday_stock_bars,
        }
        return get_stock_bars_intraday_impl(
            self,
            symbol,
            day=day,
            timeframe=timeframe,
            feed=feed,
            adjustment=adjustment,
            deps=deps,
        )

    def get_option_bars(
        self,
        symbols: list[str],
        *,
        start: date | datetime | None,
        end: date | datetime | None = None,
        interval: str = "1d",
        feed: str | None = None,
        max_chunk_size: int = 200,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        raw_symbols = [str(sym).strip() for sym in (symbols or []) if sym]
        unique_symbols = sorted({sym for sym in raw_symbols if sym})
        if not unique_symbols:
            return pd.DataFrame(columns=["contractSymbol", "timestamp", "volume", "vwap", "trade_count"])

        timeframe = _resolve_timeframe(interval)
        start_dt = _coerce_datetime(start, end_of_day=False)
        end_dt = self.effective_end(end, end_of_day=True)
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise DataFetchError("End date is before start date for Alpaca option bars request.")

        client = self.option_client
        method = getattr(client, "get_option_bars", None)
        if method is None:
            raise DataFetchError("Alpaca data client missing get_option_bars.")

        request_cls = _load_option_bars_request()
        feed_val = feed or self._options_feed
        all_chunks: list[pd.DataFrame] = []

        def _call_with_backoff(make_call):
            for attempt in range(max_retries):
                try:
                    return make_call()
                except Exception as exc:  # noqa: BLE001
                    if attempt >= max_retries - 1 or not _is_rate_limit_error(exc):
                        raise
                    time.sleep(0.5 * (2**attempt))
            return None

        for i in range(0, len(unique_symbols), max_chunk_size):
            chunk = unique_symbols[i : i + max_chunk_size]
            kwargs = {
                "symbol_or_symbols": chunk,
                "symbols": chunk,
                "timeframe": timeframe,
                "start": start_dt,
                "end": end_dt,
                "feed": feed_val,
            }

            payload = None
            if request_cls is not None:
                try:
                    request_kwargs = _filter_kwargs(request_cls, kwargs)
                    request = request_cls(**request_kwargs)
                    payload = _call_with_backoff(lambda: method(request))
                except TypeError:
                    payload = None

            if payload is None:
                payload = _call_with_backoff(lambda: method(**_filter_kwargs(method, kwargs)))

            if payload is None:
                continue

            df = _option_bars_to_dataframe(payload)
            norm = _normalize_option_bars(df)
            if not norm.empty:
                all_chunks.append(norm)

        if not all_chunks:
            return pd.DataFrame(columns=["contractSymbol", "timestamp", "volume", "vwap", "trade_count"])

        combined = pd.concat(all_chunks, ignore_index=True)
        combined = combined.drop_duplicates(subset=["contractSymbol"], keep="last")
        return combined.reset_index(drop=True)

    def get_option_bars_daily_full(
        self,
        symbols: list[str],
        *,
        start: date | datetime | None,
        end: date | datetime | None = None,
        interval: str = "1d",
        feed: str | None = None,
        chunk_size: int = 200,
        max_retries: int = 3,
        page_limit: int | None = None,
    ) -> pd.DataFrame:
        deps = {
            "resolve_timeframe": _resolve_timeframe,
            "coerce_datetime": _coerce_datetime,
            "load_option_bars_request": _load_option_bars_request,
            "filter_kwargs": _filter_kwargs,
            "extract_status_code": _extract_status_code,
            "extract_retry_after_seconds": _extract_retry_after_seconds,
            "is_rate_limit_error": _is_rate_limit_error,
            "is_timeout_error": _is_timeout_error,
            "option_bars_to_dataframe": _option_bars_to_dataframe,
            "normalize_option_bars_daily_full": _normalize_option_bars_daily_full,
            "extract_option_bars_page_token": _extract_option_bars_page_token,
            "load_market_tz": _load_market_tz,
            "market_day_bounds": _market_day_bounds,
            "normalize_intraday_option_bars": _normalize_intraday_option_bars,
        }
        return get_option_bars_daily_full_impl(
            self,
            symbols,
            start=start,
            end=end,
            interval=interval,
            feed=feed,
            chunk_size=chunk_size,
            max_retries=max_retries,
            page_limit=page_limit,
            deps=deps,
        )

    def get_option_bars_intraday(
        self,
        symbols: list[str],
        *,
        day: date,
        timeframe: str = "1Min",
        feed: str | None = None,
        max_chunk_size: int = 200,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        deps = {
            "resolve_timeframe": _resolve_timeframe,
            "coerce_datetime": _coerce_datetime,
            "load_option_bars_request": _load_option_bars_request,
            "filter_kwargs": _filter_kwargs,
            "extract_status_code": _extract_status_code,
            "extract_retry_after_seconds": _extract_retry_after_seconds,
            "is_rate_limit_error": _is_rate_limit_error,
            "is_timeout_error": _is_timeout_error,
            "option_bars_to_dataframe": _option_bars_to_dataframe,
            "normalize_option_bars_daily_full": _normalize_option_bars_daily_full,
            "extract_option_bars_page_token": _extract_option_bars_page_token,
            "load_market_tz": _load_market_tz,
            "market_day_bounds": _market_day_bounds,
            "normalize_intraday_option_bars": _normalize_intraday_option_bars,
        }
        return get_option_bars_intraday_impl(
            self,
            symbols,
            day=day,
            timeframe=timeframe,
            feed=feed,
            max_chunk_size=max_chunk_size,
            max_retries=max_retries,
            deps=deps,
        )

    def get_corporate_actions(
        self,
        symbols: list[str],
        *,
        start: date | datetime | None = None,
        end: date | datetime | None = None,
        types: list[str] | None = None,
        limit: int | None = None,
        page_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        raw_symbols = [str(sym).strip() for sym in (symbols or []) if sym]
        alpaca_symbols = [to_alpaca_symbol(sym) for sym in raw_symbols if to_alpaca_symbol(sym)]
        if not alpaca_symbols:
            return []

        start_dt = _coerce_datetime(start, end_of_day=False)
        end_dt = _coerce_datetime(end, end_of_day=True)
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise DataFetchError("End date is before start date for Alpaca corporate actions request.")

        client = self.corporate_actions_client
        method = getattr(client, "get_corporate_actions", None)
        if method is None:
            raise DataFetchError("Alpaca corporate actions client missing get_corporate_actions.")

        request_cls = _load_corporate_actions_request()
        page_token: str | None = None
        page_count = 0
        actions: list[dict[str, Any]] = []

        while True:
            page_count += 1
            if page_limit is not None and page_count > page_limit:
                raise DataFetchError("Exceeded Alpaca corporate actions page limit.")

            kwargs = {
                "symbols": alpaca_symbols,
                "symbol": alpaca_symbols,
                "start": start_dt,
                "end": end_dt,
                "types": types,
                "limit": limit,
                "page_token": page_token,
            }

            payload = None
            if request_cls is not None:
                try:
                    request_kwargs = _filter_kwargs(request_cls, kwargs)
                    request = request_cls(**request_kwargs)
                    payload = method(request)
                except TypeError:
                    payload = None

            if payload is None:
                try:
                    payload = method(**_filter_kwargs(method, kwargs))
                except Exception as exc:  # noqa: BLE001
                    raise DataFetchError("Failed to fetch Alpaca corporate actions.") from exc

            items, page_token = _extract_corporate_actions_page(payload)
            for item in items:
                actions.append(_normalize_corporate_action(item))

            if not page_token:
                break

        return actions

    def get_news(
        self,
        symbols: list[str],
        *,
        start: date | datetime | None = None,
        end: date | datetime | None = None,
        include_content: bool = False,
        limit: int | None = None,
        page_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        raw_symbols = [str(sym).strip() for sym in (symbols or []) if sym]
        alpaca_symbols = [to_alpaca_symbol(sym) for sym in raw_symbols if to_alpaca_symbol(sym)]
        if not alpaca_symbols:
            return []

        start_dt = _coerce_datetime(start, end_of_day=False)
        end_dt = _coerce_datetime(end, end_of_day=True)
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise DataFetchError("End date is before start date for Alpaca news request.")

        client = self.news_client
        method = getattr(client, "get_news", None)
        if method is None:
            raise DataFetchError("Alpaca news client missing get_news.")

        request_cls = _load_news_request()
        page_token: str | None = None
        page_count = 0
        items: list[dict[str, Any]] = []

        while True:
            page_count += 1
            if page_limit is not None and page_count > page_limit:
                raise DataFetchError("Exceeded Alpaca news page limit.")

            kwargs = {
                "symbols": alpaca_symbols,
                "symbol": alpaca_symbols,
                "start": start_dt,
                "end": end_dt,
                "limit": limit,
                "page_token": page_token,
            }

            payload = None
            if request_cls is not None:
                try:
                    request_kwargs = _filter_kwargs(request_cls, kwargs)
                    request = request_cls(**request_kwargs)
                    payload = method(request)
                except TypeError:
                    payload = None

            if payload is None:
                try:
                    payload = method(**_filter_kwargs(method, kwargs))
                except Exception as exc:  # noqa: BLE001
                    raise DataFetchError("Failed to fetch Alpaca news.") from exc

            page_items, page_token = _extract_news_page(payload)
            for item in page_items:
                items.append(_normalize_news_item(item, include_content=include_content))

            if not page_token:
                break

        return items

    def list_option_contracts(
        self,
        underlying: str | None = None,
        *,
        root_symbol: str | None = None,
        exp_gte: date | None = None,
        exp_lte: date | None = None,
        contract_status: str | None = None,
        limit: int | None = None,
        page_limit: int | None = None,
        max_requests_per_second: float | None = None,
    ) -> list[dict[str, Any]]:
        deps = {
            "to_alpaca_symbol": to_alpaca_symbol,
            "load_option_contracts_request": _load_option_contracts_request,
            "coerce_int": _coerce_int,
            "filter_kwargs": _filter_kwargs,
            "extract_contracts_page": _extract_contracts_page,
            "contract_to_dict": _contract_to_dict,
            "is_rate_limit_error": _is_rate_limit_error,
            "is_timeout_error": _is_timeout_error,
            "extract_retry_after_seconds": _extract_retry_after_seconds,
        }
        return list_option_contracts_impl(
            self,
            underlying,
            root_symbol=root_symbol,
            exp_gte=exp_gte,
            exp_lte=exp_lte,
            contract_status=contract_status,
            limit=limit,
            page_limit=page_limit,
            max_requests_per_second=max_requests_per_second,
            deps=deps,
        )

    def get_option_chain_snapshots(
        self,
        underlying: str,
        *,
        expiry: date,
        feed: str | None = None,
    ) -> Any:
        alpaca_symbol = to_alpaca_symbol(underlying)
        if not alpaca_symbol:
            raise DataFetchError(f"Invalid underlying symbol: {underlying}")

        client = self.option_client
        method = getattr(client, "get_option_chain", None)
        if method is None:
            raise DataFetchError("Alpaca data client missing get_option_chain.")

        kwargs = {
            "underlying_symbol": alpaca_symbol,
            "underlying_symbols": [alpaca_symbol],
            "underlying": alpaca_symbol,
            "symbol": alpaca_symbol,
            "expiration_date": expiry,
            "expiry": expiry,
            "expiration": expiry,
            "feed": feed or self._options_feed,
        }

        payload = None
        request_cls = _load_option_chain_request()
        if request_cls is not None:
            try:
                request_kwargs = _filter_kwargs(request_cls, kwargs)
                request = request_cls(**request_kwargs)
                payload = method(request)
            except TypeError:
                payload = None

        if payload is None:
            try:
                payload = method(**_filter_kwargs(method, kwargs))
            except Exception as exc:  # noqa: BLE001
                raise DataFetchError(
                    f"Failed to fetch Alpaca option chain for {alpaca_symbol} {expiry.isoformat()}."
                ) from exc

        if payload is None:
            raise DataFetchError(f"Empty Alpaca option chain for {alpaca_symbol} {expiry.isoformat()}.")

        return payload

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
    "contracts_to_df",
    "option_chain_to_rows",
    "to_alpaca_symbol",
    "to_repo_symbol",
]
