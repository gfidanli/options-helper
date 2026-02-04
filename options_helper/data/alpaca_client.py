from __future__ import annotations

import inspect
import os
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from typing import Any

import pandas as pd

from options_helper.analysis.osi import parse_contract_symbol
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

TimeFrame = None
TimeFrameUnit = None
StockBarsRequest = None
OptionContractsRequest = None


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


def _coerce_datetime(value: date | datetime | None, *, end_of_day: bool) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.combine(value, datetime.max.time() if end_of_day else datetime.min.time())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_timeframe(interval: str) -> Any:
    interval = (interval or "").strip().lower()
    mapping = {
        "1d": ("day", 1, "Day"),
        "1h": ("hour", 1, "Hour"),
        "1m": ("minute", 1, "Minute"),
    }
    if interval not in mapping:
        raise DataFetchError(
            f"Unsupported Alpaca interval '{interval}'. Supported: {', '.join(sorted(mapping))}."
        )
    unit_name, amount, attr = mapping[interval]
    timeframe_cls, timeframe_unit = _load_timeframe()
    if timeframe_cls is None:
        raise DataFetchError("Alpaca TimeFrame unavailable. Install with `pip install -e \".[alpaca]\"`.")
    if hasattr(timeframe_cls, attr):
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


def _filter_kwargs(target, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    allowed = {name for name in sig.parameters if name != "self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def _bars_to_dataframe(payload: Any, symbol: str) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    df = None
    if hasattr(payload, "df"):
        df = payload.df
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(payload, dict):
        candidate = payload.get(symbol) or payload.get(symbol.upper())
        if candidate is None and len(payload) == 1:
            candidate = next(iter(payload.values()))
        try:
            return pd.DataFrame(candidate)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    try:
        return pd.DataFrame(payload)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _normalize_stock_bars(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        if "symbol" in out.index.names:
            level = out.index.names.index("symbol")
            symbols = out.index.get_level_values(level)
            if symbol in symbols:
                out = out.xs(symbol, level="symbol")
            else:
                out = out.xs(symbols[0], level="symbol")
        else:
            out = out.droplevel(0)

    if "symbol" in out.columns:
        sym_col = out["symbol"].astype(str).str.upper()
        target = symbol.upper()
        filtered = out[sym_col == target]
        if not filtered.empty:
            out = filtered
        out = out.drop(columns=["symbol"])

    if not isinstance(out.index, pd.DatetimeIndex):
        for col in ("timestamp", "time", "t"):
            if col in out.columns:
                out = out.set_index(col)
                break

    rename_map = {
        "o": "Open",
        "open": "Open",
        "h": "High",
        "high": "High",
        "l": "Low",
        "low": "Low",
        "c": "Close",
        "close": "Close",
        "v": "Volume",
        "volume": "Volume",
    }
    for col in list(out.columns):
        key = col.lower()
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise DataFetchError(f"Alpaca bars missing columns: {', '.join(missing)}")

    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    if not isinstance(idx, pd.DatetimeIndex):
        raise DataFetchError("Unable to normalize Alpaca bar index to datetime")
    mask = ~idx.isna()
    out = out.loc[mask].copy()
    out.index = idx[mask].tz_convert(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _extract_contracts_page(payload: Any) -> tuple[list[Any], str | None]:
    if payload is None:
        return [], None
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        contracts = payload.get("option_contracts") or payload.get("contracts") or payload.get("data") or []
        token = (
            payload.get("next_page_token")
            or payload.get("next_token")
            or payload.get("nextPageToken")
            or payload.get("next")
        )
        return list(contracts or []), token
    if hasattr(payload, "option_contracts"):
        contracts = getattr(payload, "option_contracts") or []
        token = getattr(payload, "next_page_token", None) or getattr(payload, "next_token", None)
        return list(contracts), token
    if hasattr(payload, "contracts"):
        contracts = getattr(payload, "contracts") or []
        token = getattr(payload, "next_page_token", None) or getattr(payload, "next_token", None)
        return list(contracts), token
    return [], None


def _contract_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                payload = method()
                if isinstance(payload, dict):
                    return payload
            except Exception:  # noqa: BLE001
                pass
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    fields = [
        "symbol",
        "contractSymbol",
        "option_symbol",
        "underlying_symbol",
        "underlyingSymbol",
        "underlying",
        "expiration_date",
        "expiry",
        "option_type",
        "type",
        "strike_price",
        "strike",
        "multiplier",
        "open_interest",
        "openInterest",
        "open_interest_date",
        "openInterestDate",
        "close_price",
        "closePrice",
        "close_price_date",
        "closePriceDate",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        if hasattr(value, field):
            out[field] = getattr(value, field)
    return out


def _coerce_date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _normalize_option_type(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in {"call", "put"}:
        return raw
    if raw in {"c", "p"}:
        return "call" if raw == "c" else "put"
    if raw.startswith("call"):
        return "call"
    if raw.startswith("put"):
        return "put"
    return None


def contracts_to_df(raw_contracts: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contract in raw_contracts or []:
        data = _contract_to_dict(contract)
        contract_symbol = data.get("contractSymbol") or data.get("symbol") or data.get("option_symbol")
        underlying = (
            data.get("underlying_symbol")
            or data.get("underlyingSymbol")
            or data.get("underlying")
            or data.get("root_symbol")
        )
        expiry = data.get("expiration_date") or data.get("expiry") or data.get("expiration")
        option_type = data.get("option_type") or data.get("optionType") or data.get("type")
        strike = data.get("strike_price") or data.get("strike")
        multiplier = data.get("multiplier")
        open_interest = data.get("open_interest") or data.get("openInterest")
        open_interest_date = data.get("open_interest_date") or data.get("openInterestDate")
        close_price = data.get("close_price") or data.get("closePrice")
        close_price_date = data.get("close_price_date") or data.get("closePriceDate")

        row = {
            "contractSymbol": contract_symbol,
            "underlying": to_repo_symbol(str(underlying)) if underlying else None,
            "expiry": _coerce_date_string(expiry),
            "optionType": _normalize_option_type(option_type),
            "strike": _as_float(strike),
            "multiplier": _as_int(multiplier),
            "openInterest": _as_int(open_interest),
            "openInterestDate": _coerce_date_string(open_interest_date),
            "closePrice": _as_float(close_price),
            "closePriceDate": _coerce_date_string(close_price_date),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    required = [
        "contractSymbol",
        "underlying",
        "expiry",
        "optionType",
        "strike",
        "multiplier",
        "openInterest",
        "openInterestDate",
        "closePrice",
        "closePriceDate",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[required]

    if not df.empty and "contractSymbol" in df.columns:
        def _needs_fill(value: Any) -> bool:
            try:
                if value is None or pd.isna(value):
                    return True
            except Exception:  # noqa: BLE001
                return value is None
            if isinstance(value, str) and not value.strip():
                return True
            return False

        for idx, raw_symbol in df["contractSymbol"].items():
            if not raw_symbol or not isinstance(raw_symbol, str):
                continue
            parsed = parse_contract_symbol(raw_symbol)
            if parsed is None:
                continue
            if _needs_fill(df.at[idx, "expiry"]):
                df.at[idx, "expiry"] = parsed.expiry.isoformat()
            if _needs_fill(df.at[idx, "optionType"]):
                df.at[idx, "optionType"] = parsed.option_type
            if _needs_fill(df.at[idx, "strike"]):
                df.at[idx, "strike"] = parsed.strike
            if _needs_fill(df.at[idx, "underlying"]):
                df.at[idx, "underlying"] = parsed.underlying_norm or parsed.underlying

        df = df.drop_duplicates(subset=["contractSymbol"], keep="last")

    return df


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

    def get_stock_bars(
        self,
        symbol: str,
        *,
        start: date | datetime | None,
        end: date | datetime | None,
        interval: str,
        adjustment: str,
    ) -> pd.DataFrame:
        alpaca_symbol = to_alpaca_symbol(symbol)
        if not alpaca_symbol:
            raise DataFetchError(f"Invalid symbol: {symbol}")

        timeframe = _resolve_timeframe(interval)
        start_dt = _coerce_datetime(start, end_of_day=False)
        end_dt = _coerce_datetime(end, end_of_day=True)
        if end_dt is None:
            end_dt = datetime.now(timezone.utc) - timedelta(minutes=self._recent_bars_buffer_minutes)
        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise DataFetchError("End date is before start date for Alpaca bars request.")

        request_cls = _load_stock_bars_request()
        try:
            if request_cls is not None:
                request = request_cls(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt,
                    adjustment=adjustment,
                )
                payload = self.stock_client.get_stock_bars(request)
            else:
                payload = self.stock_client.get_stock_bars(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt,
                    adjustment=adjustment,
                )
        except DataFetchError:
            # Preserve higher-level errors (missing credentials, missing SDK, etc.).
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(
                f"Failed to fetch Alpaca stock bars for {alpaca_symbol} ({interval})."
            ) from exc

        df = _bars_to_dataframe(payload, alpaca_symbol)
        return _normalize_stock_bars(df, symbol=alpaca_symbol)

    def list_option_contracts(
        self,
        underlying: str,
        *,
        exp_gte: date | None = None,
        exp_lte: date | None = None,
        limit: int | None = None,
        page_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        alpaca_symbol = to_alpaca_symbol(underlying)
        if not alpaca_symbol:
            raise DataFetchError(f"Invalid underlying symbol: {underlying}")

        client = self.trading_client
        method = getattr(client, "get_option_contracts", None) or getattr(client, "get_option_contract", None)
        if method is None:
            raise DataFetchError("Alpaca trading client missing get_option_contracts.")

        request_cls = _load_option_contracts_request()
        page_token: str | None = None
        page_count = 0
        out: list[dict[str, Any]] = []

        while True:
            page_count += 1
            if page_limit is not None and page_count > page_limit:
                raise DataFetchError("Exceeded Alpaca option contracts page limit.")

            kwargs = {
                "underlying_symbol": alpaca_symbol,
                "underlying_symbols": [alpaca_symbol],
                "underlying": alpaca_symbol,
                "symbol": alpaca_symbol,
                "expiration_date_gte": exp_gte,
                "expiration_date_lte": exp_lte,
                "exp_gte": exp_gte,
                "exp_lte": exp_lte,
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
                payload = method(**_filter_kwargs(method, kwargs))

            contracts, next_token = _extract_contracts_page(payload)
            if contracts:
                out.extend(_contract_to_dict(c) for c in contracts)
            if not next_token:
                break
            page_token = str(next_token)

        return out

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
    "to_alpaca_symbol",
    "to_repo_symbol",
]
