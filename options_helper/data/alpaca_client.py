from __future__ import annotations

import inspect
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
OptionChainRequest = None
OptionBarsRequest = None
CorporateActionsClient = None
CorporateActionsRequest = None
NewsClient = None
NewsRequest = None


def _clean_env(value: str | None) -> str:
    return (value or "").strip()


_DEFAULT_MARKET_TZ = "America/New_York"


def _load_market_tz_name() -> str:
    return _clean_env(os.getenv("OH_ALPACA_MARKET_TZ")) or _DEFAULT_MARKET_TZ


def _load_market_tz() -> ZoneInfo:
    name = _load_market_tz_name()
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001
        return ZoneInfo(_DEFAULT_MARKET_TZ)


def _market_day_bounds(day: date, market_tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(day, datetime.min.time()).replace(tzinfo=market_tz)
    end_local = datetime.combine(day, datetime.max.time()).replace(tzinfo=market_tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


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


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


def _parse_env_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export "):
        raw = raw[len("export ") :].strip()
    if "=" not in raw:
        return None
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _strip_quotes(value)


def _load_env_file(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except Exception:  # noqa: BLE001
        return

    for line in content.splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        # Only set known Alpaca/repo knobs and never override the current process environment.
        if not (key.startswith("APCA_") or key.startswith("OH_ALPACA_")):
            continue
        if key in os.environ:
            continue
        os.environ[key] = value


def _maybe_load_alpaca_env() -> None:
    override = _clean_env(os.getenv("OH_ALPACA_ENV_FILE"))
    if override:
        _load_env_file(Path(override))
        return
    # Default locations (local-only; should be gitignored if it contains secrets).
    _load_env_file(Path("config/alpaca.env"))
    _load_env_file(Path(".env"))


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


def _filter_kwargs(target, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    allowed = {name for name in sig.parameters if name != "self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


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


def _option_bars_to_dataframe(payload: Any) -> pd.DataFrame:
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
        if len(payload) == 1:
            candidate = next(iter(payload.values()))
            try:
                return pd.DataFrame(candidate)
            except Exception:  # noqa: BLE001
                return pd.DataFrame()
        try:
            return pd.DataFrame(payload)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    try:
        return pd.DataFrame(payload)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _extract_option_bars_page_token(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        token = (
            payload.get("next_page_token")
            or payload.get("next_token")
            or payload.get("nextPageToken")
            or payload.get("next")
        )
        return str(token) if token else None
    for attr in ("next_page_token", "next_token", "nextPageToken", "next"):
        token = getattr(payload, attr, None)
        if token:
            return str(token)
    return None


def _normalize_option_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["contractSymbol", "timestamp", "volume", "vwap", "trade_count"])

    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})

    rename_map = {
        "symbol": "contractSymbol",
        "option_symbol": "contractSymbol",
        "contract_symbol": "contractSymbol",
        "contractSymbol": "contractSymbol",
        "t": "timestamp",
        "time": "timestamp",
        "timestamp": "timestamp",
        "v": "volume",
        "volume": "volume",
        "vw": "vwap",
        "vwap": "vwap",
        "n": "trade_count",
        "trade_count": "trade_count",
        "tradeCount": "trade_count",
    }
    for col in list(out.columns):
        key = col
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})
        else:
            lowered = str(col).lower()
            if lowered in rename_map:
                out = out.rename(columns={col: rename_map[lowered]})

    if "contractSymbol" not in out.columns:
        return pd.DataFrame(columns=["contractSymbol", "timestamp", "volume", "vwap", "trade_count"])

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    for col in ("volume", "vwap", "trade_count"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    out = out.dropna(subset=["contractSymbol"]).copy()
    if out.empty:
        return pd.DataFrame(columns=["contractSymbol", "timestamp", "volume", "vwap", "trade_count"])

    if "timestamp" in out.columns:
        out = out.sort_values("timestamp", na_position="last")
        latest = out.groupby("contractSymbol", as_index=False).tail(1)
    else:
        latest = out.drop_duplicates(subset=["contractSymbol"], keep="last")

    return latest[["contractSymbol", "timestamp", "volume", "vwap", "trade_count"]].reset_index(drop=True)


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status", "code"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def _parse_retry_after(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = float(raw)
            return parsed if parsed >= 0 else None
        except ValueError:
            try:
                parsed_dt = parsedate_to_datetime(raw)
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                delta = (parsed_dt - now).total_seconds()
                return max(delta, 0.0)
            except Exception:  # noqa: BLE001
                return None
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    for attr in ("retry_after", "retry_after_seconds", "retry_after_sec"):
        value = getattr(exc, attr, None)
        parsed = _parse_retry_after(value)
        if parsed is not None:
            return parsed
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if headers is not None:
        try:
            value = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:  # noqa: BLE001
            value = None
        parsed = _parse_retry_after(value)
        if parsed is not None:
            return parsed
    return None


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


def _is_rate_limit_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return "rate" in message or "429" in message


def _raise_missing_optional_client(feature: str) -> None:
    message = f"Alpaca {feature} requires alpaca-py. Install with `pip install -e \".[alpaca]\"`."
    if _ALPACA_IMPORT_ERROR is not None:
        message = f"{message} (import error: {_ALPACA_IMPORT_ERROR})"
    raise DataFetchError(message)


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
        "vw": "VWAP",
        "vwap": "VWAP",
        "n": "Trade Count",
        "trade_count": "Trade Count",
        "tradecount": "Trade Count",
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


def _normalize_intraday_stock_bars(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        )

    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})

    rename_map = {
        "symbol": "symbol",
        "t": "timestamp",
        "time": "timestamp",
        "timestamp": "timestamp",
        "o": "open",
        "open": "open",
        "h": "high",
        "high": "high",
        "l": "low",
        "low": "low",
        "c": "close",
        "close": "close",
        "v": "volume",
        "volume": "volume",
        "n": "trade_count",
        "trade_count": "trade_count",
        "tradecount": "trade_count",
        "vw": "vwap",
        "vwap": "vwap",
    }
    for col in list(out.columns):
        key = str(col).lower()
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})

    if "symbol" in out.columns:
        sym_col = out["symbol"].astype(str).str.upper()
        target = symbol.upper()
        out = out[sym_col == target].copy()
        out = out.drop(columns=["symbol"])

    if "timestamp" not in out.columns:
        raise DataFetchError("Alpaca intraday bars missing timestamp column.")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"]).copy()

    required = ["open", "high", "low", "close", "volume"]
    missing = sorted([col for col in required if col not in out.columns])
    if missing:
        raise DataFetchError(f"Alpaca intraday bars missing columns: {', '.join(missing)}")

    for col in required + ["trade_count", "vwap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    out = out.sort_values("timestamp", na_position="last")
    return out[["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]].reset_index(drop=True)


def _normalize_intraday_option_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "contractSymbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "vwap",
            ]
        )

    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})

    rename_map = {
        "symbol": "contractSymbol",
        "option_symbol": "contractSymbol",
        "contract_symbol": "contractSymbol",
        "contractsymbol": "contractSymbol",
        "t": "timestamp",
        "time": "timestamp",
        "timestamp": "timestamp",
        "o": "open",
        "open": "open",
        "h": "high",
        "high": "high",
        "l": "low",
        "low": "low",
        "c": "close",
        "close": "close",
        "v": "volume",
        "volume": "volume",
        "n": "trade_count",
        "trade_count": "trade_count",
        "tradecount": "trade_count",
        "vw": "vwap",
        "vwap": "vwap",
    }
    for col in list(out.columns):
        key = str(col).lower()
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})

    if "contractSymbol" not in out.columns:
        raise DataFetchError("Alpaca intraday option bars missing contract symbol column.")
    if "timestamp" not in out.columns:
        raise DataFetchError("Alpaca intraday option bars missing timestamp column.")

    out["contractSymbol"] = out["contractSymbol"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["contractSymbol", "timestamp"]).copy()

    required = ["open", "high", "low", "close", "volume"]
    missing = sorted([col for col in required if col not in out.columns])
    if missing:
        raise DataFetchError(f"Alpaca intraday option bars missing columns: {', '.join(missing)}")

    for col in required + ["trade_count", "vwap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    out = out.sort_values(["contractSymbol", "timestamp"], na_position="last")
    return out[
        ["contractSymbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    ].reset_index(drop=True)


def _normalize_option_bars_daily_full(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_intraday_option_bars(df)
    columns = ["contractSymbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
    if out is None or out.empty:
        return pd.DataFrame(columns=columns)
    out = out.rename(columns={"timestamp": "ts"})
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].reset_index(drop=True)


def _normalize_corporate_action(item: Any, *, default_symbol: str | None = None) -> dict[str, Any]:
    symbol = (
        _get_field(item, "symbol")
        or _get_field(item, "ticker")
        or _get_field(item, "asset_id")
        or default_symbol
    )
    action_type = (
        _get_field(item, "type")
        or _get_field(item, "action_type")
        or _get_field(item, "event_type")
        or _get_field(item, "corporate_action_type")
    )
    return {
        "type": (str(action_type).strip().lower() if action_type else None),
        "symbol": to_repo_symbol(str(symbol)) if symbol else None,
        "ex_date": _coerce_date_string(
            _get_field(item, "ex_date")
            or _get_field(item, "exDate")
            or _get_field(item, "exdate")
        ),
        "record_date": _coerce_date_string(
            _get_field(item, "record_date")
            or _get_field(item, "recordDate")
            or _get_field(item, "recorddate")
        ),
        "pay_date": _coerce_date_string(
            _get_field(item, "pay_date")
            or _get_field(item, "payment_date")
            or _get_field(item, "paymentDate")
            or _get_field(item, "payDate")
        ),
        "ratio": _as_float(
            _get_field(item, "ratio")
            or _get_field(item, "split_ratio")
            or _get_field(item, "splitRatio")
        ),
        "cash_amount": _as_float(
            _get_field(item, "cash_amount")
            or _get_field(item, "cashAmount")
            or _get_field(item, "dividend")
            or _get_field(item, "amount")
        ),
        "raw": _contract_to_dict(item) if not isinstance(item, dict) else item,
    }


def _normalize_news_item(item: Any, *, include_content: bool) -> dict[str, Any]:
    raw_symbols = _get_field(item, "symbols") or _get_field(item, "symbol") or _get_field(item, "tickers")
    symbols: list[str] = []
    if isinstance(raw_symbols, (list, tuple)):
        symbols = [to_repo_symbol(str(sym)) for sym in raw_symbols if sym]
    elif isinstance(raw_symbols, str):
        symbols = [to_repo_symbol(sym) for sym in raw_symbols.split(",") if sym.strip()]

    created_at = (
        _get_field(item, "created_at")
        or _get_field(item, "createdAt")
        or _get_field(item, "published_at")
        or _get_field(item, "publishedAt")
        or _get_field(item, "timestamp")
    )

    payload = {
        "id": _get_field(item, "id") or _get_field(item, "news_id") or _get_field(item, "newsId"),
        "created_at": _coerce_timestamp_value(created_at),
        "headline": _get_field(item, "headline") or _get_field(item, "title"),
        "summary": _get_field(item, "summary") or _get_field(item, "description"),
        "source": _get_field(item, "source") or _get_field(item, "source_name") or _get_field(item, "sourceName"),
        "symbols": [sym for sym in symbols if sym],
    }

    if include_content:
        payload["content"] = _get_field(item, "content") or _get_field(item, "body") or _get_field(
            item, "story"
        )
    return payload


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


def _extract_corporate_actions_page(payload: Any) -> tuple[list[Any], str | None]:
    if payload is None:
        return [], None
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        actions = (
            payload.get("corporate_actions")
            or payload.get("actions")
            or payload.get("data")
            or payload.get("corporateActions")
            or []
        )
        token = (
            payload.get("next_page_token")
            or payload.get("next_token")
            or payload.get("nextPageToken")
            or payload.get("next")
        )
        return list(actions or []), token
    for attr in ("corporate_actions", "actions", "data"):
        if hasattr(payload, attr):
            actions = getattr(payload, attr) or []
            token = getattr(payload, "next_page_token", None) or getattr(payload, "next_token", None)
            return list(actions), token
    return [], None


def _extract_news_page(payload: Any) -> tuple[list[Any], str | None]:
    if payload is None:
        return [], None
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        news_items = payload.get("news") or payload.get("data") or payload.get("items") or []
        token = (
            payload.get("next_page_token")
            or payload.get("next_token")
            or payload.get("nextPageToken")
            or payload.get("next")
        )
        return list(news_items or []), token
    for attr in ("news", "data", "items"):
        if hasattr(payload, attr):
            news_items = getattr(payload, attr) or []
            token = getattr(payload, "next_page_token", None) or getattr(payload, "next_token", None)
            return list(news_items), token
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


def _coerce_timestamp_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time()).replace(tzinfo=timezone.utc)
        return dt.isoformat()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        raw = value.strip()
        return raw or None
    try:
        return str(value)
    except Exception:  # noqa: BLE001
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


def _looks_like_snapshot(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        keys = {str(k) for k in value.keys()}
        markers = {
            "latest_quote",
            "latestQuote",
            "latest_trade",
            "latestTrade",
            "greeks",
            "implied_volatility",
            "impliedVolatility",
            "iv",
        }
        return bool(keys & markers)
    for attr in ("latest_quote", "latest_trade", "greeks", "implied_volatility", "impliedVolatility", "iv"):
        if hasattr(value, attr):
            return True
    return False


def _looks_like_chain_map(value: dict[str, Any]) -> bool:
    if not value:
        return False
    for item in value.values():
        if _looks_like_snapshot(item):
            return True
    return False


def _extract_chain_container(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("data", "chain", "snapshots", "option_chain", "optionChain"):
            if key in payload:
                return payload[key]
        return payload
    for attr in ("data", "chain", "snapshots", "option_chain", "optionChain"):
        candidate = getattr(payload, attr, None)
        if candidate is not None:
            return candidate
    return payload


def option_chain_to_rows(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []

    container = _extract_chain_container(payload)
    items: list[tuple[str | None, Any]] = []

    if isinstance(container, dict):
        if _looks_like_chain_map(container):
            items = [(str(key) if key is not None else None, val) for key, val in container.items()]
        elif _looks_like_snapshot(container):
            items = [(None, container)]
        else:
            return []
    elif isinstance(container, (list, tuple)):
        items = [(None, snapshot) for snapshot in container]
    else:
        items = [(None, container)]

    rows: list[dict[str, Any]] = []
    for symbol_key, snapshot in items:
        if snapshot is None:
            continue

        contract_symbol = (
            _get_field(snapshot, "symbol")
            or _get_field(snapshot, "contract_symbol")
            or _get_field(snapshot, "contractSymbol")
            or _get_field(snapshot, "option_symbol")
            or symbol_key
        )
        contract_symbol = str(contract_symbol) if contract_symbol is not None else None

        quote = (
            _get_field(snapshot, "latest_quote")
            or _get_field(snapshot, "latestQuote")
            or _get_field(snapshot, "quote")
        )
        quote_time = (
            _get_field(quote, "timestamp")
            or _get_field(quote, "t")
            or _get_field(quote, "quote_timestamp")
            or _get_field(quote, "quoteTimestamp")
        )
        bid = _as_float(
            _get_field(quote, "bid_price")
            or _get_field(quote, "bp")
            or _get_field(quote, "bid")
            or _get_field(quote, "bidPrice")
        )
        ask = _as_float(
            _get_field(quote, "ask_price")
            or _get_field(quote, "ap")
            or _get_field(quote, "ask")
            or _get_field(quote, "askPrice")
        )

        trade = (
            _get_field(snapshot, "latest_trade")
            or _get_field(snapshot, "latestTrade")
            or _get_field(snapshot, "trade")
        )
        last_price = _as_float(
            _get_field(trade, "price")
            or _get_field(trade, "p")
            or _get_field(trade, "last_price")
            or _get_field(trade, "lastPrice")
        )
        trade_time = (
            _get_field(trade, "timestamp")
            or _get_field(trade, "t")
            or _get_field(trade, "trade_timestamp")
            or _get_field(trade, "tradeTimestamp")
            or _get_field(snapshot, "lastTradeDate")
        )

        implied_volatility = _as_float(
            _get_field(snapshot, "implied_volatility")
            or _get_field(snapshot, "impliedVolatility")
            or _get_field(snapshot, "iv")
        )
        iv_source = "alpaca_snapshot" if implied_volatility is not None else None

        greeks = (
            _get_field(snapshot, "greeks")
            or _get_field(snapshot, "latest_greeks")
            or _get_field(snapshot, "latestGreeks")
        )
        delta = _as_float(_get_field(greeks, "delta"))
        gamma = _as_float(_get_field(greeks, "gamma"))
        theta = _as_float(_get_field(greeks, "theta"))
        vega = _as_float(_get_field(greeks, "vega"))
        rho = _as_float(_get_field(greeks, "rho"))

        open_interest = _as_int(
            _get_field(snapshot, "open_interest") or _get_field(snapshot, "openInterest")
        )
        volume = _as_int(_get_field(snapshot, "volume") or _get_field(snapshot, "vol"))

        rows.append(
            {
                "contractSymbol": contract_symbol,
                "bid": bid,
                "ask": ask,
                "lastPrice": last_price,
                "lastTradeDate": _coerce_timestamp_value(trade_time),
                "quoteTime": _coerce_timestamp_value(quote_time),
                "tradeTime": _coerce_timestamp_value(trade_time),
                "impliedVolatility": implied_volatility,
                "iv_source": iv_source,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
                "openInterest": open_interest,
                "volume": volume,
            }
        )

    return rows


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

    @property
    def corporate_actions_client(self):
        if self._corporate_actions_client is None:
            self._ensure_credentials()
            client_cls = _load_corporate_actions_client()
            if client_cls is None:
                _raise_missing_optional_client("corporate actions")
            self._corporate_actions_client = self._construct_client(client_cls, **self._credential_kwargs())
        return self._corporate_actions_client

    @property
    def news_client(self):
        if self._news_client is None:
            self._ensure_credentials()
            client_cls = _load_news_client()
            if client_cls is None:
                _raise_missing_optional_client("news")
            self._news_client = self._construct_client(client_cls, **self._credential_kwargs())
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
        alpaca_symbol = to_alpaca_symbol(symbol)
        if not alpaca_symbol:
            raise DataFetchError(f"Invalid symbol: {symbol}")

        timeframe = _resolve_timeframe(interval)
        start_dt = _coerce_datetime(start, end_of_day=False)
        end_dt = self.effective_end(end, end_of_day=True)
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

    def get_stock_bars_intraday(
        self,
        symbol: str,
        *,
        day: date,
        timeframe: str = "1Min",
        feed: str | None = None,
        adjustment: str = "raw",
    ) -> pd.DataFrame:
        alpaca_symbol = to_alpaca_symbol(symbol)
        if not alpaca_symbol:
            raise DataFetchError(f"Invalid symbol: {symbol}")

        market_tz = _load_market_tz()
        market_today = datetime.now(market_tz).date()
        if day > market_today:
            raise DataFetchError(f"Requested intraday day is in the future: {day.isoformat()}")

        timeframe_val = _resolve_timeframe(timeframe)
        start_dt, day_end_dt = _market_day_bounds(day, market_tz)
        if day == market_today:
            end_dt = self.effective_end(None, end_of_day=True)
            if end_dt is None:
                end_dt = day_end_dt
            end_dt = min(end_dt, day_end_dt)
            if end_dt < start_dt:
                end_dt = start_dt
        else:
            end_dt = day_end_dt

        request_cls = _load_stock_bars_request()
        kwargs = {
            "symbol_or_symbols": alpaca_symbol,
            "timeframe": timeframe_val,
            "start": start_dt,
            "end": end_dt,
            "adjustment": adjustment,
            "feed": feed or self._stock_feed,
        }

        try:
            if request_cls is not None:
                request_kwargs = _filter_kwargs(request_cls, kwargs)
                request = request_cls(**request_kwargs)
                payload = self.stock_client.get_stock_bars(request)
            else:
                payload = self.stock_client.get_stock_bars(**_filter_kwargs(self.stock_client.get_stock_bars, kwargs))
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(
                f"Failed to fetch Alpaca intraday stock bars for {alpaca_symbol} ({timeframe})."
            ) from exc

        df = _bars_to_dataframe(payload, alpaca_symbol)
        return _normalize_intraday_stock_bars(df, symbol=alpaca_symbol)

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
        raw_symbols = [str(sym).strip() for sym in (symbols or []) if sym]
        unique_symbols = sorted({sym for sym in raw_symbols if sym})
        columns = ["contractSymbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
        if not unique_symbols:
            return pd.DataFrame(columns=columns)

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
        chunk_size = int(chunk_size) if chunk_size else 200
        if chunk_size <= 0:
            chunk_size = 200
        max_retries = int(max_retries) if max_retries else 1
        if max_retries < 1:
            max_retries = 1

        def _call_with_backoff(make_call):
            for attempt in range(max_retries):
                try:
                    return make_call()
                except Exception as exc:  # noqa: BLE001
                    if attempt >= max_retries - 1:
                        raise
                    status_code = _extract_status_code(exc)
                    retry_after = _extract_retry_after_seconds(exc)
                    is_rate_limit = status_code == 429 or _is_rate_limit_error(exc)
                    if status_code is not None:
                        if 400 <= status_code < 500 and status_code not in (408, 429):
                            raise
                        should_retry = status_code >= 500 or status_code in (408, 429)
                    else:
                        should_retry = is_rate_limit or _is_timeout_error(exc)
                    if not should_retry:
                        raise
                    base_delay = 0.5 * (2**attempt)
                    delay = base_delay
                    if is_rate_limit:
                        delay += random.uniform(0, base_delay * 0.25)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    time.sleep(delay)
            return None

        all_chunks: list[pd.DataFrame] = []

        for i in range(0, len(unique_symbols), chunk_size):
            chunk = unique_symbols[i : i + chunk_size]
            page_token: str | None = None
            page_count = 0

            while True:
                page_count += 1
                if page_limit is not None and page_count > page_limit:
                    raise DataFetchError("Exceeded Alpaca option bars page limit.")

                kwargs = {
                    "symbol_or_symbols": chunk,
                    "symbols": chunk,
                    "timeframe": timeframe,
                    "start": start_dt,
                    "end": end_dt,
                    "feed": feed_val,
                    "limit": chunk_size,
                    "page_token": page_token,
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
                    break

                df = _option_bars_to_dataframe(payload)
                norm = _normalize_option_bars_daily_full(df)
                if not norm.empty:
                    all_chunks.append(norm)

                page_token = _extract_option_bars_page_token(payload)
                if not page_token:
                    break

        if not all_chunks:
            return pd.DataFrame(columns=columns)

        combined = pd.concat(all_chunks, ignore_index=True)
        combined = combined.drop_duplicates(subset=["contractSymbol", "ts"], keep="last")
        combined = combined.sort_values(["contractSymbol", "ts"], na_position="last")
        for col in columns:
            if col not in combined.columns:
                combined[col] = pd.NA
        return combined[columns].reset_index(drop=True)

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
        raw_symbols = [str(sym).strip() for sym in (symbols or []) if sym]
        unique_symbols = sorted({sym for sym in raw_symbols if sym})
        if not unique_symbols:
            return pd.DataFrame(
                columns=[
                    "contractSymbol",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trade_count",
                    "vwap",
                ]
            )

        market_tz = _load_market_tz()
        market_today = datetime.now(market_tz).date()
        if day > market_today:
            raise DataFetchError(f"Requested intraday day is in the future: {day.isoformat()}")

        timeframe_val = _resolve_timeframe(timeframe)
        start_dt, day_end_dt = _market_day_bounds(day, market_tz)
        if day == market_today:
            end_dt = self.effective_end(None, end_of_day=True)
            if end_dt is None:
                end_dt = day_end_dt
            end_dt = min(end_dt, day_end_dt)
            if end_dt < start_dt:
                end_dt = start_dt
        else:
            end_dt = day_end_dt

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
                "timeframe": timeframe_val,
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
            norm = _normalize_intraday_option_bars(df)
            if not norm.empty:
                all_chunks.append(norm)

        if not all_chunks:
            return pd.DataFrame(
                columns=[
                    "contractSymbol",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trade_count",
                    "vwap",
                ]
            )

        combined = pd.concat(all_chunks, ignore_index=True)
        combined = combined.drop_duplicates(subset=["contractSymbol", "timestamp"], keep="last")
        return combined.reset_index(drop=True)

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
