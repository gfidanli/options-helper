from __future__ import annotations

import inspect
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import pandas as pd

from options_helper.data.market_types import DataFetchError


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
    df = getattr(payload, "df", None)
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
    df = getattr(payload, "df", None)
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
    columns = ["contractSymbol", "timestamp", "volume", "vwap", "trade_count"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

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
        key = col if col in rename_map else str(col).lower()
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})

    if "contractSymbol" not in out.columns:
        return pd.DataFrame(columns=columns)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    for col in ("volume", "vwap", "trade_count"):
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.NA

    out = out.dropna(subset=["contractSymbol"]).copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    if "timestamp" in out.columns:
        out = out.sort_values("timestamp", na_position="last")
        out = out.groupby("contractSymbol", as_index=False).tail(1)
    else:
        out = out.drop_duplicates(subset=["contractSymbol"], keep="last")

    return out[columns].reset_index(drop=True)


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    if response is None:
        return None
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
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
        return parsed if parsed >= 0 else None
    except ValueError:
        pass

    try:
        parsed_dt = parsedate_to_datetime(raw)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        delta = (parsed_dt - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)
    except Exception:  # noqa: BLE001
        return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    for attr in ("retry_after", "retry_after_seconds", "retry_after_sec"):
        parsed = _parse_retry_after(getattr(exc, attr, None))
        if parsed is not None:
            return parsed

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:  # noqa: BLE001
        value = None
    return _parse_retry_after(value)


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


def ensure_required_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataFetchError(f"{context} missing columns: {', '.join(missing)}")
