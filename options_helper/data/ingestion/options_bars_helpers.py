from __future__ import annotations

from datetime import date, datetime, time
import inspect
from typing import Any

import pandas as pd

from options_helper.analysis.osi import normalize_underlying, parse_contract_symbol


def contract_symbol_from_raw(raw: dict[str, Any]) -> str | None:
    for key in ("contractSymbol", "symbol", "option_symbol", "contract"):
        val = raw.get(key)
        if val:
            return str(val).strip().upper()
    return None


def year_windows(exp_start: date, exp_end: date) -> list[tuple[int, date, date]]:
    windows: list[tuple[int, date, date]] = []
    for year in range(exp_end.year, exp_start.year - 1, -1):
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        window_start = max(exp_start, start)
        window_end = min(exp_end, end)
        if window_end < window_start:
            continue
        windows.append((year, window_start, window_end))
    return windows


def coerce_expiry(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if parsed is pd.NaT:
        return None
    if isinstance(parsed, pd.Timestamp):
        parsed = parsed.to_pydatetime()
    if isinstance(parsed, datetime):
        return parsed.date()
    return None


def expiry_from_contract_symbol(symbol: Any) -> date | None:
    if symbol is None:
        return None
    parsed = parse_contract_symbol(str(symbol))
    return parsed.expiry if parsed else None


def normalize_contracts_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["contractSymbol", "underlying", "expiry", "optionType", "strike", "multiplier"]
        )
    out = df.copy()
    if "contractSymbol" in out.columns:
        out["contractSymbol"] = out["contractSymbol"].map(
            lambda v: str(v).strip().upper() if v is not None else None
        )
    if "underlying" in out.columns:
        out["underlying"] = out["underlying"].map(
            lambda v: normalize_underlying(v) if v is not None else None
        )
    return out


def coverage_satisfies(meta: dict[str, Any] | None, *, start: date, end: date) -> bool:
    if not meta:
        return False
    status = str(meta.get("status") or "").strip().lower()
    if status not in {"ok", "partial"}:
        return False

    def _coerce_dt(value: Any) -> datetime | None:
        if value is None:
            return None
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:  # noqa: BLE001
            return None
        if parsed is pd.NaT:
            return None
        if isinstance(parsed, pd.Timestamp):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            return parsed.replace(tzinfo=None)
        return None

    end_ts = _coerce_dt(meta.get("end_ts") or meta.get("end"))
    if end_ts is None:
        return False

    desired_end = datetime.combine(end, time.max)
    return end_ts >= desired_end


def coerce_meta_dt(meta: dict[str, Any] | None, *keys: str) -> datetime | None:
    if not meta:
        return None
    for key in keys:
        if not key:
            continue
        value = meta.get(key)
        if value is None:
            continue
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        if parsed is pd.NaT:
            continue
        if isinstance(parsed, pd.Timestamp):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            return parsed.replace(tzinfo=None)
    return None


def error_status(exc: Exception) -> str:
    msg = str(exc).lower()
    if "403" in msg or "forbidden" in msg:
        return "forbidden"
    if "402" in msg or "payment required" in msg:
        return "forbidden"
    if "404" in msg or "not found" in msg:
        return "not_found"
    return "error"


def looks_like_timeout(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg


def looks_like_429(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def supports_max_rps_kw(method: Any) -> bool:
    return supports_kw(method, "max_requests_per_second")


def supports_kw(method: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return True
    return keyword in signature.parameters


def normalize_contract_status(value: str | None) -> str:
    raw = str(value or "active").strip().lower()
    if raw in {"active", "inactive", "all"}:
        return raw
    raise ValueError("contract_status must be one of: active, inactive, all")
