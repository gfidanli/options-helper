from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from options_helper.data.alpaca_symbols import to_repo_symbol


@dataclass(frozen=True)
class NormalizedEvent:
    dataset: str
    symbol: str
    row: dict[str, Any]


def _get_field(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj.get(name)
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "to_pydatetime"):
        try:
            dt = value.to_pydatetime()
        except Exception:  # noqa: BLE001
            dt = None
    elif isinstance(value, (int, float)):
        val = float(value)
        if val > 1e15:
            dt = datetime.fromtimestamp(val / 1e9, tz=timezone.utc)
        elif val > 1e12:
            dt = datetime.fromtimestamp(val / 1e3, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
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
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_symbol(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return raw.upper()


def _coerce_exchange(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    return raw or None


def _coerce_conditions(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        return raw or None
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items) if items else None
    raw = str(value).strip()
    return raw or None


def _normalize_stock_symbol(event: Any) -> str | None:
    symbol = _coerce_symbol(_get_field(event, ("symbol", "S", "sym", "ticker")))
    if not symbol:
        return None
    return to_repo_symbol(symbol)


def _normalize_option_symbol(event: Any) -> str | None:
    return _coerce_symbol(
        _get_field(event, ("symbol", "S", "contract_symbol", "contractSymbol", "option_symbol"))
    )


def normalize_stock_bar(event: Any) -> NormalizedEvent | None:
    symbol = _normalize_stock_symbol(event)
    ts = _coerce_timestamp(_get_field(event, ("timestamp", "t", "time", "ts")))
    if not symbol or ts is None:
        return None
    row = {
        "timestamp": ts,
        "open": _coerce_float(_get_field(event, ("open", "o"))),
        "high": _coerce_float(_get_field(event, ("high", "h"))),
        "low": _coerce_float(_get_field(event, ("low", "l"))),
        "close": _coerce_float(_get_field(event, ("close", "c"))),
        "volume": _coerce_int(_get_field(event, ("volume", "v"))),
        "trade_count": _coerce_int(_get_field(event, ("trade_count", "tradeCount", "n"))),
        "vwap": _coerce_float(_get_field(event, ("vwap", "vw"))),
    }
    return NormalizedEvent(dataset="stock_bars", symbol=symbol, row=row)


def normalize_option_bar(event: Any) -> NormalizedEvent | None:
    symbol = _normalize_option_symbol(event)
    ts = _coerce_timestamp(_get_field(event, ("timestamp", "t", "time", "ts")))
    if not symbol or ts is None:
        return None
    row = {
        "timestamp": ts,
        "open": _coerce_float(_get_field(event, ("open", "o"))),
        "high": _coerce_float(_get_field(event, ("high", "h"))),
        "low": _coerce_float(_get_field(event, ("low", "l"))),
        "close": _coerce_float(_get_field(event, ("close", "c"))),
        "volume": _coerce_int(_get_field(event, ("volume", "v"))),
        "trade_count": _coerce_int(_get_field(event, ("trade_count", "tradeCount", "n"))),
        "vwap": _coerce_float(_get_field(event, ("vwap", "vw"))),
    }
    return NormalizedEvent(dataset="option_bars", symbol=symbol, row=row)


def normalize_stock_quote(event: Any) -> NormalizedEvent | None:
    symbol = _normalize_stock_symbol(event)
    ts = _coerce_timestamp(_get_field(event, ("timestamp", "t", "time", "ts")))
    if not symbol or ts is None:
        return None
    row = {
        "timestamp": ts,
        "bid_price": _coerce_float(_get_field(event, ("bid_price", "bidPrice", "bp", "bid"))),
        "ask_price": _coerce_float(_get_field(event, ("ask_price", "askPrice", "ap", "ask"))),
        "bid_size": _coerce_int(_get_field(event, ("bid_size", "bidSize", "bs"))),
        "ask_size": _coerce_int(_get_field(event, ("ask_size", "askSize", "as"))),
        "exchange": _coerce_exchange(_get_field(event, ("exchange", "x", "bx", "ax"))),
    }
    return NormalizedEvent(dataset="stock_quotes", symbol=symbol, row=row)


def normalize_option_quote(event: Any) -> NormalizedEvent | None:
    symbol = _normalize_option_symbol(event)
    ts = _coerce_timestamp(_get_field(event, ("timestamp", "t", "time", "ts")))
    if not symbol or ts is None:
        return None
    row = {
        "timestamp": ts,
        "bid_price": _coerce_float(_get_field(event, ("bid_price", "bidPrice", "bp", "bid"))),
        "ask_price": _coerce_float(_get_field(event, ("ask_price", "askPrice", "ap", "ask"))),
        "bid_size": _coerce_int(_get_field(event, ("bid_size", "bidSize", "bs"))),
        "ask_size": _coerce_int(_get_field(event, ("ask_size", "askSize", "as"))),
        "exchange": _coerce_exchange(_get_field(event, ("exchange", "x", "bx", "ax"))),
    }
    return NormalizedEvent(dataset="option_quotes", symbol=symbol, row=row)


def normalize_stock_trade(event: Any) -> NormalizedEvent | None:
    symbol = _normalize_stock_symbol(event)
    ts = _coerce_timestamp(_get_field(event, ("timestamp", "t", "time", "ts")))
    if not symbol or ts is None:
        return None
    row = {
        "timestamp": ts,
        "price": _coerce_float(_get_field(event, ("price", "p", "trade_price", "last_price"))),
        "size": _coerce_int(_get_field(event, ("size", "s", "volume", "qty"))),
        "exchange": _coerce_exchange(_get_field(event, ("exchange", "x"))),
        "conditions": _coerce_conditions(_get_field(event, ("conditions", "c"))),
    }
    return NormalizedEvent(dataset="stock_trades", symbol=symbol, row=row)


def normalize_option_trade(event: Any) -> NormalizedEvent | None:
    symbol = _normalize_option_symbol(event)
    ts = _coerce_timestamp(_get_field(event, ("timestamp", "t", "time", "ts")))
    if not symbol or ts is None:
        return None
    row = {
        "timestamp": ts,
        "price": _coerce_float(_get_field(event, ("price", "p", "trade_price", "last_price"))),
        "size": _coerce_int(_get_field(event, ("size", "s", "volume", "qty"))),
        "exchange": _coerce_exchange(_get_field(event, ("exchange", "x"))),
        "conditions": _coerce_conditions(_get_field(event, ("conditions", "c"))),
    }
    return NormalizedEvent(dataset="option_trades", symbol=symbol, row=row)


__all__ = [
    "NormalizedEvent",
    "normalize_option_bar",
    "normalize_option_quote",
    "normalize_option_trade",
    "normalize_stock_bar",
    "normalize_stock_quote",
    "normalize_stock_trade",
]
