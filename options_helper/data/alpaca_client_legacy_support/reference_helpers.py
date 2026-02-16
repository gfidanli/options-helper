from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from options_helper.data.alpaca_symbols import to_repo_symbol

from .payload_helpers import _get_field


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
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
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
