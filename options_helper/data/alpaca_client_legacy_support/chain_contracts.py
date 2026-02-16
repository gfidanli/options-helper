from __future__ import annotations

from typing import Any

import pandas as pd

from options_helper.analysis.osi import parse_contract_symbol
from options_helper.data.alpaca_symbols import to_repo_symbol

from .payload_helpers import _get_field
from .reference_helpers import (
    _as_float,
    _as_int,
    _coerce_date_string,
    _coerce_timestamp_value,
    _contract_to_dict,
    _extract_chain_container,
    _looks_like_chain_map,
    _looks_like_snapshot,
    _normalize_option_type,
)


def option_chain_to_rows(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []

    container = _extract_chain_container(payload)
    items = _iter_chain_items(container)
    rows: list[dict[str, Any]] = []
    for symbol_key, snapshot in items:
        if snapshot is None:
            continue
        rows.append(_snapshot_to_row(snapshot, symbol_key))
    return rows


def contracts_to_df(raw_contracts: list[Any]) -> pd.DataFrame:
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
    rows = [_contract_row(_contract_to_dict(contract)) for contract in raw_contracts or []]
    df = pd.DataFrame(rows)
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[required]

    if not df.empty and "contractSymbol" in df.columns:
        _fill_contract_fields_from_symbol(df)
        df = df.drop_duplicates(subset=["contractSymbol"], keep="last")

    return df


def _iter_chain_items(container: Any) -> list[tuple[str | None, Any]]:
    if isinstance(container, dict):
        if _looks_like_chain_map(container):
            return [(str(key) if key is not None else None, val) for key, val in container.items()]
        if _looks_like_snapshot(container):
            return [(None, container)]
        return []
    if isinstance(container, (list, tuple)):
        return [(None, snapshot) for snapshot in container]
    return [(None, container)]


def _snapshot_to_row(snapshot: Any, symbol_key: str | None) -> dict[str, Any]:
    contract_symbol = (
        _get_field(snapshot, "symbol")
        or _get_field(snapshot, "contract_symbol")
        or _get_field(snapshot, "contractSymbol")
        or _get_field(snapshot, "option_symbol")
        or symbol_key
    )
    quote_time, bid, ask = _snapshot_quote_fields(snapshot)
    trade_time, last_price = _snapshot_trade_fields(snapshot)
    implied_volatility = _as_float(
        _get_field(snapshot, "implied_volatility")
        or _get_field(snapshot, "impliedVolatility")
        or _get_field(snapshot, "iv")
    )
    delta, gamma, theta, vega, rho = _snapshot_greeks(snapshot)
    open_interest = _as_int(_get_field(snapshot, "open_interest") or _get_field(snapshot, "openInterest"))
    volume = _as_int(_get_field(snapshot, "volume") or _get_field(snapshot, "vol"))

    return {
        "contractSymbol": str(contract_symbol) if contract_symbol is not None else None,
        "bid": bid,
        "ask": ask,
        "lastPrice": last_price,
        "lastTradeDate": _coerce_timestamp_value(trade_time),
        "quoteTime": _coerce_timestamp_value(quote_time),
        "tradeTime": _coerce_timestamp_value(trade_time),
        "impliedVolatility": implied_volatility,
        "iv_source": "alpaca_snapshot" if implied_volatility is not None else None,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "openInterest": open_interest,
        "volume": volume,
    }


def _snapshot_quote_fields(snapshot: Any) -> tuple[Any, float | None, float | None]:
    quote = _get_field(snapshot, "latest_quote") or _get_field(snapshot, "latestQuote") or _get_field(snapshot, "quote")
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
    return quote_time, bid, ask


def _snapshot_trade_fields(snapshot: Any) -> tuple[Any, float | None]:
    trade = _get_field(snapshot, "latest_trade") or _get_field(snapshot, "latestTrade") or _get_field(snapshot, "trade")
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
    return trade_time, last_price


def _snapshot_greeks(snapshot: Any) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    greeks = _get_field(snapshot, "greeks") or _get_field(snapshot, "latest_greeks") or _get_field(snapshot, "latestGreeks")
    delta = _as_float(_get_field(greeks, "delta"))
    gamma = _as_float(_get_field(greeks, "gamma"))
    theta = _as_float(_get_field(greeks, "theta"))
    vega = _as_float(_get_field(greeks, "vega"))
    rho = _as_float(_get_field(greeks, "rho"))
    return delta, gamma, theta, vega, rho


def _contract_row(data: dict[str, Any]) -> dict[str, Any]:
    contract_symbol = data.get("contractSymbol") or data.get("symbol") or data.get("option_symbol")
    underlying = data.get("underlying_symbol") or data.get("underlyingSymbol") or data.get("underlying") or data.get("root_symbol")
    expiry = data.get("expiration_date") or data.get("expiry") or data.get("expiration")
    option_type = data.get("option_type") or data.get("optionType") or data.get("type")
    strike = data.get("strike_price") or data.get("strike")
    multiplier = data.get("multiplier")
    open_interest = data.get("open_interest") or data.get("openInterest")
    open_interest_date = data.get("open_interest_date") or data.get("openInterestDate")
    close_price = data.get("close_price") or data.get("closePrice")
    close_price_date = data.get("close_price_date") or data.get("closePriceDate")

    return {
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


def _fill_contract_fields_from_symbol(df: pd.DataFrame) -> None:
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


def _needs_fill(value: Any) -> bool:
    try:
        if value is None or pd.isna(value):
            return True
    except Exception:  # noqa: BLE001
        return value is None
    return isinstance(value, str) and not value.strip()
