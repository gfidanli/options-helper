from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

from options_helper.data.market_types import DataFetchError


class _ContractsDeps(dict[str, Any]):
    pass


def list_option_contracts_impl(
    client_obj: Any,
    underlying: str | None,
    *,
    root_symbol: str | None,
    exp_gte: date | None,
    exp_lte: date | None,
    contract_status: str | None,
    limit: int | None,
    page_limit: int | None,
    max_requests_per_second: float | None,
    deps: _ContractsDeps,
) -> list[dict[str, Any]]:
    alpaca_symbol = deps["to_alpaca_symbol"](underlying) if underlying else ""
    if underlying and not alpaca_symbol:
        raise DataFetchError(f"Invalid underlying symbol: {underlying}")

    root_val = deps["to_alpaca_symbol"](root_symbol) if root_symbol else ""
    if root_symbol and not root_val:
        raise DataFetchError(f"Invalid root symbol: {root_symbol}")
    if not alpaca_symbol and not root_val:
        raise DataFetchError("Either underlying or root_symbol is required to list Alpaca option contracts.")

    method = _contracts_method(client_obj)
    request_cls = deps["load_option_contracts_request"]()
    status_values = _normalize_status_values(contract_status)
    next_allowed_at = 0.0
    min_interval_seconds = _rate_limit_interval(max_requests_per_second)
    max_retries = max(0, deps["coerce_int"](os.getenv("OH_ALPACA_MAX_RETRIES"), default=3))

    out: list[dict[str, Any]] = []
    for status in status_values:
        status_rows = _fetch_contract_status(
            method=method,
            request_cls=request_cls,
            alpaca_symbol=alpaca_symbol,
            root_val=root_val,
            exp_gte=exp_gte,
            exp_lte=exp_lte,
            status=status,
            limit=limit,
            page_limit=page_limit,
            max_retries=max_retries,
            min_interval_seconds=min_interval_seconds,
            next_allowed_at=next_allowed_at,
            deps=deps,
        )
        next_allowed_at = status_rows[0]
        out.extend(status_rows[1])

    return _dedupe_contracts(out, deps=deps)


def _fetch_contract_status(
    *,
    method,
    request_cls,
    alpaca_symbol,
    root_val,
    exp_gte,
    exp_lte,
    status,
    limit,
    page_limit,
    max_retries,
    min_interval_seconds,
    next_allowed_at,
    deps: _ContractsDeps,
) -> tuple[float, list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    page_token: str | None = None
    page_count = 0

    while True:
        page_count += 1
        if page_limit is not None and page_count > page_limit:
            raise DataFetchError("Exceeded Alpaca option contracts page limit.")

        kwargs = _contracts_kwargs(alpaca_symbol, root_val, exp_gte, exp_lte, status, limit, page_token)
        next_allowed_at = _wait_turn(next_allowed_at, min_interval_seconds)
        payload = _request_contract_page(
            method=method,
            request_cls=request_cls,
            kwargs=kwargs,
            max_retries=max_retries,
            deps=deps,
        )

        contracts, next_token = deps["extract_contracts_page"](payload)
        if contracts:
            out.extend(deps["contract_to_dict"](c) for c in contracts)
        if not next_token:
            break
        page_token = str(next_token)

    return next_allowed_at, out


def _request_contract_page(*, method, request_cls, kwargs, max_retries, deps: _ContractsDeps):
    payload = None
    if request_cls is not None:
        try:
            request_kwargs = deps["filter_kwargs"](request_cls, kwargs)
            request = request_cls(**request_kwargs)
            payload = _call_with_backoff(lambda: method(request), max_retries=max_retries, deps=deps)
        except TypeError:
            payload = None

    if payload is None:
        payload = _call_with_backoff(
            lambda: method(**deps["filter_kwargs"](method, kwargs)),
            max_retries=max_retries,
            deps=deps,
        )
    return payload


def _call_with_backoff(make_call, *, max_retries: int, deps: _ContractsDeps):
    for attempt in range(max_retries + 1):
        try:
            return make_call()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                should_retry = status_code >= 500 or status_code in (408, 429)
            else:
                should_retry = deps["is_rate_limit_error"](exc) or deps["is_timeout_error"](exc)
            if not should_retry:
                raise
            delay = 0.5 * (2**attempt)
            retry_after = deps["extract_retry_after_seconds"](exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            time.sleep(delay)
    raise DataFetchError("Failed to fetch Alpaca option contracts after retries.")


def _contracts_method(client_obj: Any):
    method = getattr(client_obj.trading_client, "get_option_contracts", None)
    if method is None:
        method = getattr(client_obj.trading_client, "get_option_contract", None)
    if method is None:
        raise DataFetchError("Alpaca trading client missing get_option_contracts.")
    return method


def _contracts_kwargs(
    alpaca_symbol: str,
    root_val: str,
    exp_gte: date | None,
    exp_lte: date | None,
    status: str,
    limit: int | None,
    page_token: str | None,
) -> dict[str, Any]:
    return {
        "underlying_symbol": alpaca_symbol or None,
        "underlying_symbols": [alpaca_symbol] if alpaca_symbol else None,
        "underlying": alpaca_symbol or None,
        "symbol": alpaca_symbol or None,
        "root_symbol": root_val or None,
        "root_symbols": [root_val] if root_val else None,
        "rootSymbol": root_val or None,
        "rootSymbols": [root_val] if root_val else None,
        "expiration_date_gte": exp_gte,
        "expiration_date_lte": exp_lte,
        "exp_gte": exp_gte,
        "exp_lte": exp_lte,
        "status": status,
        "limit": limit,
        "page_token": page_token,
    }


def _wait_turn(next_allowed_at: float, min_interval_seconds: float) -> float:
    if min_interval_seconds <= 0:
        return next_allowed_at
    now = time.monotonic()
    if next_allowed_at > now:
        time.sleep(next_allowed_at - now)
        now = time.monotonic()
    return now + min_interval_seconds


def _normalize_status_values(contract_status: str | None) -> tuple[str, ...]:
    status_raw = str(contract_status or "active").strip().lower()
    if status_raw not in {"active", "inactive", "all"}:
        raise DataFetchError("contract_status must be one of: active, inactive, all")
    if status_raw == "all":
        return ("active", "inactive")
    return (status_raw,)


def _rate_limit_interval(max_requests_per_second: float | None) -> float:
    if max_requests_per_second is None or max_requests_per_second <= 0:
        return 0.0
    return 1.0 / float(max_requests_per_second)


def _dedupe_contracts(items: list[dict[str, Any]], *, deps: _ContractsDeps) -> list[dict[str, Any]]:
    if len(items) <= 1:
        return items

    deduped_by_symbol: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for item in items:
        contract = deps["contract_to_dict"](item)
        symbol = contract.get("symbol") or contract.get("contractSymbol") or contract.get("option_symbol")
        key = str(symbol or "").strip().upper()
        if not key:
            unresolved.append(contract)
            continue
        deduped_by_symbol[key] = contract

    if not deduped_by_symbol:
        return unresolved
    return [*deduped_by_symbol.values(), *unresolved]
