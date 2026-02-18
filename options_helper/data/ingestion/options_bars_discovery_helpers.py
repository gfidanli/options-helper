from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time as time_mod
from typing import Any, Iterable, Mapping, Protocol, Sequence

from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df
from options_helper.data.ingestion.options_bars_helpers import (
    contract_symbol_from_raw as _contract_symbol_from_raw,
    looks_like_429 as _looks_like_429,
    looks_like_timeout as _looks_like_timeout,
    normalize_contracts_frame as _normalize_contracts_frame,
    supports_kw as _supports_kw,
)


@dataclass(frozen=True)
class DiscoveryScanResult:
    raw_contracts: list[dict[str, Any]]
    years_scanned: int
    empty_years: int
    status: str
    error: str | None
    calls: int
    error_count: int
    timeout_count: int
    rate_limit_429: int
    latencies_ms: list[float]


@dataclass(frozen=True)
class _DiscoveryWindowFetchResult:
    contracts: list[dict[str, Any]]
    calls: int
    error: str | None
    timeout_count: int
    rate_limit_429: int
    latency_ms: float


class _RateLimiter(Protocol):
    def wait_turn(self) -> None: ...


def list_option_contracts(
    client: AlpacaClient,
    *,
    underlying: str | None = None,
    root_symbol: str | None = None,
    exp_gte: date,
    exp_lte: date,
    limit: int | None,
    page_limit: int | None,
    max_requests_per_second: float | None,
    supports_max_rps_kw: bool,
    contract_status: str | None,
) -> list[dict[str, Any]]:
    method = getattr(client, "list_option_contracts")
    kwargs: dict[str, Any] = {
        "exp_gte": exp_gte,
        "exp_lte": exp_lte,
        "limit": limit,
        "page_limit": page_limit,
    }
    if contract_status:
        if _supports_kw(method, "contract_status"):
            kwargs["contract_status"] = contract_status
        elif _supports_kw(method, "status"):
            kwargs["status"] = contract_status
    if root_symbol:
        kwargs["root_symbol"] = root_symbol
    if supports_max_rps_kw and max_requests_per_second is not None:
        kwargs["max_requests_per_second"] = max_requests_per_second
    return method(underlying, **kwargs)


def build_scan_targets(
    *,
    underlyings: Iterable[str],
    root_symbols: Iterable[str] | None,
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = [("underlying", sym) for sym in underlyings]
    if root_symbols is not None:
        targets.extend([("root_symbol", sym) for sym in root_symbols])
    return targets


def scan_discovery_target(
    *,
    client: AlpacaClient,
    kind: str,
    token: str,
    windows: Sequence[tuple[int, date, date]],
    max_contracts: int | None, total_contracts: int,
    limit: int | None, page_limit: int | None,
    max_requests_per_second: float | None, supports_max_rps_kw: bool,
    resolved_contract_status: str | None,
    contracts_rate_limiter: _RateLimiter,
    fail_fast: bool, prefix: str | None, today: date,
) -> DiscoveryScanResult:
    raw_contracts: list[dict[str, Any]] = []
    empty_years = 0
    years_scanned = 0
    error: str | None = None
    status = "ok"
    calls = 0
    error_count = 0
    timeout_count = 0
    rate_limit_429 = 0
    latencies_ms: list[float] = []

    for _, window_start, window_end in windows:
        if max_contracts is not None and (total_contracts + len(raw_contracts)) >= max_contracts:
            break
        years_scanned += 1
        fetch = _fetch_contract_window(
            client=client,
            kind=kind,
            token=token,
            window_start=window_start,
            window_end=window_end,
            limit=limit,
            page_limit=page_limit,
            max_requests_per_second=max_requests_per_second,
            supports_max_rps_kw=supports_max_rps_kw,
            resolved_contract_status=resolved_contract_status,
            contracts_rate_limiter=contracts_rate_limiter,
            fail_fast=fail_fast,
        )
        calls += fetch.calls
        error_count += 1 if fetch.error is not None else 0
        timeout_count += fetch.timeout_count
        rate_limit_429 += fetch.rate_limit_429
        latencies_ms.append(fetch.latency_ms)
        if fetch.error is not None:
            error = fetch.error
            status = "error"
            break
        contracts = fetch.contracts
        if prefix and contracts:
            contracts = [raw for raw in contracts if (_contract_symbol_from_raw(raw) or "").startswith(prefix)]
        if not contracts:
            # Alpaca may not list far-dated expiries; don't let empty *future* windows
            # stop the scan before reaching current/past years.
            if window_start <= today:
                empty_years += 1
                if empty_years >= 3:
                    break
            continue
        empty_years = 0
        raw_contracts.extend(contracts)

    return DiscoveryScanResult(
        raw_contracts=raw_contracts,
        years_scanned=years_scanned,
        empty_years=empty_years,
        status=status,
        error=error,
        calls=calls,
        error_count=error_count,
        timeout_count=timeout_count,
        rate_limit_429=rate_limit_429,
        latencies_ms=latencies_ms,
    )


def merge_discovered_contracts(
    *,
    raw_contracts: Sequence[Mapping[str, Any]],
    contracts_by_symbol: dict[str, dict[str, Any]],
    raw_by_symbol: dict[str, dict[str, Any]],
) -> None:
    if not raw_contracts:
        return
    df = contracts_to_df(raw_contracts)
    df = _normalize_contracts_frame(df)
    if not df.empty:
        for row in df.to_dict("records"):
            symbol = _contract_symbol_from_raw(row)
            if not symbol:
                continue
            payload = dict(row)
            payload["contractSymbol"] = symbol
            contracts_by_symbol[symbol] = payload
    for raw in raw_contracts:
        symbol = _contract_symbol_from_raw(raw)
        if symbol:
            raw_by_symbol[symbol] = dict(raw)


def _fetch_contract_window(
    *,
    client: AlpacaClient,
    kind: str,
    token: str,
    window_start: date,
    window_end: date,
    limit: int | None,
    page_limit: int | None,
    max_requests_per_second: float | None,
    supports_max_rps_kw: bool,
    resolved_contract_status: str | None,
    contracts_rate_limiter: _RateLimiter,
    fail_fast: bool,
) -> _DiscoveryWindowFetchResult:
    started = time_mod.perf_counter()
    try:
        contracts_rate_limiter.wait_turn()
        contracts = list_option_contracts(
            client,
            underlying=token if kind == "underlying" else None,
            root_symbol=token if kind == "root_symbol" else None,
            exp_gte=window_start,
            exp_lte=window_end,
            limit=limit,
            page_limit=page_limit,
            max_requests_per_second=max_requests_per_second,
            supports_max_rps_kw=supports_max_rps_kw,
            contract_status=resolved_contract_status,
        )
    except Exception as exc:  # noqa: BLE001
        if fail_fast:
            raise
        return _DiscoveryWindowFetchResult(
            contracts=[],
            calls=1,
            error=str(exc),
            timeout_count=1 if _looks_like_timeout(exc) else 0,
            rate_limit_429=1 if _looks_like_429(exc) else 0,
            latency_ms=(time_mod.perf_counter() - started) * 1000.0,
        )
    return _DiscoveryWindowFetchResult(
        contracts=contracts,
        calls=1,
        error=None,
        timeout_count=0,
        rate_limit_429=0,
        latency_ms=(time_mod.perf_counter() - started) * 1000.0,
    )


__all__ = [
    "DiscoveryScanResult",
    "build_scan_targets",
    "list_option_contracts",
    "merge_discovered_contracts",
    "scan_discovery_target",
]
