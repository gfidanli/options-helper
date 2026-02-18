from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import threading
import time as time_mod
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from options_helper.analysis.osi import normalize_underlying
from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df
from options_helper.data.ingestion.options_bars_discovery_helpers import (
    DiscoveryScanResult,
    build_scan_targets as _build_scan_targets,
    merge_discovered_contracts as _merge_discovered_contracts,
    scan_discovery_target as _scan_discovery_target,
)
from options_helper.data.ingestion.options_bars_helpers import (
    coerce_expiry as _coerce_expiry,
    expiry_from_contract_symbol as _expiry_from_contract_symbol,
    normalize_contract_status as _normalize_contract_status,
    normalize_contracts_frame as _normalize_contracts_frame,
    supports_max_rps_kw as _supports_max_rps_kw,
    year_windows as _year_windows,
)
from options_helper.data.ingestion.tuning import EndpointStats, build_endpoint_stats
from options_helper.data.option_bars import OptionBarsStore


_MIN_START_DATE = date(2000, 1, 1)
_REFRESH_TAIL_DAYS = 30
_REFRESH_OVERLAP_DAYS = 3


@dataclass(frozen=True)
class UnderlyingDiscoverySummary:
    underlying: str
    contracts: int
    years_scanned: int
    empty_years: int
    status: str
    error: str | None


@dataclass(frozen=True)
class ContractDiscoveryOutput:
    contracts: pd.DataFrame
    raw_by_symbol: dict[str, dict[str, Any]]
    summaries: list[UnderlyingDiscoverySummary]
    endpoint_stats: ContractDiscoveryStats | None = None


@dataclass(frozen=True)
class PreparedContracts:
    contracts: pd.DataFrame
    expiries: list[date]


@dataclass(frozen=True)
class BarsBackfillSummary:
    total_contracts: int
    total_expiries: int
    planned_contracts: int
    skipped_contracts: int
    ok_contracts: int
    error_contracts: int
    bars_rows: int
    requests_attempted: int
    endpoint_stats: BarsEndpointStats | None = None


@dataclass(frozen=True)
class ContractDiscoveryStats:
    endpoint_stats: EndpointStats


@dataclass(frozen=True)
class BarsEndpointStats:
    endpoint_stats: EndpointStats


class _RequestRateLimiter:
    def __init__(self, max_requests_per_second: float | None) -> None:
        if max_requests_per_second is None or max_requests_per_second <= 0:
            self._min_interval_seconds = 0.0
        else:
            self._min_interval_seconds = 1.0 / float(max_requests_per_second)
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        while True:
            with self._lock:
                now = time_mod.monotonic()
                remaining = self._next_allowed_at - now
                if remaining <= 0:
                    self._next_allowed_at = now + self._min_interval_seconds
                    return
            time_mod.sleep(remaining)


def discover_option_contracts(
    client: AlpacaClient,
    *,
    underlyings: Iterable[str],
    root_symbols: Iterable[str] | None = None,
    exp_start: date,
    exp_end: date,
    contract_symbol_prefix: str | None = None,
    limit: int | None = None,
    page_limit: int | None = None,
    max_contracts: int | None = None,
    max_requests_per_second: float | None = None,
    contract_status: str | None = "active",
    fail_fast: bool = False,
) -> ContractDiscoveryOutput:
    resolved_contract_status = _normalize_contract_status(contract_status)
    contracts_by_symbol: dict[str, dict[str, Any]] = {}
    raw_by_symbol: dict[str, dict[str, Any]] = {}
    summaries: list[UnderlyingDiscoverySummary] = []
    method = getattr(client, "list_option_contracts")
    supports_max_rps_kw = _supports_max_rps_kw(method)
    contracts_rate_limiter = _RequestRateLimiter(max_requests_per_second if not supports_max_rps_kw else None)
    calls = 0
    error_count = 0
    timeout_count = 0
    rate_limit_429 = 0
    latencies_ms: list[float] = []

    prefix = str(contract_symbol_prefix or "").strip().upper() or None
    windows = _year_windows(exp_start, exp_end)
    today = date.today()
    scan_targets = _build_scan_targets(underlyings=underlyings, root_symbols=root_symbols)
    total_contracts = 0

    for kind, raw_symbol in scan_targets:
        if max_contracts is not None and total_contracts >= max_contracts:
            break
        token = normalize_underlying(raw_symbol)
        if not token:
            continue
        scan_result = _scan_discovery_target(
            client=client,
            kind=kind,
            token=token,
            windows=windows,
            max_contracts=max_contracts,
            total_contracts=total_contracts,
            limit=limit,
            page_limit=page_limit,
            max_requests_per_second=max_requests_per_second,
            supports_max_rps_kw=supports_max_rps_kw,
            resolved_contract_status=resolved_contract_status,
            contracts_rate_limiter=contracts_rate_limiter,
            fail_fast=fail_fast,
            prefix=prefix,
            today=today,
        )
        calls += scan_result.calls
        error_count += scan_result.error_count
        timeout_count += scan_result.timeout_count
        rate_limit_429 += scan_result.rate_limit_429
        latencies_ms.extend(scan_result.latencies_ms)
        summaries.append(_summary_from_scan_result(token=token, scan_result=scan_result))
        _merge_discovered_contracts(
            raw_contracts=scan_result.raw_contracts,
            contracts_by_symbol=contracts_by_symbol,
            raw_by_symbol=raw_by_symbol,
        )
        total_contracts = len(raw_by_symbol)

    return _build_discovery_output(
        contracts_by_symbol=contracts_by_symbol,
        raw_by_symbol=raw_by_symbol,
        summaries=summaries,
        calls=calls,
        error_count=error_count,
        timeout_count=timeout_count,
        rate_limit_429=rate_limit_429,
        latencies_ms=latencies_ms,
    )


def _summary_from_scan_result(
    *,
    token: str,
    scan_result: DiscoveryScanResult,
) -> UnderlyingDiscoverySummary:
    return UnderlyingDiscoverySummary(
        underlying=token,
        contracts=len(scan_result.raw_contracts),
        years_scanned=scan_result.years_scanned,
        empty_years=scan_result.empty_years,
        status=scan_result.status,
        error=scan_result.error,
    )


def _build_discovery_output(
    *,
    contracts_by_symbol: Mapping[str, Mapping[str, Any]],
    raw_by_symbol: Mapping[str, Mapping[str, Any]],
    summaries: Sequence[UnderlyingDiscoverySummary],
    calls: int,
    error_count: int,
    timeout_count: int,
    rate_limit_429: int,
    latencies_ms: Sequence[float],
) -> ContractDiscoveryOutput:
    endpoint_stats = build_endpoint_stats(
        calls=calls,
        retries=0,
        rate_limit_429=rate_limit_429,
        timeout_count=timeout_count,
        error_count=error_count,
        latencies_ms=latencies_ms,
    )
    if not contracts_by_symbol:
        return ContractDiscoveryOutput(
            contracts=contracts_to_df([]),
            raw_by_symbol={key: dict(value) for key, value in raw_by_symbol.items()},
            summaries=list(summaries),
            endpoint_stats=ContractDiscoveryStats(endpoint_stats=endpoint_stats),
        )

    combined = pd.DataFrame(list(contracts_by_symbol.values()))
    combined = _normalize_contracts_frame(combined)
    if "contractSymbol" in combined.columns:
        combined = combined.dropna(subset=["contractSymbol"])
        combined = combined.drop_duplicates(subset=["contractSymbol"], keep="last")
        combined = combined.sort_values(by=["contractSymbol"], kind="stable")
    return ContractDiscoveryOutput(
        contracts=combined,
        raw_by_symbol={key: dict(value) for key, value in raw_by_symbol.items()},
        summaries=list(summaries),
        endpoint_stats=ContractDiscoveryStats(endpoint_stats=endpoint_stats),
    )


def prepare_contracts_for_bars(
    contracts: pd.DataFrame,
    *,
    max_expiries: int | None = None,
    max_contracts: int | None = None,
) -> PreparedContracts:
    if contracts is None or contracts.empty:
        return PreparedContracts(contracts=pd.DataFrame(), expiries=[])

    df = _normalize_contracts_frame(contracts)
    if "contractSymbol" not in df.columns:
        return PreparedContracts(contracts=pd.DataFrame(), expiries=[])

    expiry_series = df.get("expiry") if "expiry" in df.columns else None
    expiry = expiry_series.map(_coerce_expiry) if expiry_series is not None else None
    if expiry is None:
        expiry = pd.Series([None] * len(df), index=df.index)

    missing = expiry.isna()
    if missing.any():
        expiry.loc[missing] = df.loc[missing, "contractSymbol"].map(_expiry_from_contract_symbol)

    df = df.copy()
    df["expiry_date"] = expiry
    df = df.dropna(subset=["contractSymbol", "expiry_date"]).copy()
    df = df.drop_duplicates(subset=["contractSymbol"], keep="last")
    if df.empty:
        return PreparedContracts(contracts=df, expiries=[])

    expiries = sorted({val for val in df["expiry_date"].tolist() if isinstance(val, date)}, reverse=True)
    if max_expiries is not None:
        expiries = expiries[:max_expiries]
        df = df[df["expiry_date"].isin(expiries)].copy()

    df = df.sort_values(["expiry_date", "contractSymbol"], ascending=[False, True])
    if max_contracts is not None:
        df = df.head(max_contracts)

    expiries = sorted({val for val in df["expiry_date"].tolist() if isinstance(val, date)}, reverse=True)
    return PreparedContracts(contracts=df.reset_index(drop=True), expiries=expiries)


def backfill_option_bars(
    client: AlpacaClient,
    store: OptionBarsStore,
    contracts: pd.DataFrame,
    *,
    provider: str,
    lookback_years: int = 10,
    page_limit: int | None = None,
    bars_concurrency: int = 1,
    bars_max_requests_per_second: float | None = None,
    bars_batch_mode: str = "adaptive",
    bars_batch_size: int = 8,
    bars_write_batch_size: int = 200,
    resume: bool = True,
    dry_run: bool = False,
    fail_fast: bool = False,
    today: date | None = None,
) -> BarsBackfillSummary:
    from options_helper.data.ingestion.options_bars_backfill_runtime_legacy import (
        backfill_option_bars_runtime,
    )

    runtime = backfill_option_bars_runtime(
        client,
        store,
        contracts,
        provider=provider,
        lookback_years=lookback_years,
        page_limit=page_limit,
        bars_concurrency=bars_concurrency,
        bars_max_requests_per_second=bars_max_requests_per_second,
        bars_batch_mode=bars_batch_mode,
        bars_batch_size=bars_batch_size,
        bars_write_batch_size=bars_write_batch_size,
        resume=resume,
        dry_run=dry_run,
        fail_fast=fail_fast,
        today=today or date.today(),
    )
    return BarsBackfillSummary(
        total_contracts=runtime.total_contracts,
        total_expiries=runtime.total_expiries,
        planned_contracts=runtime.planned_contracts,
        skipped_contracts=runtime.skipped_contracts,
        ok_contracts=runtime.ok_contracts,
        error_contracts=runtime.error_contracts,
        bars_rows=runtime.bars_rows,
        requests_attempted=runtime.requests_attempted,
        endpoint_stats=BarsEndpointStats(endpoint_stats=runtime.endpoint_stats),
    )
