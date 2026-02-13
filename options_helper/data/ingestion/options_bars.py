from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import threading
import time as time_mod
from typing import Any, Iterable

import pandas as pd

from options_helper.analysis.osi import normalize_underlying
from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df
from options_helper.data.ingestion.common import shift_years
from options_helper.data.ingestion.options_bars_helpers import (
    coerce_expiry as _coerce_expiry,
    coerce_meta_dt as _coerce_meta_dt,
    contract_symbol_from_raw as _contract_symbol_from_raw,
    coverage_satisfies as _coverage_satisfies,
    error_status as _error_status,
    expiry_from_contract_symbol as _expiry_from_contract_symbol,
    looks_like_429 as _looks_like_429,
    looks_like_timeout as _looks_like_timeout,
    normalize_contract_status as _normalize_contract_status,
    normalize_contracts_frame as _normalize_contracts_frame,
    supports_kw as _supports_kw,
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


@dataclass(frozen=True)
class _BarsFetchPlan:
    symbol: str
    fetch_start: date
    fetch_end: date
    has_coverage: bool


@dataclass(frozen=True)
class _BarsBatchPlan:
    symbols: tuple[str, ...]
    fetch_start: date
    fetch_end: date


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


def _list_option_contracts(
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

    total_contracts = 0
    today = date.today()
    windows = _year_windows(exp_start, exp_end)

    prefix = str(contract_symbol_prefix or "").strip().upper() or None
    scan_targets: list[tuple[str, str]] = []
    scan_targets.extend([("underlying", sym) for sym in underlyings])
    if root_symbols is not None:
        scan_targets.extend([("root_symbol", sym) for sym in root_symbols])

    for kind, raw_symbol in scan_targets:
        if max_contracts is not None and total_contracts >= max_contracts:
            break
        token = normalize_underlying(raw_symbol)
        if not token:
            continue

        raw_contracts: list[dict[str, Any]] = []
        empty_years = 0
        years_scanned = 0
        error: str | None = None
        status = "ok"

        for _, window_start, window_end in windows:
            if max_contracts is not None and total_contracts >= max_contracts:
                break
            years_scanned += 1
            started = time_mod.perf_counter()
            try:
                contracts_rate_limiter.wait_turn()
                calls += 1
                contracts = _list_option_contracts(
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
                latencies_ms.append((time_mod.perf_counter() - started) * 1000.0)
                error = str(exc)
                status = "error"
                error_count += 1
                if _looks_like_timeout(exc):
                    timeout_count += 1
                if _looks_like_429(exc):
                    rate_limit_429 += 1
                if fail_fast:
                    raise
                break
            latencies_ms.append((time_mod.perf_counter() - started) * 1000.0)

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

            if max_contracts is not None and (total_contracts + len(raw_contracts)) >= max_contracts:
                break

        if raw_contracts:
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
                    raw_by_symbol[symbol] = raw
            total_contracts = len(raw_by_symbol)

        summaries.append(
            UnderlyingDiscoverySummary(
                underlying=token,
                contracts=len(raw_contracts),
                years_scanned=years_scanned,
                empty_years=empty_years,
                status=status,
                error=error,
            )
        )

    if not contracts_by_symbol:
        empty = contracts_to_df([])
        endpoint_stats = build_endpoint_stats(
            calls=calls,
            retries=0,
            rate_limit_429=rate_limit_429,
            timeout_count=timeout_count,
            error_count=error_count,
            latencies_ms=latencies_ms,
        )
        return ContractDiscoveryOutput(
            contracts=empty,
            raw_by_symbol=raw_by_symbol,
            summaries=summaries,
            endpoint_stats=ContractDiscoveryStats(endpoint_stats=endpoint_stats),
        )

    combined = pd.DataFrame(list(contracts_by_symbol.values()))
    combined = _normalize_contracts_frame(combined)
    if "contractSymbol" in combined.columns:
        combined = combined.dropna(subset=["contractSymbol"])
        combined = combined.drop_duplicates(subset=["contractSymbol"], keep="last")
        combined = combined.sort_values(by=["contractSymbol"], kind="stable")

    endpoint_stats = build_endpoint_stats(
        calls=calls,
        retries=0,
        rate_limit_429=rate_limit_429,
        timeout_count=timeout_count,
        error_count=error_count,
        latencies_ms=latencies_ms,
    )
    return ContractDiscoveryOutput(
        contracts=combined,
        raw_by_symbol=raw_by_symbol,
        summaries=summaries,
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
    if contracts is None or contracts.empty:
        return BarsBackfillSummary(
            total_contracts=0,
            total_expiries=0,
            planned_contracts=0,
            skipped_contracts=0,
            ok_contracts=0,
            error_contracts=0,
            bars_rows=0,
            requests_attempted=0,
            endpoint_stats=BarsEndpointStats(
                endpoint_stats=build_endpoint_stats(calls=0, error_count=0, latencies_ms=[])
            ),
        )

    df = contracts.copy()
    df["contractSymbol"] = df["contractSymbol"].map(lambda v: str(v).strip().upper() if v is not None else None)
    df = df.dropna(subset=["contractSymbol", "expiry_date"])
    df = df.drop_duplicates(subset=["contractSymbol"], keep="last")

    expiry_groups: dict[date, list[str]] = {}
    for _, row in df.iterrows():
        exp = row["expiry_date"]
        if not isinstance(exp, date):
            continue
        sym = row["contractSymbol"]
        if not sym:
            continue
        expiry_groups.setdefault(exp, []).append(sym)

    total_contracts = len({sym for syms in expiry_groups.values() for sym in syms})
    total_expiries = len(expiry_groups)
    today = today or date.today()

    skipped_contracts = 0
    ok_contracts = 0
    error_contracts = 0
    bars_rows = 0
    requests_attempted = 0
    planned_contracts = 0
    fetch_plans: list[_BarsFetchPlan] = []
    bulk_coverage_available = False
    coverage_by_symbol: dict[str, dict[str, Any]] = {}

    if resume:
        bulk_loader = getattr(store, "coverage_bulk", None)
        if callable(bulk_loader):
            symbols_for_coverage = sorted({sym for syms in expiry_groups.values() for sym in syms if sym})
            try:
                payload = bulk_loader(symbols_for_coverage, interval="1d", provider=provider)
                if isinstance(payload, dict):
                    coverage_by_symbol = {
                        str(sym).strip().upper(): meta
                        for sym, meta in payload.items()
                        if str(sym or "").strip() and isinstance(meta, dict)
                    }
                    bulk_coverage_available = True
            except Exception:  # noqa: BLE001
                bulk_coverage_available = False

    for expiry in sorted(expiry_groups.keys(), reverse=True):
        end = min(today, expiry)
        start_candidate = shift_years(expiry, -abs(int(lookback_years)))
        start_candidate = min(today, start_candidate)
        start = max(_MIN_START_DATE, start_candidate)
        if start > end:
            continue

        desired_symbols = sorted({s for s in expiry_groups.get(expiry, []) if s})
        if not desired_symbols:
            continue

        planned_contracts += len(desired_symbols)

        is_expired = expiry < today

        for sym in desired_symbols:
            meta: dict[str, Any] | None
            if resume:
                if bulk_coverage_available:
                    meta = coverage_by_symbol.get(sym)
                else:
                    try:
                        meta = store.coverage(sym, interval="1d", provider=provider)
                    except Exception:  # noqa: BLE001
                        meta = None
            else:
                meta = None

            status = str((meta or {}).get("status") or "").strip().lower()
            last_attempt = _coerce_meta_dt(meta, "last_attempt_at", "last_attempt")
            covered_end = _coerce_meta_dt(meta, "end_ts", "end")
            has_coverage = bool(_coerce_meta_dt(meta, "start_ts", "start") or covered_end)

            if resume and meta:
                if status == "forbidden":
                    skipped_contracts += 1
                    continue
                if is_expired and status in {"ok", "partial", "not_found", "forbidden"}:
                    skipped_contracts += 1
                    continue
                if last_attempt is not None and last_attempt.date() == today:
                    skipped_contracts += 1
                    continue
                if not is_expired and _coverage_satisfies(meta, start=start, end=end):
                    skipped_contracts += 1
                    continue

            fetch_start = start
            if not is_expired:
                tail_floor = max(start, end - timedelta(days=_REFRESH_TAIL_DAYS))
                if covered_end is not None and status in {"ok", "partial"}:
                    overlap_start = covered_end.date() - timedelta(days=_REFRESH_OVERLAP_DAYS)
                    fetch_start = max(tail_floor, overlap_start)
                else:
                    fetch_start = tail_floor

            if dry_run:
                requests_attempted += 1
                continue

            fetch_plans.append(
                _BarsFetchPlan(
                    symbol=sym,
                    fetch_start=fetch_start,
                    fetch_end=end,
                    has_coverage=has_coverage,
                )
            )

    if dry_run:
        return BarsBackfillSummary(
            total_contracts=total_contracts,
            total_expiries=total_expiries,
            planned_contracts=planned_contracts,
            skipped_contracts=skipped_contracts,
            ok_contracts=ok_contracts,
            error_contracts=error_contracts,
            bars_rows=bars_rows,
            requests_attempted=requests_attempted,
            endpoint_stats=BarsEndpointStats(
                endpoint_stats=build_endpoint_stats(
                    calls=requests_attempted,
                    error_count=error_contracts,
                    latencies_ms=[],
                )
            ),
        )

    rate_limiter = _RequestRateLimiter(bars_max_requests_per_second)
    max_workers = max(1, int(bars_concurrency))
    write_batch_size = max(1, int(bars_write_batch_size))
    batch_mode = str(bars_batch_mode or "adaptive").strip().lower()
    if batch_mode not in {"adaptive", "per-contract"}:
        batch_mode = "adaptive"
    batch_size = max(1, int(bars_batch_size))
    if fail_fast:
        max_workers = 1
    pending_bars: list[pd.DataFrame] = []
    pending_success_symbols: list[str] = []
    pending_success_rows: dict[str, int] = {}
    pending_success_start: dict[str, datetime] = {}
    pending_success_end: dict[str, datetime] = {}
    pending_success_count = 0
    pending_errors: dict[tuple[str, str], list[str]] = {}
    pending_error_count = 0
    apply_write_batch = getattr(store, "apply_write_batch", None)
    endpoint_calls = 0
    endpoint_errors = 0
    endpoint_timeout_count = 0
    endpoint_rate_limit_429 = 0
    endpoint_latencies_ms: list[float] = []
    endpoint_split_count = 0
    endpoint_fallback_count = 0
    endpoint_lock = threading.Lock()
    plan_by_symbol = {plan.symbol: plan for plan in fetch_plans}

    def _fetch_bars(
        symbols: list[str],
        *,
        fetch_start: date,
        fetch_end: date,
    ) -> pd.DataFrame:
        nonlocal endpoint_calls, endpoint_errors, endpoint_timeout_count, endpoint_rate_limit_429
        rate_limiter.wait_turn()
        started = time_mod.perf_counter()
        try:
            return client.get_option_bars_daily_full(
                symbols,
                start=fetch_start,
                end=fetch_end,
                interval="1d",
                chunk_size=max(1, len(symbols)),
                page_limit=page_limit,
            )
        except Exception as exc:  # noqa: BLE001
            with endpoint_lock:
                endpoint_errors += 1
                if _looks_like_timeout(exc):
                    endpoint_timeout_count += 1
                if _looks_like_429(exc):
                    endpoint_rate_limit_429 += 1
            raise
        finally:
            elapsed_ms = (time_mod.perf_counter() - started) * 1000.0
            with endpoint_lock:
                endpoint_calls += 1
                endpoint_latencies_ms.append(elapsed_ms)

    def _flush_success_buffers() -> None:
        nonlocal pending_success_count
        if pending_bars:
            merged = pd.concat(pending_bars, ignore_index=True)
            store.upsert_bars(merged, interval="1d", provider=provider)
            pending_bars.clear()
        if pending_success_symbols:
            store.mark_meta_success(
                pending_success_symbols,
                interval="1d",
                provider=provider,
                rows=pending_success_rows or None,
                start_ts=pending_success_start or None,
                end_ts=pending_success_end or None,
            )
            pending_success_symbols.clear()
            pending_success_rows.clear()
            pending_success_start.clear()
            pending_success_end.clear()
            pending_success_count = 0

    def _flush_error_buffers() -> None:
        nonlocal pending_error_count
        if not pending_errors:
            return
        for (status, error_text), symbols in pending_errors.items():
            if not symbols:
                continue
            store.mark_meta_error(
                symbols,
                interval="1d",
                provider=provider,
                error=error_text,
                status=status,
            )
        pending_errors.clear()
        pending_error_count = 0

    def _flush_buffers() -> None:
        nonlocal pending_success_count, pending_error_count
        if callable(apply_write_batch):
            merged = pd.concat(pending_bars, ignore_index=True) if pending_bars else None
            error_groups = [(symbols, status, error_text) for (status, error_text), symbols in pending_errors.items()]
            if merged is not None or pending_success_symbols or error_groups:
                apply_write_batch(
                    bars_df=merged,
                    interval="1d",
                    provider=provider,
                    success_symbols=pending_success_symbols,
                    success_rows=pending_success_rows or None,
                    success_start_ts=pending_success_start or None,
                    success_end_ts=pending_success_end or None,
                    error_groups=error_groups,
                )
            pending_bars.clear()
            pending_success_symbols.clear()
            pending_success_rows.clear()
            pending_success_start.clear()
            pending_success_end.clear()
            pending_success_count = 0
            pending_errors.clear()
            pending_error_count = 0
            return

        _flush_success_buffers()
        _flush_error_buffers()

    def _queue_success(
        *,
        symbol: str,
        rows: int,
        start_ts: datetime | None,
        end_ts: datetime | None,
        bars_df: pd.DataFrame | None = None,
    ) -> None:
        nonlocal pending_success_count
        if bars_df is not None and not bars_df.empty:
            pending_bars.append(bars_df)
        pending_success_symbols.append(symbol)
        pending_success_rows[symbol] = int(rows)
        if start_ts is not None:
            pending_success_start[symbol] = start_ts
        if end_ts is not None:
            pending_success_end[symbol] = end_ts
        pending_success_count += 1
        if pending_success_count >= write_batch_size:
            _flush_buffers()

    def _queue_error(*, symbol: str, status: str, error_text: str) -> None:
        nonlocal pending_error_count
        key = (str(status or "error").strip().lower() or "error", str(error_text))
        pending_errors.setdefault(key, []).append(symbol)
        pending_error_count += 1
        if pending_error_count >= write_batch_size:
            _flush_buffers()

    def _record_error(plan: _BarsFetchPlan, exc: Exception) -> None:
        nonlocal error_contracts
        _queue_error(
            symbol=plan.symbol,
            status=_error_status(exc),
            error_text=str(exc),
        )
        error_contracts += 1

    def _record_success(plan: _BarsFetchPlan, df_bars: pd.DataFrame | None) -> None:
        nonlocal bars_rows, ok_contracts, error_contracts
        if df_bars is None or df_bars.empty:
            if plan.has_coverage:
                _queue_success(
                    symbol=plan.symbol,
                    rows=0,
                    start_ts=None,
                    end_ts=None,
                )
                ok_contracts += 1
            else:
                _queue_error(
                    symbol=plan.symbol,
                    status="not_found",
                    error_text="no bars returned",
                )
                error_contracts += 1
            return

        bars_rows += int(len(df_bars))

        normalized = df_bars.copy()
        normalized["contractSymbol"] = normalized["contractSymbol"].map(
            lambda v: str(v).strip().upper() if v is not None else None
        )
        normalized = normalized.dropna(subset=["contractSymbol"])
        rows = normalized[normalized["contractSymbol"] == plan.symbol]
        if not rows.empty:
            _queue_success(
                symbol=plan.symbol,
                rows=int(len(rows)),
                start_ts=rows["ts"].min(),
                end_ts=rows["ts"].max(),
                bars_df=df_bars,
            )
            ok_contracts += 1
            return

        if plan.has_coverage:
            _queue_success(
                symbol=plan.symbol,
                rows=0,
                start_ts=None,
                end_ts=None,
                bars_df=df_bars,
            )
            ok_contracts += 1
            return

        _queue_error(
            symbol=plan.symbol,
            status="not_found",
            error_text="no bars returned",
        )
        error_contracts += 1

    def _fetch_single_plan(plan: _BarsFetchPlan) -> tuple[_BarsFetchPlan, pd.DataFrame | None, Exception | None]:
        try:
            df_bars = _fetch_bars([plan.symbol], fetch_start=plan.fetch_start, fetch_end=plan.fetch_end)
            return plan, df_bars, None
        except Exception as exc:  # noqa: BLE001
            return plan, None, exc

    def _split_batch(symbols: list[str]) -> tuple[list[str], list[str]]:
        midpoint = max(1, len(symbols) // 2)
        return symbols[:midpoint], symbols[midpoint:]

    def _resolve_adaptive_batch(batch_plan: _BarsBatchPlan) -> list[tuple[_BarsFetchPlan, pd.DataFrame | None, Exception | None]]:
        nonlocal endpoint_split_count, endpoint_fallback_count
        symbols = [sym for sym in batch_plan.symbols if sym]
        if not symbols:
            return []

        try:
            df_bars = _fetch_bars(
                symbols,
                fetch_start=batch_plan.fetch_start,
                fetch_end=batch_plan.fetch_end,
            )
        except Exception as exc:  # noqa: BLE001
            if len(symbols) == 1:
                plan = plan_by_symbol.get(symbols[0])
                if plan is None:
                    return []
                return [(plan, None, exc)]
            with endpoint_lock:
                endpoint_split_count += 1
            left_symbols, right_symbols = _split_batch(symbols)
            return _resolve_adaptive_batch(
                _BarsBatchPlan(
                    symbols=tuple(left_symbols),
                    fetch_start=batch_plan.fetch_start,
                    fetch_end=batch_plan.fetch_end,
                )
            ) + _resolve_adaptive_batch(
                _BarsBatchPlan(
                    symbols=tuple(right_symbols),
                    fetch_start=batch_plan.fetch_start,
                    fetch_end=batch_plan.fetch_end,
                )
            )

        if df_bars is None or df_bars.empty:
            if len(symbols) == 1:
                plan = plan_by_symbol.get(symbols[0])
                if plan is None:
                    return []
                return [(plan, pd.DataFrame(), None)]
            with endpoint_lock:
                endpoint_split_count += 1
                endpoint_fallback_count += 1
            left_symbols, right_symbols = _split_batch(symbols)
            return _resolve_adaptive_batch(
                _BarsBatchPlan(
                    symbols=tuple(left_symbols),
                    fetch_start=batch_plan.fetch_start,
                    fetch_end=batch_plan.fetch_end,
                )
            ) + _resolve_adaptive_batch(
                _BarsBatchPlan(
                    symbols=tuple(right_symbols),
                    fetch_start=batch_plan.fetch_start,
                    fetch_end=batch_plan.fetch_end,
                )
            )

        normalized = df_bars.copy()
        if "contractSymbol" not in normalized.columns:
            normalized["contractSymbol"] = pd.NA
        normalized["contractSymbol"] = normalized["contractSymbol"].map(
            lambda value: str(value).strip().upper() if value is not None else None
        )
        normalized = normalized.dropna(subset=["contractSymbol"]).copy()
        if not normalized.empty:
            normalized = normalized[normalized["contractSymbol"].isin(symbols)].copy()

        present_symbols = (
            sorted({str(value).strip().upper() for value in normalized["contractSymbol"].tolist()})
            if not normalized.empty
            else []
        )
        present_set = set(present_symbols)
        outcomes: list[tuple[_BarsFetchPlan, pd.DataFrame | None, Exception | None]] = []
        for sym in present_symbols:
            plan = plan_by_symbol.get(sym)
            if plan is None:
                continue
            rows = normalized[normalized["contractSymbol"] == sym].copy()
            outcomes.append((plan, rows, None))

        missing_symbols = [sym for sym in symbols if sym not in present_set]
        if not missing_symbols:
            return outcomes

        if len(symbols) == 1:
            plan = plan_by_symbol.get(symbols[0])
            if plan is not None:
                outcomes.append((plan, pd.DataFrame(), None))
            return outcomes

        with endpoint_lock:
            endpoint_split_count += 1
            endpoint_fallback_count += 1
        left_symbols, right_symbols = _split_batch(missing_symbols)
        return outcomes + _resolve_adaptive_batch(
            _BarsBatchPlan(
                symbols=tuple(left_symbols),
                fetch_start=batch_plan.fetch_start,
                fetch_end=batch_plan.fetch_end,
            )
        ) + _resolve_adaptive_batch(
            _BarsBatchPlan(
                symbols=tuple(right_symbols),
                fetch_start=batch_plan.fetch_start,
                fetch_end=batch_plan.fetch_end,
            )
        )

    def _process_outcome(plan: _BarsFetchPlan, df_bars: pd.DataFrame | None, exc: Exception | None) -> None:
        if exc is not None:
            _record_error(plan, exc)
            return
        _record_success(plan, df_bars)

    if batch_mode == "per-contract":
        if max_workers <= 1:
            for plan in fetch_plans:
                item_plan, df_bars, exc = _fetch_single_plan(plan)
                _process_outcome(item_plan, df_bars, exc)
                if fail_fast and exc is not None:
                    _flush_buffers()
                    raise exc
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_fetch_single_plan, plan): plan for plan in fetch_plans}
                for future in as_completed(futures):
                    item_plan, df_bars, exc = future.result()
                    _process_outcome(item_plan, df_bars, exc)
    else:
        grouped_symbols: dict[tuple[date, date], list[str]] = {}
        for plan in fetch_plans:
            key = (plan.fetch_start, plan.fetch_end)
            grouped_symbols.setdefault(key, []).append(plan.symbol)

        batch_plans: list[_BarsBatchPlan] = []
        for (fetch_start, fetch_end), symbols in grouped_symbols.items():
            ordered = sorted({sym for sym in symbols if sym})
            for index in range(0, len(ordered), batch_size):
                chunk = ordered[index : index + batch_size]
                if not chunk:
                    continue
                batch_plans.append(
                    _BarsBatchPlan(
                        symbols=tuple(chunk),
                        fetch_start=fetch_start,
                        fetch_end=fetch_end,
                    )
                )

        if max_workers <= 1:
            for batch_plan in batch_plans:
                outcomes = _resolve_adaptive_batch(batch_plan)
                for item_plan, df_bars, exc in outcomes:
                    _process_outcome(item_plan, df_bars, exc)
                    if fail_fast and exc is not None:
                        _flush_buffers()
                        raise exc
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_resolve_adaptive_batch, batch_plan): batch_plan for batch_plan in batch_plans}
                for future in as_completed(futures):
                    outcomes = future.result()
                    for item_plan, df_bars, exc in outcomes:
                        _process_outcome(item_plan, df_bars, exc)

    _flush_buffers()
    requests_attempted = endpoint_calls
    endpoint_stats = build_endpoint_stats(
        calls=endpoint_calls,
        retries=0,
        rate_limit_429=endpoint_rate_limit_429,
        timeout_count=endpoint_timeout_count,
        error_count=endpoint_errors,
        latencies_ms=endpoint_latencies_ms,
        split_count=endpoint_split_count,
        fallback_count=endpoint_fallback_count,
    )

    return BarsBackfillSummary(
        total_contracts=total_contracts,
        total_expiries=total_expiries,
        planned_contracts=planned_contracts,
        skipped_contracts=skipped_contracts,
        ok_contracts=ok_contracts,
        error_contracts=error_contracts,
        bars_rows=bars_rows,
        requests_attempted=requests_attempted,
        endpoint_stats=BarsEndpointStats(endpoint_stats=endpoint_stats),
    )
