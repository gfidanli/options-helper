from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import inspect
import threading
import time as time_mod
from typing import Any, Iterable

import pandas as pd

from options_helper.analysis.osi import normalize_underlying, parse_contract_symbol
from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df
from options_helper.data.option_bars import OptionBarsStore
from options_helper.data.ingestion.common import shift_years


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


@dataclass(frozen=True)
class _BarsFetchPlan:
    symbol: str
    fetch_start: date
    fetch_end: date
    has_coverage: bool


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


def _contract_symbol_from_raw(raw: dict[str, Any]) -> str | None:
    for key in ("contractSymbol", "symbol", "option_symbol", "contract"):
        val = raw.get(key)
        if val:
            return str(val).strip().upper()
    return None


def _year_windows(exp_start: date, exp_end: date) -> list[tuple[int, date, date]]:
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


def _coerce_expiry(value: Any) -> date | None:
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


def _expiry_from_contract_symbol(symbol: Any) -> date | None:
    if symbol is None:
        return None
    parsed = parse_contract_symbol(str(symbol))
    return parsed.expiry if parsed else None


def _normalize_contracts_frame(df: pd.DataFrame) -> pd.DataFrame:
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


def _coverage_satisfies(meta: dict[str, Any] | None, *, start: date, end: date) -> bool:
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


def _coerce_meta_dt(meta: dict[str, Any] | None, *keys: str) -> datetime | None:
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


def _error_status(exc: Exception) -> str:
    msg = str(exc).lower()
    if "403" in msg or "forbidden" in msg:
        return "forbidden"
    if "402" in msg or "payment required" in msg:
        return "forbidden"
    if "404" in msg or "not found" in msg:
        return "not_found"
    return "error"


def _list_option_contracts(
    client: AlpacaClient,
    *,
    underlying: str,
    exp_gte: date,
    exp_lte: date,
    page_limit: int | None,
    max_requests_per_second: float | None,
) -> list[dict[str, Any]]:
    method = getattr(client, "list_option_contracts")
    kwargs: dict[str, Any] = {
        "exp_gte": exp_gte,
        "exp_lte": exp_lte,
        "page_limit": page_limit,
    }
    if max_requests_per_second is not None:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "max_requests_per_second" in signature.parameters:
            kwargs["max_requests_per_second"] = max_requests_per_second
    return method(underlying, **kwargs)


def discover_option_contracts(
    client: AlpacaClient,
    *,
    underlyings: Iterable[str],
    exp_start: date,
    exp_end: date,
    page_limit: int | None = None,
    max_contracts: int | None = None,
    max_requests_per_second: float | None = None,
    fail_fast: bool = False,
) -> ContractDiscoveryOutput:
    frames: list[pd.DataFrame] = []
    raw_by_symbol: dict[str, dict[str, Any]] = {}
    summaries: list[UnderlyingDiscoverySummary] = []
    contracts_rate_limiter = _RequestRateLimiter(max_requests_per_second)

    total_contracts = 0
    today = date.today()
    windows = _year_windows(exp_start, exp_end)

    for raw_symbol in underlyings:
        if max_contracts is not None and total_contracts >= max_contracts:
            break
        underlying = normalize_underlying(raw_symbol)
        if not underlying:
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
            try:
                contracts_rate_limiter.wait_turn()
                contracts = _list_option_contracts(
                    client,
                    underlying=underlying,
                    exp_gte=window_start,
                    exp_lte=window_end,
                    page_limit=page_limit,
                    max_requests_per_second=max_requests_per_second,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                status = "error"
                if fail_fast:
                    raise
                break

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
                frames.append(df)
            for raw in raw_contracts:
                symbol = _contract_symbol_from_raw(raw)
                if symbol:
                    raw_by_symbol[symbol] = raw
            total_contracts = len(raw_by_symbol)

        summaries.append(
            UnderlyingDiscoverySummary(
                underlying=underlying,
                contracts=len(raw_contracts),
                years_scanned=years_scanned,
                empty_years=empty_years,
                status=status,
                error=error,
            )
        )

    if not frames:
        empty = contracts_to_df([])
        return ContractDiscoveryOutput(contracts=empty, raw_by_symbol=raw_by_symbol, summaries=summaries)

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalize_contracts_frame(combined)
    if "contractSymbol" in combined.columns:
        combined = combined.dropna(subset=["contractSymbol"])
        combined = combined.drop_duplicates(subset=["contractSymbol"], keep="last")

    return ContractDiscoveryOutput(contracts=combined, raw_by_symbol=raw_by_symbol, summaries=summaries)


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
        )

    requests_attempted += len(fetch_plans)
    rate_limiter = _RequestRateLimiter(bars_max_requests_per_second)
    max_workers = max(1, int(bars_concurrency))
    write_batch_size = max(1, int(bars_write_batch_size))
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

    def _fetch_bars(plan: _BarsFetchPlan) -> pd.DataFrame:
        rate_limiter.wait_turn()
        return client.get_option_bars_daily_full(
            [plan.symbol],
            start=plan.fetch_start,
            end=plan.fetch_end,
            interval="1d",
            page_limit=page_limit,
        )

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
            _flush_success_buffers()

    def _queue_error(*, symbol: str, status: str, error_text: str) -> None:
        nonlocal pending_error_count
        key = (str(status or "error").strip().lower() or "error", str(error_text))
        pending_errors.setdefault(key, []).append(symbol)
        pending_error_count += 1
        if pending_error_count >= write_batch_size:
            _flush_error_buffers()

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

    if max_workers <= 1:
        for plan in fetch_plans:
            try:
                df_bars = _fetch_bars(plan)
            except Exception as exc:  # noqa: BLE001
                _record_error(plan, exc)
                if fail_fast:
                    _flush_buffers()
                    raise
                continue
            _record_success(plan, df_bars)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_bars, plan): plan for plan in fetch_plans}
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    df_bars = future.result()
                except Exception as exc:  # noqa: BLE001
                    _record_error(plan, exc)
                    continue
                _record_success(plan, df_bars)

    _flush_buffers()

    return BarsBackfillSummary(
        total_contracts=total_contracts,
        total_expiries=total_expiries,
        planned_contracts=planned_contracts,
        skipped_contracts=skipped_contracts,
        ok_contracts=ok_contracts,
        error_contracts=error_contracts,
        bars_rows=bars_rows,
        requests_attempted=requests_attempted,
    )
