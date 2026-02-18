from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from options_helper.data.ingestion.common import shift_years
from options_helper.data.ingestion.options_bars_helpers import (
    coerce_meta_dt as _coerce_meta_dt,
    coverage_satisfies as _coverage_satisfies,
)
from options_helper.data.option_bars import OptionBarsStore

_MIN_START_DATE = date(2000, 1, 1)
_REFRESH_TAIL_DAYS = 30
_REFRESH_OVERLAP_DAYS = 3


@dataclass(frozen=True)
class BarsFetchPlan:
    symbol: str
    fetch_start: date
    fetch_end: date
    has_coverage: bool


@dataclass(frozen=True)
class BarsBatchPlan:
    symbols: tuple[str, ...]
    fetch_start: date
    fetch_end: date


@dataclass(frozen=True)
class BackfillPlanResult:
    total_contracts: int
    total_expiries: int
    planned_contracts: int
    skipped_contracts: int
    requests_attempted: int
    fetch_plans: list[BarsFetchPlan]


def _normalize_contracts(contracts: pd.DataFrame) -> pd.DataFrame:
    df = contracts.copy()
    df["contractSymbol"] = df["contractSymbol"].map(lambda v: str(v).strip().upper() if v is not None else None)
    df = df.dropna(subset=["contractSymbol", "expiry_date"])
    return df.drop_duplicates(subset=["contractSymbol"], keep="last")


def _build_expiry_groups(df: pd.DataFrame) -> dict[date, list[str]]:
    groups: dict[date, list[str]] = {}
    for _, row in df.iterrows():
        exp = row["expiry_date"]
        sym = row["contractSymbol"]
        if isinstance(exp, date) and sym:
            groups.setdefault(exp, []).append(sym)
    return groups


def _load_bulk_coverage(
    *,
    store: OptionBarsStore,
    expiry_groups: dict[date, list[str]],
    resume: bool,
    provider: str,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    if not resume:
        return False, {}
    bulk_loader = getattr(store, "coverage_bulk", None)
    if not callable(bulk_loader):
        return False, {}
    symbols = sorted({sym for syms in expiry_groups.values() for sym in syms if sym})
    try:
        payload = bulk_loader(symbols, interval="1d", provider=provider)
    except Exception:  # noqa: BLE001
        return False, {}
    if not isinstance(payload, dict):
        return False, {}
    coverage = {
        str(sym).strip().upper(): meta
        for sym, meta in payload.items()
        if str(sym or "").strip() and isinstance(meta, dict)
    }
    return True, coverage


def _should_skip_plan(
    *,
    resume: bool,
    meta: dict[str, Any] | None,
    is_expired: bool,
    start: date,
    end: date,
    today: date,
) -> bool:
    if not resume or not meta:
        return False
    status = str(meta.get("status") or "").strip().lower()
    last_attempt = _coerce_meta_dt(meta, "last_attempt_at", "last_attempt")
    if status == "forbidden":
        return True
    if is_expired and status in {"ok", "partial", "not_found", "forbidden"}:
        return True
    if last_attempt is not None and last_attempt.date() == today:
        return True
    return bool((not is_expired) and _coverage_satisfies(meta, start=start, end=end))


def _fetch_start_for_symbol(
    *,
    start: date,
    end: date,
    is_expired: bool,
    meta: dict[str, Any] | None,
) -> date:
    if is_expired:
        return start
    status = str((meta or {}).get("status") or "").strip().lower()
    covered_end = _coerce_meta_dt(meta, "end_ts", "end")
    tail_floor = max(start, end - timedelta(days=_REFRESH_TAIL_DAYS))
    if covered_end is not None and status in {"ok", "partial"}:
        overlap_start = covered_end.date() - timedelta(days=_REFRESH_OVERLAP_DAYS)
        return max(tail_floor, overlap_start)
    return tail_floor


def _coverage_meta_for_symbol(
    *,
    store: OptionBarsStore,
    resume: bool,
    bulk_coverage_available: bool,
    coverage_by_symbol: dict[str, dict[str, Any]],
    symbol: str,
    provider: str,
) -> dict[str, Any] | None:
    if not resume:
        return None
    if bulk_coverage_available:
        return coverage_by_symbol.get(symbol)
    try:
        return store.coverage(symbol, interval="1d", provider=provider)
    except Exception:  # noqa: BLE001
        return None


def build_backfill_plan(
    *,
    store: OptionBarsStore,
    contracts: pd.DataFrame,
    provider: str,
    lookback_years: int,
    resume: bool,
    dry_run: bool,
    today: date,
) -> BackfillPlanResult:
    df = _normalize_contracts(contracts)
    expiry_groups = _build_expiry_groups(df)
    total_contracts = len({sym for syms in expiry_groups.values() for sym in syms})
    total_expiries = len(expiry_groups)
    bulk_coverage_available, coverage_by_symbol = _load_bulk_coverage(
        store=store,
        expiry_groups=expiry_groups,
        resume=resume,
        provider=provider,
    )
    planned_contracts = 0
    skipped_contracts = 0
    requests_attempted = 0
    fetch_plans: list[BarsFetchPlan] = []

    for expiry in sorted(expiry_groups.keys(), reverse=True):
        end = min(today, expiry)
        start_candidate = min(today, shift_years(expiry, -abs(int(lookback_years))))
        start = max(_MIN_START_DATE, start_candidate)
        if start > end:
            continue
        desired_symbols = sorted({s for s in expiry_groups.get(expiry, []) if s})
        if not desired_symbols:
            continue
        planned_contracts += len(desired_symbols)
        is_expired = expiry < today
        for sym in desired_symbols:
            meta = _coverage_meta_for_symbol(
                store=store,
                resume=resume,
                bulk_coverage_available=bulk_coverage_available,
                coverage_by_symbol=coverage_by_symbol,
                symbol=sym,
                provider=provider,
            )
            if _should_skip_plan(resume=resume, meta=meta, is_expired=is_expired, start=start, end=end, today=today):
                skipped_contracts += 1
                continue
            fetch_start = _fetch_start_for_symbol(start=start, end=end, is_expired=is_expired, meta=meta)
            if dry_run:
                requests_attempted += 1
                continue
            has_coverage = bool(_coerce_meta_dt(meta, "start_ts", "start") or _coerce_meta_dt(meta, "end_ts", "end"))
            fetch_plans.append(BarsFetchPlan(symbol=sym, fetch_start=fetch_start, fetch_end=end, has_coverage=has_coverage))

    return BackfillPlanResult(
        total_contracts=total_contracts,
        total_expiries=total_expiries,
        planned_contracts=planned_contracts,
        skipped_contracts=skipped_contracts,
        requests_attempted=requests_attempted,
        fetch_plans=fetch_plans,
    )


__all__ = [
    "BarsBatchPlan",
    "BarsFetchPlan",
    "BackfillPlanResult",
    "build_backfill_plan",
]
