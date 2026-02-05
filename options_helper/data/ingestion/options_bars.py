from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Iterable

import pandas as pd

from options_helper.analysis.osi import normalize_underlying, parse_contract_symbol
from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df
from options_helper.data.option_bars import OptionBarsStore
from options_helper.data.ingestion.common import shift_years


_MIN_START_DATE = date(2000, 1, 1)


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
    chunks_attempted: int


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
    if status in {"forbidden", "not_found"}:
        return True
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

    start_ts = _coerce_dt(meta.get("start_ts") or meta.get("start"))
    end_ts = _coerce_dt(meta.get("end_ts") or meta.get("end"))
    if start_ts is None or end_ts is None:
        return False

    desired_start = datetime.combine(start, time.min)
    desired_end = datetime.combine(end, time.max)
    return start_ts <= desired_start and end_ts >= desired_end


def _error_status(exc: Exception) -> str:
    msg = str(exc).lower()
    if "403" in msg or "forbidden" in msg:
        return "forbidden"
    if "402" in msg or "payment required" in msg:
        return "forbidden"
    if "404" in msg or "not found" in msg:
        return "not_found"
    return "error"


def discover_option_contracts(
    client: AlpacaClient,
    *,
    underlyings: Iterable[str],
    exp_start: date,
    exp_end: date,
    page_limit: int | None = None,
    max_contracts: int | None = None,
    fail_fast: bool = False,
) -> ContractDiscoveryOutput:
    frames: list[pd.DataFrame] = []
    raw_by_symbol: dict[str, dict[str, Any]] = {}
    summaries: list[UnderlyingDiscoverySummary] = []

    total_contracts = 0
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
                contracts = client.list_option_contracts(
                    underlying,
                    exp_gte=window_start,
                    exp_lte=window_end,
                    page_limit=page_limit,
                )
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                status = "error"
                if fail_fast:
                    raise
                break

            if not contracts:
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
    chunk_size: int = 200,
    page_limit: int | None = None,
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
            chunks_attempted=0,
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
    chunks_attempted = 0
    planned_contracts = 0

    for expiry in sorted(expiry_groups.keys(), reverse=True):
        end = min(today, expiry)
        start_candidate = shift_years(expiry, -abs(int(lookback_years)))
        start_candidate = min(today, start_candidate)
        start = max(_MIN_START_DATE, start_candidate)
        if start > end:
            continue

        desired_symbols = sorted({s for s in expiry_groups.get(expiry, []) if s})
        if resume:
            filtered: list[str] = []
            for sym in desired_symbols:
                try:
                    meta = store.coverage(sym, interval="1d", provider=provider)
                except Exception:  # noqa: BLE001
                    meta = None
                if _coverage_satisfies(meta, start=start, end=end):
                    skipped_contracts += 1
                    continue
                filtered.append(sym)
            desired_symbols = filtered

        if not desired_symbols:
            continue

        planned_contracts += len(desired_symbols)

        def _process_chunk(chunk: list[str]) -> None:
            nonlocal ok_contracts, error_contracts, bars_rows, chunks_attempted
            if not chunk:
                return
            if dry_run:
                chunks_attempted += 1
                return

            chunks_attempted += 1
            try:
                df_bars = client.get_option_bars_daily_full(
                    chunk,
                    start=start,
                    end=end,
                    interval="1d",
                    chunk_size=max(len(chunk), 1),
                    page_limit=page_limit,
                )
            except Exception as exc:  # noqa: BLE001
                if fail_fast:
                    if len(chunk) == 1:
                        store.mark_meta_error(
                            chunk,
                            interval="1d",
                            provider=provider,
                            error=str(exc),
                            status=_error_status(exc),
                        )
                        error_contracts += 1
                    raise
                if len(chunk) == 1:
                    store.mark_meta_error(
                        chunk,
                        interval="1d",
                        provider=provider,
                        error=str(exc),
                        status=_error_status(exc),
                    )
                    error_contracts += 1
                    if fail_fast:
                        raise
                else:
                    mid = len(chunk) // 2
                    _process_chunk(chunk[:mid])
                    _process_chunk(chunk[mid:])
                return

            if df_bars is None or df_bars.empty:
                for sym in chunk:
                    store.mark_meta_error(
                        [sym],
                        interval="1d",
                        provider=provider,
                        error="no bars returned",
                        status="not_found",
                    )
                    error_contracts += 1
                return

            store.upsert_bars(df_bars, interval="1d", provider=provider)
            bars_rows += int(len(df_bars))

            df_bars = df_bars.copy()
            df_bars["contractSymbol"] = df_bars["contractSymbol"].map(
                lambda v: str(v).strip().upper() if v is not None else None
            )
            df_bars = df_bars.dropna(subset=["contractSymbol"])
            grouped = df_bars.groupby("contractSymbol")
            stats: dict[str, dict[str, Any]] = {}
            for sym, rows in grouped:
                stats[str(sym).upper()] = {
                    "rows": int(len(rows)),
                    "start": rows["ts"].min(),
                    "end": rows["ts"].max(),
                }

            for sym in chunk:
                info = stats.get(sym)
                if info:
                    store.mark_meta_success(
                        [sym],
                        interval="1d",
                        provider=provider,
                        rows=info["rows"],
                        start_ts=info["start"],
                        end_ts=info["end"],
                    )
                    ok_contracts += 1
                else:
                    store.mark_meta_error(
                        [sym],
                        interval="1d",
                        provider=provider,
                        error="no bars returned",
                        status="not_found",
                    )
                    error_contracts += 1

        for i in range(0, len(desired_symbols), max(1, chunk_size)):
            _process_chunk(desired_symbols[i : i + max(1, chunk_size)])

    return BarsBackfillSummary(
        total_contracts=total_contracts,
        total_expiries=total_expiries,
        planned_contracts=planned_contracts,
        skipped_contracts=skipped_contracts,
        ok_contracts=ok_contracts,
        error_contracts=error_contracts,
        bars_rows=bars_rows,
        chunks_attempted=chunks_attempted,
    )
