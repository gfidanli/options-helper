from __future__ import annotations

import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import typer

from options_helper.analysis.osi import normalize_underlying
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.market_types import DataFetchError


def parse_date(value: str) -> date:
    raw = str(value or "").strip()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD.") from exc


def read_exclude_symbols(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = normalize_underlying(line)
        if not token or token.startswith("#"):
            continue
        out.add(token)
    return out


def load_target_symbols(*, client: AlpacaClient, excludes: set[str], max_symbols: int | None) -> list[str]:
    request_cls, status_enum, class_enum = load_asset_request_types()
    request = request_cls(status=status_enum.ACTIVE, asset_class=class_enum.US_EQUITY)
    assets = client.trading_client.get_all_assets(request)
    symbols: list[str] = []
    for asset in assets or []:
        symbol = normalize_underlying(str(getattr(asset, "symbol", "") or ""))
        if not symbol or not bool(getattr(asset, "tradable", False)):
            continue
        if symbol in excludes:
            continue
        symbols.append(symbol)
    unique = sorted(set(symbols))
    return unique[: int(max_symbols)] if max_symbols is not None else unique


def resolve_end_day(*, client: AlpacaClient, market_tz, override: str | None) -> date:
    if override:
        return parse_date(override)
    today = datetime.now(market_tz).date()
    recent_days = load_market_days(client=client, start_day=today - timedelta(days=20), end_day=today)
    if not recent_days:
        raise typer.BadParameter("Unable to resolve Alpaca market calendar days for end date.")
    now_local = datetime.now(market_tz)
    if now_local.time() < dt_time(16, 5):
        prior = [day for day in recent_days if day < today]
        if prior:
            return prior[-1]
    eligible = [day for day in recent_days if day <= today]
    if not eligible:
        raise typer.BadParameter("No eligible completed market day found for requested window.")
    return eligible[-1]


def load_market_days(*, client: AlpacaClient, start_day: date, end_day: date) -> list[date]:
    request_cls = load_calendar_request_type()
    request = request_cls(start=start_day, end=end_day)
    rows = client.trading_client.get_calendar(request)
    days: list[date] = []
    for row in rows or []:
        value = getattr(row, "date", None)
        if isinstance(value, date):
            days.append(value)
    return sorted(set(days))


def resolve_first_day(daily_df: pd.DataFrame, *, market_tz) -> date | None:
    if daily_df is None or daily_df.empty:
        return None
    idx = pd.to_datetime(daily_df.index, errors="coerce", utc=True).dropna()
    if idx.empty:
        return None
    return idx.min().tz_convert(market_tz).date()


def iter_month_ranges(first_day: date, end_day: date) -> Iterable[tuple[str, date, date]]:
    cursor = date(first_day.year, first_day.month, 1)
    last_month = date(end_day.year, end_day.month, 1)
    while cursor <= last_month:
        month_start = cursor if cursor >= first_day else first_day
        next_month = next_month_start(cursor)
        month_end = min(end_day, next_month - timedelta(days=1))
        yield cursor.strftime("%Y-%m"), month_start, month_end
        cursor = next_month


def next_month_start(day: date) -> date:
    return date(day.year + (1 if day.month == 12 else 0), 1 if day.month == 12 else day.month + 1, 1)


def process_symbol_month(
    *,
    client: AlpacaClient,
    store: IntradayStore,
    symbol: str,
    year_month: str,
    month_start: date,
    month_end: date,
    existing_days: set[date],
    market_days: list[date],
    market_tz,
) -> dict[str, Any]:
    expected_days = {day for day in market_days if month_start <= day <= month_end}
    if expected_days and expected_days.issubset(existing_days):
        return month_record(symbol=symbol, year_month=year_month, status="skipped_existing")

    fetch_started = time.perf_counter()
    try:
        history = client.get_stock_bars(
            symbol,
            start=month_start,
            end=month_end,
            interval="1m",
            adjustment="raw",
        )
    except DataFetchError as exc:
        fetch_seconds = time.perf_counter() - fetch_started
        return month_record(
            symbol=symbol,
            year_month=year_month,
            status="error",
            fetch_seconds=fetch_seconds,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
    fetch_seconds = time.perf_counter() - fetch_started

    frame = stock_history_to_intraday_frame(history)
    if frame.empty:
        return month_record(symbol=symbol, year_month=year_month, status="no_data", fetch_seconds=fetch_seconds)

    write_seconds, written_days, skipped_days = write_symbol_days(
        store=store,
        symbol=symbol,
        frame=frame,
        existing_days=existing_days,
        market_tz=market_tz,
        provider_version=client.provider_version,
        feed=client.stock_feed,
    )
    return month_record(
        symbol=symbol,
        year_month=year_month,
        status="ok" if written_days > 0 else "skipped_existing",
        rows=len(frame.index),
        fetch_seconds=fetch_seconds,
        write_seconds=write_seconds,
        written_days=written_days,
        skipped_days=skipped_days,
    )


def write_symbol_days(
    *,
    store: IntradayStore,
    symbol: str,
    frame: pd.DataFrame,
    existing_days: set[date],
    market_tz,
    provider_version: str | None,
    feed: str | None,
) -> tuple[float, int, int]:
    write_seconds = 0.0
    written_days = 0
    skipped_days = 0
    frame = frame.copy()
    frame["market_day"] = frame["timestamp"].dt.tz_convert(market_tz).dt.date
    for day, part in frame.groupby("market_day", sort=True):
        if day in existing_days:
            skipped_days += 1
            continue
        payload = part.drop(columns=["market_day"]).reset_index(drop=True)
        meta = {
            "provider": "alpaca",
            "provider_version": provider_version,
            "request": {"timeframe": "1Min", "feed": feed},
        }
        started = time.perf_counter()
        store.save_partition("stocks", "bars", "1Min", symbol, day, payload, meta)
        write_seconds += time.perf_counter() - started
        written_days += 1
        existing_days.add(day)
    return write_seconds, written_days, skipped_days


def stock_history_to_intraday_frame(history: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    if history is None or history.empty:
        return pd.DataFrame(columns=columns)
    frame = history.reset_index().rename(
        columns={
            history.index.name or "index": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Trade Count": "trade_count",
            "VWAP": "vwap",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).copy()
    for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.NA
    return frame[columns].sort_values("timestamp").reset_index(drop=True)


def month_record(
    *,
    symbol: str,
    year_month: str,
    status: str,
    rows: int = 0,
    fetch_seconds: float = 0.0,
    write_seconds: float = 0.0,
    written_days: int = 0,
    skipped_days: int = 0,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "year_month": year_month,
        "status": status,
        "rows": int(rows),
        "fetch_seconds": round(float(fetch_seconds), 6),
        "write_seconds": round(float(write_seconds), 6),
        "total_seconds": round(float(fetch_seconds) + float(write_seconds), 6),
        "written_days": int(written_days),
        "skipped_days": int(skipped_days),
        "error_type": error_type,
        "error_message": error_message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_asset_request_types():
    try:
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter("Alpaca trading request classes are required for this command.") from exc
    return GetAssetsRequest, AssetStatus, AssetClass


def load_calendar_request_type():
    try:
        from alpaca.trading.requests import GetCalendarRequest
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter("Alpaca calendar request classes are required for this command.") from exc
    return GetCalendarRequest
