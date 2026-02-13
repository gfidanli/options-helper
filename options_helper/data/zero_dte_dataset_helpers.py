from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from options_helper.db.warehouse import DuckDBWarehouse


def us_equity_holidays(year: int) -> set[date]:
    holidays: set[date] = {
        observed_fixed_holiday(year, 1, 1),
        nth_weekday(year, 1, calendar.MONDAY, 3),  # MLK day
        nth_weekday(year, 2, calendar.MONDAY, 3),  # Presidents day
        easter_sunday(year) - timedelta(days=2),  # Good Friday
        last_weekday(year, 5, calendar.MONDAY),  # Memorial day
        observed_fixed_holiday(year, 7, 4),
        nth_weekday(year, 9, calendar.MONDAY, 1),  # Labor day
        nth_weekday(year, 11, calendar.THURSDAY, 4),  # Thanksgiving
        observed_fixed_holiday(year, 12, 25),
    }

    if year >= 2022:
        holidays.add(observed_fixed_holiday(year, 6, 19))

    return holidays


def us_equity_half_days(year: int) -> set[date]:
    holidays = us_equity_holidays(year)
    half_days: set[date] = set()

    thanksgiving = nth_weekday(year, 11, calendar.THURSDAY, 4)
    black_friday = thanksgiving + timedelta(days=1)
    if black_friday.weekday() == calendar.FRIDAY and black_friday not in holidays:
        half_days.add(black_friday)

    july_3 = date(year, 7, 3)
    if july_3.weekday() < 5 and july_3 not in holidays:
        half_days.add(july_3)

    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in holidays:
        half_days.add(christmas_eve)

    return half_days


def coerce_decision_timestamp(
    value: str | time | datetime,
    *,
    session_date: date,
    market_tz: ZoneInfo,
) -> pd.Timestamp:
    if isinstance(value, time):
        dt = datetime.combine(session_date, value, tzinfo=market_tz)
        return pd.Timestamp(dt)

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=market_tz)
        else:
            dt = dt.astimezone(market_tz)
        return pd.Timestamp(dt)

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("decision time cannot be empty")

    if looks_like_time_only(raw):
        parsed_time = time.fromisoformat(raw)
        dt = datetime.combine(session_date, parsed_time, tzinfo=market_tz)
        return pd.Timestamp(dt)

    parsed = pd.to_datetime(raw, errors="raise")
    if isinstance(parsed, pd.Timestamp):
        ts = parsed
    else:
        ts = pd.Timestamp(parsed)

    if ts.tzinfo is None:
        ts = ts.tz_localize(market_tz)
    else:
        ts = ts.tz_convert(market_tz)
    return ts


def looks_like_time_only(raw: str) -> bool:
    if "T" in raw or " " in raw:
        return False
    parts = raw.split(":")
    return len(parts) in {2, 3}


def first_present(columns: Sequence[str], *candidates: str) -> str | None:
    existing = set(columns)
    for name in candidates:
        if name in existing:
            return name
    return None


def normalize_symbol(value: str, *, default: str | None = None) -> str:
    text = str(value or "").strip().upper()
    if text:
        return text
    if default is not None:
        return str(default).strip().upper()
    raise ValueError("symbol required")


def observed_fixed_holiday(year: int, month: int, day: int) -> date:
    observed = date(year, month, day)
    if observed.weekday() == calendar.SATURDAY:
        return observed - timedelta(days=1)
    if observed.weekday() == calendar.SUNDAY:
        return observed + timedelta(days=1)
    return observed


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    first_delta = (weekday - first.weekday()) % 7
    day_of_month = 1 + first_delta + ((nth - 1) * 7)
    return date(year, month, day_of_month)


def last_weekday(year: int, month: int, weekday: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    out = date(year, month, last_day)
    delta = (out.weekday() - weekday) % 7
    return out - timedelta(days=delta)


def easter_sunday(year: int) -> date:
    # Meeus/Jones/Butcher Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    leap_correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * leap_correction) // 451
    month = (h + leap_correction - 7 * m + 114) // 31
    day = ((h + leap_correction - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def duckdb_table_exists(warehouse: DuckDBWarehouse, table_name: str) -> bool:
    try:
        frame = warehouse.fetch_df(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            LIMIT 1
            """,
            [table_name],
        )
    except Exception:  # noqa: BLE001
        return False
    return frame is not None and not frame.empty
