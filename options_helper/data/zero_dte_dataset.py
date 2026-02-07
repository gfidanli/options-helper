from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from options_helper.data.intraday_store import IntradayStore
from options_helper.db.warehouse import DuckDBWarehouse


DEFAULT_PROXY_UNDERLYING = "SPY"
DEFAULT_MARKET_TZ = "America/New_York"
_DEFAULT_STOCK_SPEC = ("stocks", "bars", "1Min")
_DEFAULT_OPTION_SPEC = ("options", "bars", "1Min")

_UNDERLYING_COLUMNS = (
    "timestamp",
    "timestamp_market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
)

_STATE_COLUMNS = (
    "session_date",
    "decision_ts",
    "decision_ts_market",
    "bar_ts",
    "bar_ts_market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "bar_age_seconds",
    "status",
    "is_half_day",
)


@dataclass(frozen=True)
class SessionWindow:
    session_date: date
    is_trading_day: bool
    is_half_day: bool
    market_open: datetime | None
    market_close: datetime | None
    close_reason: str | None = None


@dataclass(frozen=True)
class IntradayStateDataset:
    underlying_symbol: str
    proxy_symbol: str
    session: SessionWindow
    underlying_bars: pd.DataFrame
    state_rows: pd.DataFrame
    option_snapshot: pd.DataFrame
    option_bars: pd.DataFrame
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ZeroDTEIntradayDatasetLoader:
    intraday_store: IntradayStore
    market_tz_name: str = DEFAULT_MARKET_TZ
    options_snapshot_store: Any | None = None
    warehouse: DuckDBWarehouse | None = None

    @property
    def market_tz(self) -> ZoneInfo:
        return ZoneInfo(self.market_tz_name)

    def load_day(
        self,
        session_date: date,
        *,
        underlying_symbol: str = DEFAULT_PROXY_UNDERLYING,
        decision_times: Sequence[str | time | datetime] | None = None,
        option_contract_symbols: Sequence[str] | None = None,
        include_option_snapshot: bool = False,
        include_option_bars: bool = False,
        stock_timeframe: str = "1Min",
        option_timeframe: str = "1Min",
    ) -> IntradayStateDataset:
        symbol = _normalize_symbol(underlying_symbol, default=DEFAULT_PROXY_UNDERLYING)
        session = build_us_equity_session(session_date, market_tz=self.market_tz)
        notes: list[str] = []

        underlying_bars = self.load_underlying_bars(
            session_date,
            underlying_symbol=symbol,
            timeframe=stock_timeframe,
            session=session,
        )
        if underlying_bars.empty:
            notes.append(f"No underlying bars found for {symbol} on {session_date.isoformat()}.")

        state_rows = build_state_rows(
            session=session,
            underlying_bars=underlying_bars,
            market_tz=self.market_tz,
            decision_times=decision_times,
        )

        option_snapshot = pd.DataFrame()
        if include_option_snapshot:
            option_snapshot = self._load_option_snapshot(symbol, session_date, notes)

        option_bars = pd.DataFrame()
        if include_option_bars and option_contract_symbols:
            option_bars = self._load_option_bars(
                session_date,
                option_contract_symbols,
                timeframe=option_timeframe,
                notes=notes,
            )

        return IntradayStateDataset(
            underlying_symbol=symbol,
            proxy_symbol=DEFAULT_PROXY_UNDERLYING,
            session=session,
            underlying_bars=underlying_bars,
            state_rows=state_rows,
            option_snapshot=option_snapshot,
            option_bars=option_bars,
            notes=tuple(notes),
        )

    def load_underlying_bars(
        self,
        session_date: date,
        *,
        underlying_symbol: str = DEFAULT_PROXY_UNDERLYING,
        timeframe: str = "1Min",
        session: SessionWindow | None = None,
    ) -> pd.DataFrame:
        resolved_session = session or build_us_equity_session(session_date, market_tz=self.market_tz)
        if not resolved_session.is_trading_day:
            return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

        raw = self.intraday_store.load_partition(
            _DEFAULT_STOCK_SPEC[0],
            _DEFAULT_STOCK_SPEC[1],
            timeframe,
            underlying_symbol,
            session_date,
        )
        normalized = normalize_intraday_bars(raw, market_tz=self.market_tz)
        if normalized.empty:
            return normalized

        open_ts = pd.Timestamp(resolved_session.market_open)
        close_ts = pd.Timestamp(resolved_session.market_close)
        in_session = (normalized["timestamp_market"] >= open_ts) & (
            normalized["timestamp_market"] <= close_ts
        )
        out = normalized.loc[in_session].copy()
        if out.empty:
            return pd.DataFrame(columns=_UNDERLYING_COLUMNS)
        return out.reset_index(drop=True)

    def _load_option_snapshot(self, underlying_symbol: str, session_date: date, notes: list[str]) -> pd.DataFrame:
        store = self.options_snapshot_store
        if store is None or not hasattr(store, "load_day"):
            notes.append("Option snapshot store unavailable; returning empty snapshot frame.")
            return pd.DataFrame()
        try:
            loaded = store.load_day(underlying_symbol, session_date)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Option snapshot load failed for {underlying_symbol}: {exc}")
            return pd.DataFrame()
        if loaded is None:
            return pd.DataFrame()
        return loaded.copy()

    def _load_option_bars(
        self,
        session_date: date,
        contract_symbols: Sequence[str],
        *,
        timeframe: str,
        notes: list[str],
    ) -> pd.DataFrame:
        normalized_contracts = [_normalize_symbol(contract) for contract in contract_symbols if str(contract).strip()]
        frames: list[pd.DataFrame] = []

        for contract in normalized_contracts:
            part = self.intraday_store.load_partition(
                _DEFAULT_OPTION_SPEC[0],
                _DEFAULT_OPTION_SPEC[1],
                timeframe,
                contract,
                session_date,
            )
            if part is None or part.empty:
                continue
            normalized = normalize_intraday_bars(part, market_tz=self.market_tz)
            if normalized.empty:
                continue
            normalized["contract_symbol"] = contract
            frames.append(normalized)

        if frames:
            return pd.concat(frames, ignore_index=True)

        warehouse = self.warehouse
        if warehouse is None:
            return pd.DataFrame()
        if not _duckdb_table_exists(warehouse, "option_bars"):
            notes.append("DuckDB table missing: option_bars")
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in normalized_contracts)
        if not placeholders:
            return pd.DataFrame()

        start_local = datetime.combine(session_date, time.min, tzinfo=self.market_tz)
        end_local = datetime.combine(session_date + timedelta(days=1), time.min, tzinfo=self.market_tz)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

        try:
            loaded = warehouse.fetch_df(
                f"""
                SELECT
                  contract_symbol,
                  ts AS timestamp,
                  open,
                  high,
                  low,
                  close,
                  volume,
                  vwap,
                  trade_count
                FROM option_bars
                WHERE interval = ?
                  AND contract_symbol IN ({placeholders})
                  AND ts >= ?
                  AND ts < ?
                ORDER BY contract_symbol ASC, ts ASC
                """,
                [timeframe.lower(), *normalized_contracts, start_utc, end_utc],
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"DuckDB option bars query failed: {exc}")
            return pd.DataFrame()

        if loaded is None or loaded.empty:
            return pd.DataFrame()
        if "contract_symbol" not in loaded.columns:
            return normalize_intraday_bars(loaded, market_tz=self.market_tz)

        frames = []
        for contract_symbol, sub in loaded.groupby("contract_symbol", sort=False):
            normalized = normalize_intraday_bars(sub, market_tz=self.market_tz)
            if normalized.empty:
                continue
            normalized["contract_symbol"] = str(contract_symbol)
            frames.append(normalized)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def normalize_intraday_bars(df: pd.DataFrame | None, *, market_tz: ZoneInfo) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

    table = df.copy()
    table.columns = [str(col) for col in table.columns]
    ts_col = _first_present(table.columns, "timestamp", "ts", "time")
    if ts_col is None:
        return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

    table["timestamp"] = pd.to_datetime(table[ts_col], errors="coerce", utc=True)
    table = table.loc[~table["timestamp"].isna()].copy()
    if table.empty:
        return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

    table["timestamp_market"] = table["timestamp"].dt.tz_convert(market_tz)

    for col in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
        else:
            table[col] = pd.NA

    table = table.sort_values("timestamp", kind="mergesort")
    table = table.drop_duplicates(subset=["timestamp"], keep="last")

    out = table.loc[:, list(_UNDERLYING_COLUMNS)].copy()
    return out.reset_index(drop=True)


def build_state_rows(
    *,
    session: SessionWindow,
    underlying_bars: pd.DataFrame,
    market_tz: ZoneInfo,
    decision_times: Sequence[str | time | datetime] | None = None,
) -> pd.DataFrame:
    if decision_times is None:
        decisions = [pd.Timestamp(ts) for ts in underlying_bars.get("timestamp_market", pd.Series(dtype=object)).tolist()]
    else:
        decisions = [
            _coerce_decision_timestamp(item, session_date=session.session_date, market_tz=market_tz)
            for item in decision_times
        ]

    rows: list[dict[str, object]] = []
    for decision_ts_market in decisions:
        rows.append(
            _build_state_row(
                session=session,
                underlying_bars=underlying_bars,
                decision_ts_market=decision_ts_market,
            )
        )

    if not rows:
        return pd.DataFrame(columns=_STATE_COLUMNS)
    return pd.DataFrame(rows, columns=_STATE_COLUMNS)


def _build_state_row(
    *,
    session: SessionWindow,
    underlying_bars: pd.DataFrame,
    decision_ts_market: pd.Timestamp,
) -> dict[str, object]:
    decision_utc = decision_ts_market.tz_convert("UTC")
    base: dict[str, object] = {
        "session_date": session.session_date.isoformat(),
        "decision_ts": decision_utc,
        "decision_ts_market": decision_ts_market,
        "bar_ts": pd.NaT,
        "bar_ts_market": pd.NaT,
        "open": pd.NA,
        "high": pd.NA,
        "low": pd.NA,
        "close": pd.NA,
        "volume": pd.NA,
        "vwap": pd.NA,
        "trade_count": pd.NA,
        "bar_age_seconds": pd.NA,
        "status": "no_underlying_data",
        "is_half_day": session.is_half_day,
    }

    if not session.is_trading_day or session.market_open is None or session.market_close is None:
        base["status"] = "market_closed"
        return base

    open_ts = pd.Timestamp(session.market_open)
    close_ts = pd.Timestamp(session.market_close)
    if decision_ts_market < open_ts or decision_ts_market > close_ts:
        base["status"] = "outside_session"
        return base

    if underlying_bars.empty:
        return base

    eligible = underlying_bars.loc[underlying_bars["timestamp_market"] <= decision_ts_market]
    if eligible.empty:
        base["status"] = "no_prior_bar"
        return base

    row = eligible.iloc[-1]
    bar_market_ts = pd.Timestamp(row["timestamp_market"])
    bar_utc_ts = pd.Timestamp(row["timestamp"])
    base.update(
        {
            "bar_ts": bar_utc_ts,
            "bar_ts_market": bar_market_ts,
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
            "vwap": row.get("vwap"),
            "trade_count": row.get("trade_count"),
            "bar_age_seconds": float((decision_ts_market - bar_market_ts).total_seconds()),
            "status": "ok",
        }
    )
    return base


def build_us_equity_session(session_date: date, *, market_tz: ZoneInfo) -> SessionWindow:
    if session_date.weekday() >= 5:
        return SessionWindow(
            session_date=session_date,
            is_trading_day=False,
            is_half_day=False,
            market_open=None,
            market_close=None,
            close_reason="weekend",
        )

    holidays = _us_equity_holidays(session_date.year)
    if session_date in holidays:
        return SessionWindow(
            session_date=session_date,
            is_trading_day=False,
            is_half_day=False,
            market_open=None,
            market_close=None,
            close_reason="holiday",
        )

    half_days = _us_equity_half_days(session_date.year)
    is_half_day = session_date in half_days
    close_time = time(13, 0) if is_half_day else time(16, 0)

    return SessionWindow(
        session_date=session_date,
        is_trading_day=True,
        is_half_day=is_half_day,
        market_open=datetime.combine(session_date, time(9, 30), tzinfo=market_tz),
        market_close=datetime.combine(session_date, close_time, tzinfo=market_tz),
        close_reason="half_day" if is_half_day else "regular",
    )


def _us_equity_holidays(year: int) -> set[date]:
    holidays: set[date] = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, calendar.MONDAY, 3),  # MLK day
        _nth_weekday(year, 2, calendar.MONDAY, 3),  # Presidents day
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, calendar.MONDAY),  # Memorial day
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, calendar.MONDAY, 1),  # Labor day
        _nth_weekday(year, 11, calendar.THURSDAY, 4),  # Thanksgiving
        _observed_fixed_holiday(year, 12, 25),
    }

    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))

    return holidays


def _us_equity_half_days(year: int) -> set[date]:
    holidays = _us_equity_holidays(year)
    half_days: set[date] = set()

    thanksgiving = _nth_weekday(year, 11, calendar.THURSDAY, 4)
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


def _coerce_decision_timestamp(
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

    if _looks_like_time_only(raw):
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


def _looks_like_time_only(raw: str) -> bool:
    if "T" in raw or " " in raw:
        return False
    parts = raw.split(":")
    return len(parts) in {2, 3}


def _first_present(columns: Sequence[str], *candidates: str) -> str | None:
    existing = set(columns)
    for name in candidates:
        if name in existing:
            return name
    return None


def _normalize_symbol(value: str, *, default: str | None = None) -> str:
    text = str(value or "").strip().upper()
    if text:
        return text
    if default is not None:
        return str(default).strip().upper()
    raise ValueError("symbol required")


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    observed = date(year, month, day)
    if observed.weekday() == calendar.SATURDAY:
        return observed - timedelta(days=1)
    if observed.weekday() == calendar.SUNDAY:
        return observed + timedelta(days=1)
    return observed


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    first = date(year, month, 1)
    first_delta = (weekday - first.weekday()) % 7
    day_of_month = 1 + first_delta + ((nth - 1) * 7)
    return date(year, month, day_of_month)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    out = date(year, month, last_day)
    delta = (out.weekday() - weekday) % 7
    return out - timedelta(days=delta)


def _easter_sunday(year: int) -> date:
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


def _duckdb_table_exists(warehouse: DuckDBWarehouse, table_name: str) -> bool:
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


__all__ = [
    "DEFAULT_MARKET_TZ",
    "DEFAULT_PROXY_UNDERLYING",
    "IntradayStateDataset",
    "SessionWindow",
    "ZeroDTEIntradayDatasetLoader",
    "build_state_rows",
    "build_us_equity_session",
    "normalize_intraday_bars",
]
