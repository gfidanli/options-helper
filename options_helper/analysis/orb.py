from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_MARKET_TZ = ZoneInfo("America/New_York")
_BASE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "timestamp_market",
    "session_date",
    "row_position",
)
_SIGNAL_COLUMNS: tuple[str, ...] = (
    "orb_opening_range_high",
    "orb_opening_range_low",
    "orb_breakout_direction",
    "orb_signal",
    "orb_signal_confirmed_ts",
    "orb_entry_ts",
    "orb_stop_price",
    "orb_range_end_ts",
    "orb_cutoff_ts",
)


def normalize_intraday_ohlc_frame(intraday_bars: pd.DataFrame | None) -> pd.DataFrame:
    if intraday_bars is None or intraday_bars.empty:
        return pd.DataFrame(columns=list(_BASE_COLUMNS))

    ts_source: pd.Series | pd.DatetimeIndex
    if "timestamp" in intraday_bars.columns:
        ts_source = intraday_bars["timestamp"]
    elif "ts" in intraday_bars.columns:
        ts_source = intraday_bars["ts"]
    elif isinstance(intraday_bars.index, pd.DatetimeIndex):
        ts_source = intraday_bars.index
    else:
        raise ValueError("Missing intraday timestamp source; expected 'timestamp', 'ts', or DatetimeIndex")

    out = pd.DataFrame(index=intraday_bars.index)
    out["timestamp"] = pd.to_datetime(ts_source, errors="coerce", utc=True)
    for canonical in ("open", "high", "low", "close"):
        source = _resolve_price_column(intraday_bars, canonical)
        if source is None:
            raise ValueError(f"Missing intraday OHLC column: {canonical}")
        out[canonical] = pd.to_numeric(intraday_bars[source], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if out.empty:
        return pd.DataFrame(columns=list(_BASE_COLUMNS))

    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out["timestamp_market"] = out["timestamp"].dt.tz_convert(_MARKET_TZ)
    minute_of_day = (out["timestamp_market"].dt.hour * 60) + out["timestamp_market"].dt.minute
    regular_mask = minute_of_day.ge(9 * 60 + 30) & minute_of_day.lt(16 * 60)
    out = out.loc[regular_mask].copy()
    if out.empty:
        return pd.DataFrame(columns=list(_BASE_COLUMNS))

    out = out.reset_index(drop=True)
    out["session_date"] = out["timestamp_market"].dt.date
    out["row_position"] = np.arange(len(out), dtype="int64")
    return out.reindex(columns=list(_BASE_COLUMNS))


def compute_orb_signals(
    intraday_bars: pd.DataFrame | None,
    *,
    range_minutes: int = 15,
    cutoff_et: str = "10:30",
) -> pd.DataFrame:
    range_window = int(range_minutes)
    if range_window < 1:
        raise ValueError("range_minutes must be >= 1")

    cutoff_time = _parse_cutoff_time(cutoff_et)
    out = normalize_intraday_ohlc_frame(intraday_bars).copy()

    out["orb_opening_range_high"] = np.nan
    out["orb_opening_range_low"] = np.nan
    out["orb_breakout_direction"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["orb_signal"] = False
    out["orb_signal_confirmed_ts"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["orb_entry_ts"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["orb_stop_price"] = np.nan
    out["orb_range_end_ts"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["orb_cutoff_ts"] = pd.Series([None] * len(out), index=out.index, dtype="object")

    if out.empty:
        return out.reindex(columns=[*list(_BASE_COLUMNS), *list(_SIGNAL_COLUMNS)])

    bar_duration = _infer_bar_duration(out)
    # Model close-confirmed signals as just-before-next-bar-open so entry anchor is strictly
    # after confirmation while preserving anti-lookahead semantics.
    confirmation_offset = max(bar_duration - pd.Timedelta(microseconds=1), pd.Timedelta(microseconds=1))

    grouped = out.groupby("session_date", sort=True)
    for session_day, session in grouped:
        session_open_market = _session_timestamp(session_day, time(9, 30))
        range_end_market = session_open_market + pd.Timedelta(minutes=range_window)
        cutoff_market = _session_timestamp(session_day, cutoff_time)
        range_end_utc = range_end_market.tz_convert("UTC")
        cutoff_utc = cutoff_market.tz_convert("UTC")

        opening = session.loc[session["timestamp_market"] < range_end_market]
        if opening.empty:
            continue

        opening_range_high = float(opening["high"].max())
        opening_range_low = float(opening["low"].min())
        session_index = list(session.index)
        out.loc[session_index, "orb_opening_range_high"] = opening_range_high
        out.loc[session_index, "orb_opening_range_low"] = opening_range_low
        out.loc[session_index, "orb_range_end_ts"] = range_end_utc
        out.loc[session_index, "orb_cutoff_ts"] = cutoff_utc

        for local_pos, row_index in enumerate(session_index):
            row_market_ts = pd.Timestamp(session.at[row_index, "timestamp_market"])
            if row_market_ts < range_end_market:
                continue

            close_value = float(session.at[row_index, "close"])
            direction: str | None = None
            stop_price: float | None = None
            if close_value > opening_range_high:
                direction = "long"
                stop_price = opening_range_low
            elif close_value < opening_range_low:
                direction = "short"
                stop_price = opening_range_high
            if direction is None or stop_price is None:
                continue

            signal_ts = pd.Timestamp(session.at[row_index, "timestamp"])
            signal_confirmed_ts = signal_ts + confirmation_offset
            if signal_confirmed_ts > cutoff_utc:
                break

            if local_pos + 1 >= len(session_index):
                continue
            entry_index = session_index[local_pos + 1]
            entry_ts = pd.Timestamp(session.at[entry_index, "timestamp"])
            if entry_ts <= signal_confirmed_ts:
                continue

            out.at[row_index, "orb_breakout_direction"] = direction
            out.at[row_index, "orb_signal"] = True
            out.at[row_index, "orb_signal_confirmed_ts"] = signal_confirmed_ts
            out.at[row_index, "orb_entry_ts"] = entry_ts
            out.at[row_index, "orb_stop_price"] = float(stop_price)
            break

    return out.reindex(columns=[*list(_BASE_COLUMNS), *list(_SIGNAL_COLUMNS)])


def extract_orb_signal_candidates(signals: pd.DataFrame | None) -> list[dict[str, Any]]:
    if signals is None or signals.empty or "orb_signal" not in signals.columns:
        return []

    selected = signals.loc[signals["orb_signal"].eq(True)]
    if selected.empty:
        return []

    candidates: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        direction = str(row.get("orb_breakout_direction") or "").strip().lower()
        if direction not in {"long", "short"}:
            continue

        signal_ts = _to_timestamp(row.get("timestamp"))
        signal_confirmed_ts = _to_timestamp(row.get("orb_signal_confirmed_ts"))
        entry_ts = _to_timestamp(row.get("orb_entry_ts"))
        if signal_ts is None or signal_confirmed_ts is None or entry_ts is None:
            continue
        if entry_ts <= signal_confirmed_ts:
            continue

        candidates.append(
            {
                "row_position": _to_int(row.get("row_position"), default=-1),
                "direction": direction,
                "timestamp": signal_ts.isoformat(),
                "signal_ts": signal_ts,
                "signal_confirmed_ts": signal_confirmed_ts,
                "entry_ts": entry_ts,
                "session_date": str(row.get("session_date")),
                "candle_open": _to_float(row.get("open")),
                "candle_high": _to_float(row.get("high")),
                "candle_low": _to_float(row.get("low")),
                "candle_close": _to_float(row.get("close")),
                "opening_range_high": _to_float(row.get("orb_opening_range_high")),
                "opening_range_low": _to_float(row.get("orb_opening_range_low")),
                "stop_price": _to_float(row.get("orb_stop_price")),
                "range_end_ts": _to_timestamp(row.get("orb_range_end_ts")),
                "cutoff_ts": _to_timestamp(row.get("orb_cutoff_ts")),
            }
        )

    candidates.sort(
        key=lambda row: (
            int(pd.Timestamp(row["signal_confirmed_ts"]).value),
            int(pd.Timestamp(row["signal_ts"]).value),
            int(pd.Timestamp(row["entry_ts"]).value),
            str(row["direction"]),
        )
    )
    return candidates


def _resolve_price_column(frame: pd.DataFrame, canonical: str) -> str | None:
    alias_map = {str(column).lower(): str(column) for column in frame.columns}
    if canonical in frame.columns:
        return canonical
    return alias_map.get(canonical)


def _parse_cutoff_time(value: str) -> time:
    text = str(value).strip()
    try:
        parsed = datetime.strptime(text, "%H:%M")
    except ValueError as exc:
        raise ValueError("cutoff_et must be HH:MM in 24-hour time") from exc
    return parsed.time()


def _session_timestamp(session_day: object, local_time: time) -> pd.Timestamp:
    day = pd.Timestamp(session_day).date()
    return pd.Timestamp.combine(day, local_time).tz_localize(_MARKET_TZ)


def _infer_bar_duration(frame: pd.DataFrame) -> pd.Timedelta:
    if frame.empty:
        return pd.Timedelta(minutes=1)

    deltas: list[int] = []
    grouped = frame.groupby("session_date", sort=False)
    for _, session in grouped:
        ts_ns = pd.to_datetime(session["timestamp"], errors="coerce", utc=True).astype("int64").to_numpy()
        if ts_ns.size < 2:
            continue
        diff = np.diff(ts_ns)
        positive = diff[diff > 0]
        if positive.size:
            deltas.extend(int(value) for value in positive)

    if not deltas:
        return pd.Timedelta(minutes=1)
    return pd.Timedelta(int(np.min(np.asarray(deltas, dtype="int64"))), unit="ns")


def _to_timestamp(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(number):
        return None
    return number


def _to_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


__all__ = [
    "compute_orb_signals",
    "extract_orb_signal_candidates",
    "normalize_intraday_ohlc_frame",
]
