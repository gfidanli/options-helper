from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from options_helper.data.intraday_store import IntradayStore

_INTRADAY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
)
_CHART_TIMEFRAME_MINUTES: tuple[int, ...] = (1, 5, 15, 30, 60)


@dataclass(frozen=True)
class ChartBarsResult:
    timeframe: str
    bars: pd.DataFrame
    warning: str | None
    skipped: bool


def extract_selected_rows(event: object) -> list[int]:
    if event is None:
        return []

    selection = _lookup_field(event, "selection")
    rows_payload = _lookup_field(selection, "rows")
    if rows_payload is None:
        rows_payload = _lookup_field(event, "selected_rows")

    if rows_payload is None:
        return []
    if isinstance(rows_payload, (str, bytes)) or not isinstance(rows_payload, Sequence):
        candidates = [rows_payload]
    else:
        candidates = list(rows_payload)

    out: list[int] = []
    seen: set[int] = set()
    for value in candidates:
        parsed = _coerce_nonnegative_int(value)
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def selected_trade_id_from_event(event: object, displayed_df: pd.DataFrame) -> str | None:
    if not isinstance(displayed_df, pd.DataFrame) or displayed_df.empty:
        return None
    if "trade_id" not in displayed_df.columns:
        return None

    selected_rows = extract_selected_rows(event)
    if not selected_rows:
        return None

    row_idx = selected_rows[0]
    if row_idx < 0 or row_idx >= len(displayed_df.index):
        return None

    value = displayed_df.iloc[row_idx].get("trade_id")
    if value is None or pd.isna(value):
        return None

    trade_id = str(value).strip()
    return trade_id or None


def load_intraday_window(
    store_root: str | Path,
    symbol: str,
    timeframe: str,
    start_ts: object,
    end_ts: object,
) -> pd.DataFrame:
    start = _coerce_utc_timestamp(start_ts)
    end = _coerce_utc_timestamp(end_ts)
    if start is None or end is None or end < start:
        return _empty_intraday_frame()

    symbol_text = str(symbol or "").strip()
    timeframe_text = str(timeframe or "").strip()
    if not symbol_text or not timeframe_text:
        return _empty_intraday_frame()

    store = IntradayStore(root_dir=Path(store_root))
    parts: list[pd.DataFrame] = []
    for day in _inclusive_days(start, end):
        try:
            part = store.load_partition("stocks", "bars", timeframe_text, symbol_text, day)
        except Exception:  # noqa: BLE001
            continue
        normalized = _normalize_intraday_frame(part)
        if not normalized.empty:
            parts.append(normalized)

    if not parts:
        return _empty_intraday_frame()

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    window = merged.loc[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)].copy()
    if window.empty:
        return _empty_intraday_frame()

    window = window.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    return window.reindex(columns=list(_INTRADAY_COLUMNS))


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    frame = _normalize_intraday_frame(df)
    if frame.empty:
        return _empty_intraday_frame()

    frequency = str(freq or "").strip()
    if not frequency:
        return _empty_intraday_frame()

    indexed = frame.set_index("timestamp")
    try:
        grouped = indexed.resample(frequency, label="left", closed="left")
    except Exception:  # noqa: BLE001
        return _empty_intraday_frame()

    out = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return _empty_intraday_frame()

    volume = grouped["volume"].sum(min_count=1).reindex(out.index).fillna(0.0).astype(float)
    trade_count = grouped["trade_count"].sum(min_count=1).reindex(out.index).fillna(0.0).astype(float)

    weighted_price = indexed["vwap"].where(indexed["vwap"].notna(), indexed["close"])
    weighted_notional = (
        (weighted_price * indexed["volume"].fillna(0.0))
        .resample(frequency, label="left", closed="left")
        .sum(min_count=1)
        .reindex(out.index)
        .fillna(0.0)
        .astype(float)
    )

    vwap_values = out["close"].to_numpy(dtype=float).copy()
    volume_values = volume.to_numpy(dtype=float)
    weighted_values = weighted_notional.to_numpy(dtype=float)
    positive_mask = volume_values > 0.0
    vwap_values[positive_mask] = weighted_values[positive_mask] / volume_values[positive_mask]

    out = out.copy()
    out["volume"] = volume
    out["trade_count"] = trade_count
    out["vwap"] = vwap_values

    out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    return out.reindex(columns=list(_INTRADAY_COLUMNS))


def supported_chart_timeframes(base_tf: str) -> list[str]:
    base_minutes = _timeframe_minutes(base_tf)
    default = [_timeframe_label(value) for value in _CHART_TIMEFRAME_MINUTES]

    if base_minutes is None or base_minutes <= 0:
        return default

    supported = [
        _timeframe_label(minutes)
        for minutes in _CHART_TIMEFRAME_MINUTES
        if minutes >= base_minutes and minutes % base_minutes == 0
    ]
    if supported:
        return supported
    return [_timeframe_label(base_minutes)]


def resample_for_chart(
    df: pd.DataFrame,
    *,
    base_timeframe: str,
    chart_timeframe: str,
    max_bars: int = 5000,
) -> ChartBarsResult:
    try:
        max_rows = int(max_bars)
    except Exception:  # noqa: BLE001
        max_rows = 5000
    if max_rows <= 0:
        max_rows = 5000
    supported = supported_chart_timeframes(base_timeframe)
    if not supported:
        return ChartBarsResult(
            timeframe=str(chart_timeframe or ""),
            bars=_empty_intraday_frame(),
            warning="No supported chart timeframes available for the selected base timeframe.",
            skipped=True,
        )

    requested = str(chart_timeframe or "").strip()
    selected = requested if requested in set(supported) else supported[0]

    candidate_timeframes = _upsample_candidates(supported, selected)
    sampled_by_timeframe: dict[str, pd.DataFrame] = {}

    selection_warning: str | None = None
    if requested and requested not in set(supported):
        selection_warning = (
            f"Requested chart timeframe {requested} is unsupported for base timeframe {base_timeframe}; "
            f"using {selected}."
        )

    for timeframe in candidate_timeframes:
        sampled = resample_ohlc(df, timeframe)
        sampled_by_timeframe[timeframe] = sampled
        if len(sampled.index) <= max_rows:
            warning = selection_warning
            if timeframe != selected:
                auto = (
                    f"Auto-adjusted chart timeframe from {selected} to {timeframe} "
                    f"to keep bars <= {max_rows}."
                )
                warning = auto if warning is None else f"{warning} {auto}"
            return ChartBarsResult(timeframe=timeframe, bars=sampled, warning=warning, skipped=False)

    widest = candidate_timeframes[-1]
    widest_rows = int(len(sampled_by_timeframe.get(widest, _empty_intraday_frame()).index))
    skip_msg = f"Skipping chart: {widest_rows} bars at {widest} exceeds limit {max_rows}."
    if selection_warning:
        skip_msg = f"{selection_warning} {skip_msg}"
    return ChartBarsResult(
        timeframe=widest,
        bars=_empty_intraday_frame(),
        warning=skip_msg,
        skipped=True,
    )


def _normalize_intraday_frame(frame: object) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _empty_intraday_frame()

    out = frame.copy()
    ts_col = "timestamp" if "timestamp" in out.columns else "ts" if "ts" in out.columns else None
    if ts_col is None:
        return _empty_intraday_frame()

    out["timestamp"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = np.nan

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if out.empty:
        return _empty_intraday_frame()

    out = out.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    return out.reindex(columns=list(_INTRADAY_COLUMNS))


def _lookup_field(value: object, field: str) -> object:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _coerce_nonnegative_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    parsed: int | None = None
    if isinstance(value, int):
        parsed = int(value)
    elif isinstance(value, float):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed_float = float(text)
        except ValueError:
            return None
        if not np.isfinite(parsed_float) or not parsed_float.is_integer():
            return None
        parsed = int(parsed_float)
    else:
        return None

    if parsed < 0:
        return None
    return parsed


def _coerce_utc_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(ts, pd.Timestamp) or pd.isna(ts):
        return None
    return ts


def _inclusive_days(start: pd.Timestamp, end: pd.Timestamp) -> list[date]:
    if end < start:
        return []
    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D", tz="UTC")
    return [value.date() for value in days]


def _timeframe_minutes(value: object) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None

    normalized = text.replace("minutes", "min").replace("minute", "min").replace("mins", "min")
    if normalized.endswith("m") and not normalized.endswith("min"):
        normalized = normalized[:-1] + "min"

    if normalized.endswith("min"):
        amount = normalized[:-3].strip()
    else:
        amount = normalized

    try:
        minutes = int(amount)
    except ValueError:
        return None

    if minutes <= 0:
        return None
    return minutes


def _timeframe_label(minutes: int) -> str:
    return f"{int(minutes)}Min"


def _upsample_candidates(supported: list[str], selected: str) -> list[str]:
    selected_minutes = _timeframe_minutes(selected)
    if selected_minutes is None:
        return list(supported)

    candidates = [
        timeframe
        for timeframe in supported
        if (_timeframe_minutes(timeframe) or 0) >= selected_minutes
    ]
    return candidates or list(supported)


def _empty_intraday_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))


__all__ = [
    "ChartBarsResult",
    "extract_selected_rows",
    "load_intraday_window",
    "resample_for_chart",
    "resample_ohlc",
    "selected_trade_id_from_event",
    "supported_chart_timeframes",
]
