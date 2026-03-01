from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import re

import pandas as pd

from options_helper.data.intraday_store import IntradayStore

logger = logging.getLogger(__name__)

_INTRADAY_KIND = "stocks"
_INTRADAY_DATASET = "bars"
_CANDLE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
_INTERVAL_RE = re.compile(r"^(?P<count>\d+)\s*(?P<unit>min|m|h|hr|hour|hours)$")


@dataclass(frozen=True)
class IntradayCoverage:
    symbol: str
    base_timeframe: str
    target_interval: str
    requested_days: tuple[date, ...]
    loaded_days: tuple[date, ...]
    missing_days: tuple[date, ...]
    empty_days: tuple[date, ...]
    loaded_row_count: int
    output_row_count: int

    @property
    def requested_day_count(self) -> int:
        return len(self.requested_days)

    @property
    def loaded_day_count(self) -> int:
        return len(self.loaded_days)

    @property
    def missing_day_count(self) -> int:
        return len(self.missing_days)

    @property
    def empty_day_count(self) -> int:
        return len(self.empty_days)


@dataclass(frozen=True)
class IntradayCandleLoadResult:
    candles: pd.DataFrame
    coverage: IntradayCoverage


@dataclass(frozen=True)
class _LoadedIntradayParts:
    loaded_parts: tuple[pd.DataFrame, ...]
    loaded_days: tuple[date, ...]
    missing_days: tuple[date, ...]
    empty_days: tuple[date, ...]
    loaded_rows: int


def load_intraday_candles(
    *,
    symbol: str,
    start_day: date,
    end_day: date,
    base_timeframe: str,
    target_interval: str,
    intraday_dir: Path = Path("data/intraday"),
    store: IntradayStore | None = None,
) -> IntradayCandleLoadResult:
    if end_day < start_day:
        raise ValueError(f"end_day ({end_day.isoformat()}) must be on/after start_day ({start_day.isoformat()})")
    symbol_text = _normalize_symbol(symbol)
    base_tf, base_minutes = _canonical_base_timeframe(base_timeframe)
    target_minutes = _parse_target_interval_minutes(target_interval)
    _validate_interval_compatibility(
        base_timeframe=base_tf,
        base_minutes=base_minutes,
        target_interval=target_interval,
        target_minutes=target_minutes,
    )

    intraday_store = store if store is not None else IntradayStore(Path(intraday_dir))
    requested_days = _inclusive_days(start_day, end_day)
    loaded = _load_daily_intraday_partitions(
        store=intraday_store,
        symbol=symbol_text,
        base_timeframe=base_tf,
        requested_days=requested_days,
    )
    candles = _merge_candle_parts(list(loaded.loaded_parts))
    if target_minutes != base_minutes:
        candles = _resample_candles(candles, target_minutes)

    coverage = IntradayCoverage(
        symbol=symbol_text,
        base_timeframe=base_tf,
        target_interval=f"{target_minutes}Min",
        requested_days=requested_days,
        loaded_days=loaded.loaded_days,
        missing_days=loaded.missing_days,
        empty_days=loaded.empty_days,
        loaded_row_count=loaded.loaded_rows,
        output_row_count=int(len(candles.index)),
    )
    return IntradayCandleLoadResult(candles=candles, coverage=coverage)


def _normalize_symbol(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("symbol must be non-empty")
    return text


def _canonical_base_timeframe(value: str) -> tuple[str, int]:
    minutes = _parse_target_interval_minutes(value)
    if minutes == 1:
        return "1Min", 1
    if minutes == 5:
        return "5Min", 5
    raise ValueError(f"base_timeframe must be '1Min' or '5Min'; got {value!r}")


def _parse_target_interval_minutes(value: str) -> int:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError("target interval must be non-empty")
    normalized = (
        text.replace("minutes", "min").replace("minute", "min").replace("mins", "min").replace(" ", "")
    )
    if normalized.endswith("m") and not normalized.endswith("min"):
        normalized = normalized[:-1] + "min"
    match = _INTERVAL_RE.match(normalized)
    if not match:
        raise ValueError(
            f"Unsupported interval {value!r}; expected values like '1Min', '5Min', '15m', '30m', or '1h'."
        )
    count = int(match.group("count"))
    if count <= 0:
        raise ValueError(f"Interval must be > 0; got {value!r}")
    unit = match.group("unit")
    return count * 60 if unit in {"h", "hr", "hour", "hours"} else count


def _validate_interval_compatibility(
    *,
    base_timeframe: str,
    base_minutes: int,
    target_interval: str,
    target_minutes: int,
) -> None:
    if target_minutes < base_minutes:
        raise ValueError(
            f"target_interval {target_interval!r} must be >= base_timeframe {base_timeframe!r}"
        )
    if target_minutes % base_minutes != 0:
        raise ValueError(
            f"target_interval {target_interval!r} must be an integer multiple of base_timeframe {base_timeframe!r}"
        )


def _inclusive_days(start_day: date, end_day: date) -> tuple[date, ...]:
    days = pd.date_range(start=start_day.isoformat(), end=end_day.isoformat(), freq="D")
    return tuple(day.date() for day in days)


def _load_daily_intraday_partitions(
    *,
    store: IntradayStore,
    symbol: str,
    base_timeframe: str,
    requested_days: tuple[date, ...],
) -> _LoadedIntradayParts:
    loaded_days: list[date] = []
    missing_days: list[date] = []
    empty_days: list[date] = []
    loaded_parts: list[pd.DataFrame] = []
    loaded_rows = 0

    for day in requested_days:
        status, normalized = _load_partition_for_day(
            store=store,
            symbol=symbol,
            base_timeframe=base_timeframe,
            day=day,
        )
        if status == "loaded":
            loaded_days.append(day)
            loaded_rows += int(len(normalized.index))
            loaded_parts.append(normalized)
        elif status == "empty":
            empty_days.append(day)
        else:
            missing_days.append(day)

    return _LoadedIntradayParts(
        loaded_parts=tuple(loaded_parts),
        loaded_days=tuple(loaded_days),
        missing_days=tuple(missing_days),
        empty_days=tuple(empty_days),
        loaded_rows=loaded_rows,
    )


def _load_partition_for_day(
    *,
    store: IntradayStore,
    symbol: str,
    base_timeframe: str,
    day: date,
) -> tuple[str, pd.DataFrame]:
    if not store.partition_path(_INTRADAY_KIND, _INTRADAY_DATASET, base_timeframe, symbol, day).exists():
        _warn_missing_partition(symbol=symbol, base_timeframe=base_timeframe, day=day)
        return "missing", _empty_candle_frame()

    if _partition_meta_reports_empty(store, symbol=symbol, base_timeframe=base_timeframe, day=day):
        _warn_empty_partition(symbol=symbol, base_timeframe=base_timeframe, day=day)
        return "empty", _empty_candle_frame()

    try:
        raw = store.load_partition(_INTRADAY_KIND, _INTRADAY_DATASET, base_timeframe, symbol, day)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to load intraday partition for %s %s on %s: %s",
            symbol,
            base_timeframe,
            day.isoformat(),
            exc,
        )
        return "missing", _empty_candle_frame()

    normalized = _normalize_intraday_partition(raw)
    if normalized.empty:
        _warn_empty_partition(symbol=symbol, base_timeframe=base_timeframe, day=day)
        return "empty", _empty_candle_frame()
    return "loaded", normalized


def _partition_meta_reports_empty(
    store: IntradayStore,
    *,
    symbol: str,
    base_timeframe: str,
    day: date,
) -> bool:
    try:
        meta = store.load_meta(_INTRADAY_KIND, _INTRADAY_DATASET, base_timeframe, symbol, day)
    except Exception:  # noqa: BLE001
        return False
    return _meta_reports_empty_partition(meta)


def _warn_missing_partition(*, symbol: str, base_timeframe: str, day: date) -> None:
    logger.warning(
        "Missing intraday partition for %s %s on %s (%s/%s).",
        symbol,
        base_timeframe,
        day.isoformat(),
        _INTRADAY_KIND,
        _INTRADAY_DATASET,
    )


def _warn_empty_partition(*, symbol: str, base_timeframe: str, day: date) -> None:
    logger.warning(
        "Skipping empty intraday partition for %s %s on %s.",
        symbol,
        base_timeframe,
        day.isoformat(),
    )


def _normalize_intraday_partition(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_candle_frame()

    out = frame.copy()
    lower_cols = {str(col).strip().lower(): col for col in out.columns}
    ts_col = lower_cols.get("timestamp") or lower_cols.get("ts") or lower_cols.get("time")
    if ts_col is None:
        return _empty_candle_frame()

    out["timestamp"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    for source, target in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
        out[target] = pd.to_numeric(out[lower_cols.get(source)], errors="coerce") if source in lower_cols else pd.NA
    out["Volume"] = pd.to_numeric(out[lower_cols.get("volume")], errors="coerce") if "volume" in lower_cols else pd.NA

    out = out.dropna(subset=["timestamp", "Open", "High", "Low", "Close"])
    if out.empty:
        return _empty_candle_frame()

    out["timestamp"] = out["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    out = out.sort_values(by="timestamp", kind="stable").drop_duplicates(subset=["timestamp"], keep="last")
    out = out.set_index("timestamp")
    out = out.loc[:, list(_CANDLE_COLUMNS)]
    return out


def _merge_candle_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return _empty_candle_frame()
    merged = pd.concat(parts, axis=0)
    merged = merged.sort_index(kind="stable")
    if merged.index.has_duplicates:
        merged = merged.loc[~merged.index.duplicated(keep="last")]
    return merged.loc[:, list(_CANDLE_COLUMNS)]


def _resample_candles(candles: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    if candles.empty:
        return candles.copy()
    grouped = candles.resample(f"{target_minutes}min", label="left", closed="left")
    out = grouped.agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    )
    volume = grouped["Volume"].sum(min_count=1)
    out["Volume"] = volume.reindex(out.index)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out.loc[:, list(_CANDLE_COLUMNS)]


def _meta_reports_empty_partition(meta: dict[str, object] | None) -> bool:
    if not meta:
        return False
    try:
        return int(meta.get("rows") or 0) <= 0
    except Exception:  # noqa: BLE001
        return False


def _empty_candle_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_CANDLE_COLUMNS))


__all__ = [
    "IntradayCandleLoadResult",
    "IntradayCoverage",
    "load_intraday_candles",
]
