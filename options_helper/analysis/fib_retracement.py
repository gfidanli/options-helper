from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import numpy as np
import pandas as pd

from options_helper.analysis.msb import compute_msb_signals


def normalize_fib_retracement_pct(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("fib_retracement_pct must be numeric in (0, 100].")

    try:
        numeric = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("fib_retracement_pct must be numeric in (0, 100].") from exc

    if not np.isfinite(numeric):
        raise ValueError("fib_retracement_pct must be finite.")
    if 0.0 < numeric <= 1.0:
        return float(numeric * 100.0)
    if 1.0 < numeric <= 100.0:
        return float(numeric)
    raise ValueError("fib_retracement_pct must be in (0, 1] ratio or (1, 100] percent form.")


@dataclass
class _LongSetup:
    msb_idx: int
    msb_timestamp: str
    range_low_idx: int
    range_low_level: float
    broken_swing_level: float
    broken_swing_timestamp: str | None
    range_high_idx: int | None = None
    range_high_level: float = np.nan
    entry_level: float = np.nan
    scan_start_idx: int | None = None


@dataclass
class _ShortSetup:
    msb_idx: int
    msb_timestamp: str
    range_high_idx: int
    range_high_level: float
    broken_swing_level: float
    broken_swing_timestamp: str | None
    range_low_idx: int | None = None
    range_low_level: float = np.nan
    entry_level: float = np.nan
    scan_start_idx: int | None = None


@dataclass
class _FibRuntime:
    out: pd.DataFrame
    index_values: list[object]
    high: np.ndarray
    low: np.ndarray
    swing_high: np.ndarray
    swing_low: np.ndarray
    bullish_msb: np.ndarray
    bearish_msb: np.ndarray
    broken_high_level: np.ndarray
    broken_low_level: np.ndarray
    broken_high_ts: list[object]
    broken_low_ts: list[object]
    last_confirmed_high_idx: np.ndarray
    last_confirmed_low_idx: np.ndarray
    swing_right_bars: int
    fib_ratio: float
    fib_percent: float


def compute_fib_retracement_signals(
    ohlc: pd.DataFrame,
    *,
    fib_retracement_pct: float = 61.8,
    timeframe: str | None = None,
) -> pd.DataFrame:
    fib_percent = normalize_fib_retracement_pct(fib_retracement_pct)
    fib_ratio = fib_percent / 100.0
    swing_right_bars = _msb_default_swing_right_bars()

    out = compute_msb_signals(ohlc, timeframe=timeframe)
    n = len(out)

    out["fib_retracement_long"] = False
    out["fib_retracement_short"] = False
    out["fib_entry_level"] = np.nan
    out["fib_retracement_pct"] = np.nan
    out["fib_range_high_level"] = np.nan
    out["fib_range_low_level"] = np.nan
    out["fib_range_high_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["fib_range_low_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["fib_msb_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["fib_broken_swing_level"] = np.nan
    out["fib_broken_swing_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")

    if n == 0:
        return out

    runtime = _build_fib_runtime(
        out=out,
        n=n,
        swing_right_bars=swing_right_bars,
        fib_ratio=fib_ratio,
        fib_percent=fib_percent,
    )
    long_setup: _LongSetup | None = None
    short_setup: _ShortSetup | None = None

    for i in range(n):
        long_setup = _advance_long_setup(runtime=runtime, row=i, setup=long_setup)
        short_setup = _advance_short_setup(runtime=runtime, row=i, setup=short_setup)
        long_setup = _emit_long_signal_if_triggered(runtime=runtime, row=i, setup=long_setup)
        short_setup = _emit_short_signal_if_triggered(runtime=runtime, row=i, setup=short_setup)

    return out


def _build_fib_runtime(
    *,
    out: pd.DataFrame,
    n: int,
    swing_right_bars: int,
    fib_ratio: float,
    fib_percent: float,
) -> _FibRuntime:
    high = pd.to_numeric(out["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["Low"], errors="coerce").to_numpy(dtype=float)
    swing_high = out["swing_high"].fillna(False).to_numpy(dtype=bool)
    swing_low = out["swing_low"].fillna(False).to_numpy(dtype=bool)
    bullish_msb = out["bullish_msb"].fillna(False).to_numpy(dtype=bool)
    bearish_msb = out["bearish_msb"].fillna(False).to_numpy(dtype=bool)
    broken_high_level = pd.to_numeric(out["broken_swing_high_level"], errors="coerce").to_numpy(dtype=float)
    broken_low_level = pd.to_numeric(out["broken_swing_low_level"], errors="coerce").to_numpy(dtype=float)
    broken_high_ts = out["broken_swing_high_timestamp"].tolist()
    broken_low_ts = out["broken_swing_low_timestamp"].tolist()
    last_high_idx, last_low_idx = _last_confirmed_swing_indices(
        n=n,
        swing_high=swing_high,
        swing_low=swing_low,
        swing_right_bars=swing_right_bars,
    )
    return _FibRuntime(
        out=out,
        index_values=list(out.index),
        high=high,
        low=low,
        swing_high=swing_high,
        swing_low=swing_low,
        bullish_msb=bullish_msb,
        bearish_msb=bearish_msb,
        broken_high_level=broken_high_level,
        broken_low_level=broken_low_level,
        broken_high_ts=broken_high_ts,
        broken_low_ts=broken_low_ts,
        last_confirmed_high_idx=last_high_idx,
        last_confirmed_low_idx=last_low_idx,
        swing_right_bars=swing_right_bars,
        fib_ratio=fib_ratio,
        fib_percent=fib_percent,
    )


def _last_confirmed_swing_indices(
    *,
    n: int,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    swing_right_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    last_confirmed_high_idx = np.full(n, -1, dtype=int)
    last_confirmed_low_idx = np.full(n, -1, dtype=int)
    current_high_idx = -1
    current_low_idx = -1
    for i in range(n):
        confirm_idx = i - swing_right_bars
        if confirm_idx >= 0:
            if swing_high[confirm_idx]:
                current_high_idx = confirm_idx
            if swing_low[confirm_idx]:
                current_low_idx = confirm_idx
        last_confirmed_high_idx[i] = current_high_idx
        last_confirmed_low_idx[i] = current_low_idx
    return last_confirmed_high_idx, last_confirmed_low_idx


def _start_long_setup(runtime: _FibRuntime, msb_idx: int) -> _LongSetup | None:
    range_low_idx = int(runtime.last_confirmed_low_idx[msb_idx])
    if range_low_idx < 0:
        return None
    range_low_level = float(runtime.low[range_low_idx])
    if not np.isfinite(range_low_level):
        return None
    return _LongSetup(
        msb_idx=msb_idx,
        msb_timestamp=_timestamp_label(runtime.index_values[msb_idx]),
        range_low_idx=range_low_idx,
        range_low_level=range_low_level,
        broken_swing_level=float(runtime.broken_high_level[msb_idx]),
        broken_swing_timestamp=_optional_timestamp_label(runtime.broken_high_ts[msb_idx]),
    )


def _start_short_setup(runtime: _FibRuntime, msb_idx: int) -> _ShortSetup | None:
    range_high_idx = int(runtime.last_confirmed_high_idx[msb_idx])
    if range_high_idx < 0:
        return None
    range_high_level = float(runtime.high[range_high_idx])
    if not np.isfinite(range_high_level):
        return None
    return _ShortSetup(
        msb_idx=msb_idx,
        msb_timestamp=_timestamp_label(runtime.index_values[msb_idx]),
        range_high_idx=range_high_idx,
        range_high_level=range_high_level,
        broken_swing_level=float(runtime.broken_low_level[msb_idx]),
        broken_swing_timestamp=_optional_timestamp_label(runtime.broken_low_ts[msb_idx]),
    )


def _advance_long_setup(*, runtime: _FibRuntime, row: int, setup: _LongSetup | None) -> _LongSetup | None:
    if runtime.bullish_msb[row]:
        next_setup = _start_long_setup(runtime, row)
        if setup is None or setup.range_high_idx is None:
            # While waiting for the next swing high, latest bullish MSB wins.
            setup = next_setup
    if setup is None or setup.range_high_idx is not None:
        return setup

    confirm_idx = row - runtime.swing_right_bars
    if not (confirm_idx > setup.msb_idx and confirm_idx >= 0 and runtime.swing_high[confirm_idx]):
        return setup

    range_high_level = float(runtime.high[confirm_idx])
    if not np.isfinite(range_high_level) or not (range_high_level > setup.range_low_level):
        return None
    if np.isfinite(setup.broken_swing_level) and not (range_high_level > setup.broken_swing_level):
        return None
    setup.range_high_idx = confirm_idx
    setup.range_high_level = range_high_level
    setup.entry_level = range_high_level - ((range_high_level - setup.range_low_level) * runtime.fib_ratio)
    setup.scan_start_idx = confirm_idx + runtime.swing_right_bars + 1
    return setup


def _advance_short_setup(*, runtime: _FibRuntime, row: int, setup: _ShortSetup | None) -> _ShortSetup | None:
    if runtime.bearish_msb[row]:
        next_setup = _start_short_setup(runtime, row)
        if setup is None or setup.range_low_idx is None:
            # While waiting for the next swing low, latest bearish MSB wins.
            setup = next_setup
    if setup is None or setup.range_low_idx is not None:
        return setup

    confirm_idx = row - runtime.swing_right_bars
    if not (confirm_idx > setup.msb_idx and confirm_idx >= 0 and runtime.swing_low[confirm_idx]):
        return setup

    range_low_level = float(runtime.low[confirm_idx])
    if not np.isfinite(range_low_level) or not (setup.range_high_level > range_low_level):
        return None
    if np.isfinite(setup.broken_swing_level) and not (range_low_level < setup.broken_swing_level):
        return None
    setup.range_low_idx = confirm_idx
    setup.range_low_level = range_low_level
    setup.entry_level = range_low_level + ((setup.range_high_level - range_low_level) * runtime.fib_ratio)
    setup.scan_start_idx = confirm_idx + runtime.swing_right_bars + 1
    return setup


def _emit_long_signal_if_triggered(*, runtime: _FibRuntime, row: int, setup: _LongSetup | None) -> _LongSetup | None:
    if (
        setup is None
        or setup.range_high_idx is None
        or setup.scan_start_idx is None
        or row < setup.scan_start_idx
    ):
        return setup
    if not (
        np.isfinite(runtime.low[row])
        and np.isfinite(runtime.high[row])
        and runtime.low[row] <= setup.entry_level <= runtime.high[row]
    ):
        return setup
    if row + 1 < len(runtime.high):
        _write_signal_row(
            out=runtime.out,
            row=row,
            is_long=True,
            entry_level=setup.entry_level,
            fib_percent=runtime.fib_percent,
            range_high_level=setup.range_high_level,
            range_low_level=setup.range_low_level,
            range_high_timestamp=_timestamp_label(runtime.index_values[setup.range_high_idx]),
            range_low_timestamp=_timestamp_label(runtime.index_values[setup.range_low_idx]),
            msb_timestamp=setup.msb_timestamp,
            broken_swing_level=setup.broken_swing_level,
            broken_swing_timestamp=setup.broken_swing_timestamp,
        )
    return None


def _emit_short_signal_if_triggered(
    *,
    runtime: _FibRuntime,
    row: int,
    setup: _ShortSetup | None,
) -> _ShortSetup | None:
    if (
        setup is None
        or setup.range_low_idx is None
        or setup.scan_start_idx is None
        or row < setup.scan_start_idx
    ):
        return setup
    if not (
        np.isfinite(runtime.low[row])
        and np.isfinite(runtime.high[row])
        and runtime.low[row] <= setup.entry_level <= runtime.high[row]
    ):
        return setup
    if row + 1 < len(runtime.high):
        _write_signal_row(
            out=runtime.out,
            row=row,
            is_long=False,
            entry_level=setup.entry_level,
            fib_percent=runtime.fib_percent,
            range_high_level=setup.range_high_level,
            range_low_level=setup.range_low_level,
            range_high_timestamp=_timestamp_label(runtime.index_values[setup.range_high_idx]),
            range_low_timestamp=_timestamp_label(runtime.index_values[setup.range_low_idx]),
            msb_timestamp=setup.msb_timestamp,
            broken_swing_level=setup.broken_swing_level,
            broken_swing_timestamp=setup.broken_swing_timestamp,
        )
    return None


def _write_signal_row(
    *,
    out: pd.DataFrame,
    row: int,
    is_long: bool,
    entry_level: float,
    fib_percent: float,
    range_high_level: float,
    range_low_level: float,
    range_high_timestamp: str,
    range_low_timestamp: str,
    msb_timestamp: str,
    broken_swing_level: float,
    broken_swing_timestamp: str | None,
) -> None:
    if is_long:
        out.iat[row, out.columns.get_loc("fib_retracement_long")] = True
    else:
        out.iat[row, out.columns.get_loc("fib_retracement_short")] = True
    out.iat[row, out.columns.get_loc("fib_entry_level")] = float(entry_level)
    out.iat[row, out.columns.get_loc("fib_retracement_pct")] = float(fib_percent)
    out.iat[row, out.columns.get_loc("fib_range_high_level")] = float(range_high_level)
    out.iat[row, out.columns.get_loc("fib_range_low_level")] = float(range_low_level)
    out.iat[row, out.columns.get_loc("fib_range_high_timestamp")] = range_high_timestamp
    out.iat[row, out.columns.get_loc("fib_range_low_timestamp")] = range_low_timestamp
    out.iat[row, out.columns.get_loc("fib_msb_timestamp")] = msb_timestamp
    out.iat[row, out.columns.get_loc("fib_broken_swing_level")] = float(broken_swing_level)
    out.iat[row, out.columns.get_loc("fib_broken_swing_timestamp")] = broken_swing_timestamp


def _timestamp_label(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _optional_timestamp_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None
    return text


def _msb_default_swing_right_bars() -> int:
    try:
        default = signature(compute_msb_signals).parameters["swing_right_bars"].default
        right = int(default)
    except Exception:  # noqa: BLE001
        right = 1
    return max(1, right)


__all__ = [
    "compute_fib_retracement_signals",
    "normalize_fib_retracement_pct",
]
