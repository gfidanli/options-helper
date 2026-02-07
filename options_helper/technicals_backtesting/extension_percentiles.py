from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExtensionQuantiles:
    years: int
    window_bars: int
    p5: float | None
    p50: float | None
    p95: float | None


@dataclass(frozen=True)
class ExtensionTailEvent:
    date: str
    percentile: float
    extension_atr: float
    close: float
    direction: str  # "high" or "low"
    forward_extension_percentiles: dict[int, float | None]
    forward_returns: dict[int, float | None]


@dataclass(frozen=True)
class ExtensionPercentileReport:
    asof: str
    extension_atr: float | None
    current_percentiles: dict[int, float]
    quantiles_by_window: dict[int, ExtensionQuantiles]
    tail_window_years: int | None
    tail_events: list[ExtensionTailEvent]


@dataclass(frozen=True)
class ExtensionPercentilesBundle:
    daily: ExtensionPercentileReport
    weekly: ExtensionPercentileReport


def _percentile_rank(values: np.ndarray, value: float) -> float:
    """
    Percentile rank (0-100) using the "average rank" method.
    - If all values are equal, returns 50.
    """
    if values.size == 0:
        return float("nan")
    less = np.sum(values < value)
    equal = np.sum(values == value)
    rank = less + 0.5 * equal
    return float(rank / values.size * 100.0)


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    if series.empty or window <= 0:
        return pd.Series([], index=series.index, dtype="float64")

    def _pct(arr: np.ndarray) -> float:
        val = float(arr[-1])
        return _percentile_rank(arr, val)

    return series.rolling(window=window, min_periods=window).apply(_pct, raw=True)


def compute_extension_quantiles(series: pd.Series, window: int) -> tuple[float | None, float | None, float | None]:
    if series.empty or window <= 0 or len(series) < window:
        return (None, None, None)
    window_vals = series.iloc[-window:].dropna().to_numpy(dtype=float)
    if window_vals.size == 0:
        return (None, None, None)
    return (
        float(np.percentile(window_vals, 5)),
        float(np.percentile(window_vals, 50)),
        float(np.percentile(window_vals, 95)),
    )


def compute_extension_percentiles(
    *,
    extension_series: pd.Series,
    close_series: pd.Series,
    open_series: pd.Series | None = None,
    windows_years: Iterable[int],
    days_per_year: int,
    tail_high_pct: float,
    tail_low_pct: float,
    forward_days: Iterable[int],
    include_tail_events: bool = True,
) -> ExtensionPercentileReport:
    if extension_series.empty:
        return ExtensionPercentileReport(
            asof="-",
            extension_atr=None,
            current_percentiles={},
            quantiles_by_window={},
            tail_window_years=None,
            tail_events=[],
        )

    extension_series = extension_series.dropna()
    entry_series = close_series if open_series is None else open_series
    aligned = pd.concat(
        [
            extension_series.rename("extension"),
            entry_series.reindex(extension_series.index).rename("entry"),
            close_series.reindex(extension_series.index).rename("close"),
        ],
        axis=1,
    ).dropna(subset=["extension", "entry", "close"])
    if aligned.empty:
        return ExtensionPercentileReport(
            asof="-",
            extension_atr=None,
            current_percentiles={},
            quantiles_by_window={},
            tail_window_years=None,
            tail_events=[],
        )
    extension_series = aligned["extension"].astype("float64")
    entry_series = aligned["entry"].astype("float64")
    close_series = aligned["close"].astype("float64")

    asof_idx = extension_series.index[-1]
    asof = asof_idx.date().isoformat() if isinstance(asof_idx, pd.Timestamp) else str(asof_idx)
    current_val = float(extension_series.iloc[-1])

    windows = sorted({int(w) for w in windows_years if int(w) > 0})
    window_bars = {w: int(w * days_per_year) for w in windows}

    current_percentiles: dict[int, float] = {}
    quantiles_by_window: dict[int, ExtensionQuantiles] = {}

    for years, bars in window_bars.items():
        bars = bars if len(extension_series) >= bars else len(extension_series)
        if bars <= 1:
            continue
        pct_series = rolling_percentile_rank(extension_series, bars)
        pct_val = pct_series.iloc[-1]
        if not np.isnan(pct_val):
            current_percentiles[years] = float(pct_val)
        p5, p50, p95 = compute_extension_quantiles(extension_series, bars)
        quantiles_by_window[years] = ExtensionQuantiles(
            years=years,
            window_bars=bars,
            p5=p5,
            p50=p50,
            p95=p95,
        )

    # Tail events: use the longest available window for stability.
    tail_window_years = None
    tail_events: list[ExtensionTailEvent] = []
    if include_tail_events and current_percentiles:
        tail_window_years = max(current_percentiles.keys())
        bars = window_bars[tail_window_years]
        bars = bars if len(extension_series) >= bars else len(extension_series)
        pct_series = rolling_percentile_rank(extension_series, bars)
        for i, (idx, pct) in enumerate(pct_series.items()):
            if np.isnan(pct):
                continue
            if not (pct >= tail_high_pct or pct <= tail_low_pct):
                continue
            direction = "high" if pct >= tail_high_pct else "low"
            ext_val = float(extension_series.iloc[i])
            close_val = float(close_series.iloc[i]) if i < len(close_series) else float("nan")

            forward_extension_percentiles: dict[int, float | None] = {}
            forward_returns: dict[int, float | None] = {}
            for d in forward_days:
                j = i + int(d)
                entry_i = i + 1
                if j >= len(extension_series) or entry_i >= len(entry_series):
                    forward_extension_percentiles[int(d)] = None
                    forward_returns[int(d)] = None
                    continue
                f_pct = pct_series.iloc[j]
                forward_extension_percentiles[int(d)] = None if np.isnan(f_pct) else float(f_pct)
                if entry_i <= j and j < len(close_series):
                    c0 = float(entry_series.iloc[entry_i])
                    c1 = float(close_series.iloc[j])
                    forward_returns[int(d)] = (c1 / c0 - 1.0) if c0 else None
                else:
                    forward_returns[int(d)] = None

            tail_events.append(
                ExtensionTailEvent(
                    date=idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                    percentile=float(pct),
                    extension_atr=ext_val,
                    close=close_val,
                    direction=direction,
                    forward_extension_percentiles=forward_extension_percentiles,
                    forward_returns=forward_returns,
                )
            )

    return ExtensionPercentileReport(
        asof=asof,
        extension_atr=current_val,
        current_percentiles=current_percentiles,
        quantiles_by_window=quantiles_by_window,
        tail_window_years=tail_window_years,
        tail_events=tail_events,
    )


def build_weekly_extension_series(
    df_ohlc: pd.DataFrame, *, sma_window: int, atr_window: int, resample_rule: str
) -> tuple[pd.Series, pd.Series]:
    weekly = (
        df_ohlc[["Open", "High", "Low", "Close"]]
        .resample(resample_rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    close = weekly["Close"]
    sma = close.rolling(window=sma_window).mean()
    tr = pd.concat(
        [
            weekly["High"] - weekly["Low"],
            (weekly["High"] - close.shift(1)).abs(),
            (weekly["Low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window).mean()
    extension = (close - sma) / atr
    return extension, close
