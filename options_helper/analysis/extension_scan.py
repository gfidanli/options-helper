from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from options_helper.technicals_backtesting.extension_percentiles import rolling_percentile_rank
from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars


@dataclass(frozen=True)
class ExtensionScanResult:
    asof: str
    extension_atr: float | None
    percentile: float | None
    window_years: int | None
    window_bars: int | None
    history_bars: int


def _empty_extension_scan_result() -> ExtensionScanResult:
    return ExtensionScanResult(
        asof="-",
        extension_atr=None,
        percentile=None,
        window_years=None,
        window_bars=None,
        history_bars=0,
    )


def _resolve_window_years(
    *,
    available_bars: int,
    days_per_year: int,
    percentile_window_years: int | None,
) -> int:
    if percentile_window_years is not None:
        window_years = int(percentile_window_years)
        if window_years <= 0:
            raise ValueError("percentile_window_years must be >= 1")
        return window_years
    history_years = (float(available_bars) / float(days_per_year)) if days_per_year > 0 else 0.0
    return 1 if history_years < 5.0 else 3


def _extension_series(features: pd.DataFrame, cfg: dict) -> pd.Series:
    atr_window = int(cfg["indicators"]["atr"]["window_default"])
    sma_window = int(cfg["indicators"]["sma"]["window_default"])
    ext_col = f"extension_atr_{sma_window}_{atr_window}"
    if ext_col not in features.columns:
        raise ValueError(f"Missing extension column: {ext_col}")
    return features[ext_col].dropna()


def compute_current_extension_percentile(
    df: pd.DataFrame,
    cfg: dict,
    *,
    percentile_window_years: int | None = None,
) -> ExtensionScanResult:
    if df is None or df.empty:
        return _empty_extension_scan_result()

    features = compute_features(df, cfg)
    warmup = warmup_bars(cfg)
    if warmup > 0:
        features = features.iloc[warmup:]
    if features.empty:
        return _empty_extension_scan_result()
    ext_series = _extension_series(features, cfg)
    if ext_series.empty:
        return _empty_extension_scan_result()

    asof_idx = ext_series.index[-1]
    asof = asof_idx.date().isoformat() if isinstance(asof_idx, pd.Timestamp) else str(asof_idx)
    ext_val = float(ext_series.iloc[-1])
    ext_cfg = cfg.get("extension_percentiles", {})
    days_per_year = int(ext_cfg.get("days_per_year", 252))
    available_bars = int(ext_series.shape[0])
    window_years = _resolve_window_years(
        available_bars=available_bars,
        days_per_year=days_per_year,
        percentile_window_years=percentile_window_years,
    )
    window_bars = int(window_years * days_per_year) if days_per_year > 0 else available_bars
    window_bars = window_bars if available_bars >= window_bars else available_bars

    percentile = None
    if window_bars > 1:
        pct_series = rolling_percentile_rank(ext_series, window_bars)
        pct_val = pct_series.iloc[-1]
        if pct_val == pct_val:  # not NaN
            percentile = float(pct_val)

    return ExtensionScanResult(
        asof=asof,
        extension_atr=ext_val,
        percentile=percentile,
        window_years=window_years,
        window_bars=window_bars,
        history_bars=available_bars,
    )
