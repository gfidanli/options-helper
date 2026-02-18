from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
import typer


@dataclass(frozen=True)
class PreparedFeatures:
    features: pd.DataFrame
    atr_window: int
    sma_window: int
    ext_col: str
    warmup_warning: bool


@dataclass(frozen=True)
class PreparedReports:
    report_daily: Any
    report_weekly: Any
    weekly_candles: pd.DataFrame
    weekly_ext: pd.Series
    weekly_close: pd.Series


@dataclass(frozen=True)
class PreparedSeries:
    ext_series_daily: pd.Series
    open_series_daily: pd.Series
    close_series_daily: pd.Series
    high_series_daily: pd.Series | None
    low_series_daily: pd.Series | None
    rsi_window: int | None
    rsi_series_daily: pd.Series | None
    weekly_open_series: pd.Series
    weekly_close_series: pd.Series
    weekly_high_series: pd.Series
    weekly_low_series: pd.Series
    weekly_rsi_series: pd.Series | None


def resolve_tail_thresholds(*, ext_cfg: dict[str, Any], tail_pct: float | None) -> tuple[float, float]:
    tail_high_cfg = float(ext_cfg.get("tail_high_pct", 97.5))
    tail_low_cfg = float(ext_cfg.get("tail_low_pct", 2.5))
    if tail_pct is None:
        tail_high_pct = tail_high_cfg
        tail_low_pct = tail_low_cfg
    else:
        tp = float(tail_pct)
        if tp < 0.0 or tp >= 50.0:
            raise typer.BadParameter("--tail-pct must be >= 0 and < 50")
        tail_low_pct = tp
        tail_high_pct = 100.0 - tp
    if tail_low_pct >= tail_high_pct:
        raise typer.BadParameter("Tail thresholds must satisfy low < high")
    return tail_low_pct, tail_high_pct


def resolve_window_years(
    *,
    available_bars: int,
    days_per_year: int,
    percentile_window_years: int | None,
) -> int:
    if percentile_window_years is None:
        history_years = (float(available_bars) / float(days_per_year)) if days_per_year > 0 else 0.0
        window_years = 1 if history_years < 5.0 else 3
    else:
        window_years = int(percentile_window_years)
    if window_years <= 0:
        raise typer.BadParameter("--percentile-window-years must be >= 1")
    return window_years


def resolve_forward_days(ext_cfg: dict[str, Any]) -> tuple[list[int], list[int]]:
    forward_days_base = [int(d) for d in (ext_cfg.get("forward_days", [1, 3, 5, 10]) or [])]
    forward_days_daily = [
        int(d)
        for d in (
            ext_cfg.get("forward_days_daily", None)
            or sorted({*forward_days_base, 15})
        )
    ]
    forward_days_weekly = [
        int(d) for d in (ext_cfg.get("forward_days_weekly", None) or forward_days_base or [1, 3, 5, 10])
    ]
    return forward_days_daily, forward_days_weekly


def prepare_features(*, df: pd.DataFrame, cfg: dict[str, Any]) -> PreparedFeatures:
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars

    features = compute_features(df, cfg)
    w = warmup_bars(cfg)
    warmup_warning = bool(w > 0 and len(features) <= w)
    if w > 0 and len(features) > w:
        features = features.iloc[w:]
    if features.empty:
        raise typer.BadParameter("No features after warmup; check candle history.")

    atr_window = int(cfg["indicators"]["atr"]["window_default"])
    sma_window = int(cfg["indicators"]["sma"]["window_default"])
    ext_col = f"extension_atr_{sma_window}_{atr_window}"
    if ext_col not in features.columns:
        raise typer.BadParameter(f"Missing extension column: {ext_col}")
    return PreparedFeatures(
        features=features,
        atr_window=atr_window,
        sma_window=sma_window,
        ext_col=ext_col,
        warmup_warning=warmup_warning,
    )


def prepare_reports(
    *,
    df: pd.DataFrame,
    prepared: PreparedFeatures,
    windows_years: list[int],
    days_per_year: int,
    tail_high_pct: float,
    tail_low_pct: float,
    forward_days_daily: list[int],
    forward_days_weekly: list[int],
    weekly_rule: str,
) -> PreparedReports:
    from options_helper.technicals_backtesting.extension_percentiles import (
        build_weekly_extension_series,
        compute_extension_percentiles,
    )

    report_daily = compute_extension_percentiles(
        extension_series=prepared.features[prepared.ext_col],
        open_series=prepared.features["Open"],
        close_series=prepared.features["Close"],
        windows_years=windows_years,
        days_per_year=days_per_year,
        tail_high_pct=tail_high_pct,
        tail_low_pct=tail_low_pct,
        forward_days=forward_days_daily,
        include_tail_events=True,
    )
    weekly_candles = (
        df[["Open", "High", "Low", "Close"]]
        .resample(weekly_rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    weekly_ext, weekly_close = build_weekly_extension_series(
        df[["Open", "High", "Low", "Close"]],
        sma_window=prepared.sma_window,
        atr_window=prepared.atr_window,
        resample_rule=weekly_rule,
    )
    report_weekly = compute_extension_percentiles(
        extension_series=weekly_ext,
        open_series=weekly_candles["Open"],
        close_series=weekly_close,
        windows_years=windows_years,
        days_per_year=int(days_per_year / 5),
        tail_high_pct=tail_high_pct,
        tail_low_pct=tail_low_pct,
        forward_days=forward_days_weekly,
        include_tail_events=True,
    )
    return PreparedReports(
        report_daily=report_daily,
        report_weekly=report_weekly,
        weekly_candles=weekly_candles,
        weekly_ext=weekly_ext,
        weekly_close=weekly_close,
    )


def apply_report_asof_fallbacks(*, reports: PreparedReports, df: pd.DataFrame) -> tuple[Any, Any]:
    report_daily = reports.report_daily
    report_weekly = reports.report_weekly
    if report_daily.asof == "-" and isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        try:
            fallback_daily = df.index.max().date().isoformat()
        except Exception:  # noqa: BLE001
            fallback_daily = None
        if fallback_daily:
            report_daily = replace(report_daily, asof=fallback_daily)

    if report_weekly.asof == "-" and not reports.weekly_close.empty:
        try:
            fallback_weekly = reports.weekly_close.index.max().date().isoformat()
        except Exception:  # noqa: BLE001
            fallback_weekly = None
        if fallback_weekly:
            report_weekly = replace(report_weekly, asof=fallback_weekly)
    return report_daily, report_weekly


def _rsi_config(cfg: dict[str, Any]) -> tuple[int | None, str | None]:
    rsi_cfg = (cfg.get("indicators", {}) or {}).get("rsi", {}) or {}
    rsi_enabled = bool(rsi_cfg.get("enabled", False))
    rsi_window = int(rsi_cfg.get("window_default", 14)) if rsi_enabled else None
    rsi_col = f"rsi_{rsi_window}" if rsi_window is not None else None
    return rsi_window, rsi_col


def prepare_series(
    *,
    prepared: PreparedFeatures,
    reports: PreparedReports,
    cfg: dict[str, Any],
) -> PreparedSeries:
    ext_series_daily = prepared.features[prepared.ext_col].dropna()
    open_series_daily = prepared.features["Open"].reindex(ext_series_daily.index)
    close_series_daily = prepared.features["Close"].reindex(ext_series_daily.index)
    high_series_daily = (
        prepared.features["High"].reindex(ext_series_daily.index) if "High" in prepared.features.columns else None
    )
    low_series_daily = (
        prepared.features["Low"].reindex(ext_series_daily.index) if "Low" in prepared.features.columns else None
    )

    rsi_window, rsi_col = _rsi_config(cfg)
    rsi_series_daily = (
        prepared.features[rsi_col].reindex(ext_series_daily.index)
        if rsi_col and rsi_col in prepared.features.columns
        else None
    )
    weekly_open_series = reports.weekly_candles["Open"]
    weekly_close_series = reports.weekly_candles["Close"]
    weekly_high_series = reports.weekly_candles["High"]
    weekly_low_series = reports.weekly_candles["Low"]

    weekly_rsi_series = None
    if rsi_window is not None and not weekly_close_series.empty:
        try:
            from ta.momentum import RSIIndicator

            weekly_rsi_series = RSIIndicator(close=weekly_close_series, window=int(rsi_window)).rsi()
        except Exception:  # noqa: BLE001
            weekly_rsi_series = None

    return PreparedSeries(
        ext_series_daily=ext_series_daily,
        open_series_daily=open_series_daily,
        close_series_daily=close_series_daily,
        high_series_daily=high_series_daily,
        low_series_daily=low_series_daily,
        rsi_window=rsi_window,
        rsi_series_daily=rsi_series_daily,
        weekly_open_series=weekly_open_series,
        weekly_close_series=weekly_close_series,
        weekly_high_series=weekly_high_series,
        weekly_low_series=weekly_low_series,
        weekly_rsi_series=weekly_rsi_series,
    )


def build_rsi_divergence_cfg(
    *,
    series_ctx: PreparedSeries,
    divergence_window_days: int,
    divergence_min_extension_days: int,
    min_ext_pct: float,
    max_ext_pct: float,
    divergence_min_price_delta_pct: float,
    divergence_min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> dict[str, Any] | None:
    if series_ctx.rsi_window is None:
        return None
    return {
        "window_bars": int(divergence_window_days),
        "min_extension_bars": int(divergence_min_extension_days),
        "min_extension_percentile": min_ext_pct,
        "max_extension_percentile": max_ext_pct,
        "min_price_delta_pct": float(divergence_min_price_delta_pct),
        "min_rsi_delta": float(divergence_min_rsi_delta),
        "rsi_overbought": float(rsi_overbought),
        "rsi_oversold": float(rsi_oversold),
        "require_rsi_extreme": bool(require_rsi_extreme),
        "rsi_window": int(series_ctx.rsi_window),
    }


__all__ = [
    "PreparedFeatures",
    "PreparedReports",
    "PreparedSeries",
    "apply_report_asof_fallbacks",
    "build_rsi_divergence_cfg",
    "prepare_features",
    "prepare_reports",
    "prepare_series",
    "resolve_forward_days",
    "resolve_tail_thresholds",
    "resolve_window_years",
]
