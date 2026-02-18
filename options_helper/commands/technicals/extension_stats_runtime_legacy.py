from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from options_helper.commands.technicals.extension_stats_runtime_parts_legacy import (
    apply_report_asof_fallbacks,
    build_rsi_divergence_cfg,
    prepare_features,
    prepare_reports,
    prepare_series,
    resolve_forward_days,
    resolve_tail_thresholds,
    resolve_window_years,
)
from options_helper.commands.technicals.extension_stats_runtime_utils_legacy import (
    compute_divergence_payload,
)


@dataclass(frozen=True)
class ExtensionStatsRuntime:
    sym_label: str
    report_daily: Any
    report_weekly: Any
    payload: dict[str, Any]
    ext_cfg: dict[str, Any]
    ext_series_daily: pd.Series
    open_series_daily: pd.Series
    close_series_daily: pd.Series
    high_series_daily: pd.Series | None
    low_series_daily: pd.Series | None
    rsi_series_daily: pd.Series | None
    weekly_ext: pd.Series
    weekly_open_series: pd.Series
    weekly_close_series: pd.Series
    weekly_high_series: pd.Series
    weekly_low_series: pd.Series
    weekly_rsi_series: pd.Series | None
    forward_days_daily: list[int]
    forward_days_weekly: list[int]
    max_return_horizons_days: dict[str, int]
    rsi_overbought: float
    rsi_oversold: float
    rsi_divergence_cfg: dict[str, Any] | None
    rsi_divergence_daily: dict[str, Any] | None
    rsi_divergence_weekly: dict[str, Any] | None
    warmup_warning: bool


@dataclass(frozen=True)
class _PreparedRuntimeInputs:
    prepared: Any
    ext_cfg: dict[str, Any]
    tail_low_pct: float
    tail_high_pct: float
    windows_years: list[int]
    forward_days_daily: list[int]
    forward_days_weekly: list[int]
    reports: Any
    report_daily: Any
    report_weekly: Any
    series_ctx: Any


def _divergence_thresholds(
    *,
    tail_low_pct: float,
    tail_high_pct: float,
    divergence_min_extension_percentile: float | None,
    divergence_max_extension_percentile: float | None,
) -> tuple[float, float]:
    min_ext_pct = (
        float(divergence_min_extension_percentile)
        if divergence_min_extension_percentile is not None
        else float(tail_high_pct)
    )
    max_ext_pct = (
        float(divergence_max_extension_percentile)
        if divergence_max_extension_percentile is not None
        else float(tail_low_pct)
    )
    return min_ext_pct, max_ext_pct


def _compute_divergence_outputs(
    *,
    ext_cfg: dict[str, Any],
    report_daily: Any,
    report_weekly: Any,
    series_ctx: Any,
    weekly_ext: pd.Series,
    divergence_window_days: int,
    divergence_min_extension_days: int,
    min_ext_pct: float,
    max_ext_pct: float,
    divergence_min_price_delta_pct: float,
    divergence_min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    from options_helper.technicals_backtesting.extension_percentiles import rolling_percentile_rank
    from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags

    daily = compute_divergence_payload(
        report=report_daily,
        ext_series=series_ctx.ext_series_daily,
        close_series=series_ctx.close_series_daily,
        rsi_series=series_ctx.rsi_series_daily,
        divergence_window_days=divergence_window_days,
        divergence_min_extension_days=divergence_min_extension_days,
        min_ext_pct=min_ext_pct,
        max_ext_pct=max_ext_pct,
        divergence_min_price_delta_pct=divergence_min_price_delta_pct,
        divergence_min_rsi_delta=divergence_min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
        bars_per_year=int(ext_cfg.get("days_per_year", 252)),
        rolling_percentile_rank=rolling_percentile_rank,
        compute_rsi_divergence_flags=compute_rsi_divergence_flags,
    )
    weekly_ext_non_null = weekly_ext.dropna()
    weekly = compute_divergence_payload(
        report=report_weekly,
        ext_series=weekly_ext_non_null,
        close_series=series_ctx.weekly_close_series.reindex(weekly_ext_non_null.index),
        rsi_series=(
            series_ctx.weekly_rsi_series.reindex(weekly_ext_non_null.index)
            if series_ctx.weekly_rsi_series is not None
            else None
        ),
        divergence_window_days=divergence_window_days,
        divergence_min_extension_days=divergence_min_extension_days,
        min_ext_pct=min_ext_pct,
        max_ext_pct=max_ext_pct,
        divergence_min_price_delta_pct=divergence_min_price_delta_pct,
        divergence_min_rsi_delta=divergence_min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
        bars_per_year=int(ext_cfg.get("days_per_year", 252) / 5),
        rolling_percentile_rank=rolling_percentile_rank,
        compute_rsi_divergence_flags=compute_rsi_divergence_flags,
    )
    return daily, weekly


def _effective_extension_config(
    *,
    ext_cfg: dict[str, Any],
    windows_years: list[int],
    tail_high_pct: float,
    tail_low_pct: float,
    forward_days_daily: list[int],
    forward_days_weekly: list[int],
) -> dict[str, Any]:
    out = dict(ext_cfg)
    out["windows_years"] = windows_years
    out["tail_high_pct"] = float(tail_high_pct)
    out["tail_low_pct"] = float(tail_low_pct)
    out["forward_days_daily"] = forward_days_daily
    out["forward_days_weekly"] = forward_days_weekly
    return out


def _build_payload_base(
    *,
    sym_label: str,
    report_daily: Any,
    report_weekly: Any,
    ext_cfg_effective: dict[str, Any],
    prepared: Any,
    rsi_divergence_cfg: dict[str, Any] | None,
    max_return_horizons_days: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "symbol": sym_label,
        "asof": report_daily.asof,
        "config": {
            "extension_percentiles": ext_cfg_effective,
            "atr_window": prepared.atr_window,
            "sma_window": prepared.sma_window,
            "extension_column": prepared.ext_col,
            "rsi_divergence": rsi_divergence_cfg,
            "max_forward_returns": {
                "method": "directional_mfe",
                "entry_anchor": "next_bar_open",
                "horizons_days": max_return_horizons_days,
            },
        },
        "report_daily": asdict(report_daily),
        "report_weekly": asdict(report_weekly),
    }


def _prepare_runtime_inputs(
    *,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    tail_pct: float | None,
    percentile_window_years: int | None,
) -> _PreparedRuntimeInputs:
    prepared = prepare_features(df=df, cfg=cfg)
    ext_cfg = cfg.get("extension_percentiles", {}) or {}
    days_per_year = int(ext_cfg.get("days_per_year", 252))
    tail_low_pct, tail_high_pct = resolve_tail_thresholds(ext_cfg=ext_cfg, tail_pct=tail_pct)
    available_bars = int(prepared.features[prepared.ext_col].dropna().shape[0])
    window_years = resolve_window_years(
        available_bars=available_bars,
        days_per_year=days_per_year,
        percentile_window_years=percentile_window_years,
    )
    windows_years = [window_years]
    forward_days_daily, forward_days_weekly = resolve_forward_days(ext_cfg)
    reports = prepare_reports(
        df=df,
        prepared=prepared,
        windows_years=windows_years,
        days_per_year=days_per_year,
        tail_high_pct=float(tail_high_pct),
        tail_low_pct=float(tail_low_pct),
        forward_days_daily=forward_days_daily,
        forward_days_weekly=forward_days_weekly,
        weekly_rule=cfg["weekly_regime"].get("resample_rule", "W-FRI"),
    )
    report_daily, report_weekly = apply_report_asof_fallbacks(reports=reports, df=df)
    series_ctx = prepare_series(prepared=prepared, reports=reports, cfg=cfg)
    return _PreparedRuntimeInputs(
        prepared=prepared,
        ext_cfg=ext_cfg,
        tail_low_pct=tail_low_pct,
        tail_high_pct=tail_high_pct,
        windows_years=windows_years,
        forward_days_daily=forward_days_daily,
        forward_days_weekly=forward_days_weekly,
        reports=reports,
        report_daily=report_daily,
        report_weekly=report_weekly,
        series_ctx=series_ctx,
    )


def _build_runtime_result(
    *,
    inputs: _PreparedRuntimeInputs,
    symbol: str | None,
    rsi_overbought: float,
    rsi_oversold: float,
    rsi_divergence_cfg: dict[str, Any] | None,
    rsi_divergence_daily: dict[str, Any] | None,
    rsi_divergence_weekly: dict[str, Any] | None,
) -> ExtensionStatsRuntime:
    ext_cfg_effective = _effective_extension_config(
        ext_cfg=inputs.ext_cfg,
        windows_years=inputs.windows_years,
        tail_high_pct=inputs.tail_high_pct,
        tail_low_pct=inputs.tail_low_pct,
        forward_days_daily=inputs.forward_days_daily,
        forward_days_weekly=inputs.forward_days_weekly,
    )
    max_return_horizons_days = {"1w": 5, "4w": 20, "3m": 63, "6m": 126, "9m": 189, "1y": 252}
    sym_label = symbol.upper() if symbol else "UNKNOWN"
    payload = _build_payload_base(
        sym_label=sym_label,
        report_daily=inputs.report_daily,
        report_weekly=inputs.report_weekly,
        ext_cfg_effective=ext_cfg_effective,
        prepared=inputs.prepared,
        rsi_divergence_cfg=rsi_divergence_cfg,
        max_return_horizons_days=max_return_horizons_days,
    )
    payload["rsi_divergence_daily"] = rsi_divergence_daily
    payload["rsi_divergence_weekly"] = rsi_divergence_weekly
    return ExtensionStatsRuntime(
        sym_label=sym_label,
        report_daily=inputs.report_daily,
        report_weekly=inputs.report_weekly,
        payload=payload,
        ext_cfg=inputs.ext_cfg,
        ext_series_daily=inputs.series_ctx.ext_series_daily,
        open_series_daily=inputs.series_ctx.open_series_daily,
        close_series_daily=inputs.series_ctx.close_series_daily,
        high_series_daily=inputs.series_ctx.high_series_daily,
        low_series_daily=inputs.series_ctx.low_series_daily,
        rsi_series_daily=inputs.series_ctx.rsi_series_daily,
        weekly_ext=inputs.reports.weekly_ext,
        weekly_open_series=inputs.series_ctx.weekly_open_series,
        weekly_close_series=inputs.series_ctx.weekly_close_series,
        weekly_high_series=inputs.series_ctx.weekly_high_series,
        weekly_low_series=inputs.series_ctx.weekly_low_series,
        weekly_rsi_series=inputs.series_ctx.weekly_rsi_series,
        forward_days_daily=inputs.forward_days_daily,
        forward_days_weekly=inputs.forward_days_weekly,
        max_return_horizons_days=max_return_horizons_days,
        rsi_overbought=float(rsi_overbought),
        rsi_oversold=float(rsi_oversold),
        rsi_divergence_cfg=rsi_divergence_cfg,
        rsi_divergence_daily=rsi_divergence_daily,
        rsi_divergence_weekly=rsi_divergence_weekly,
        warmup_warning=inputs.prepared.warmup_warning,
    )


def build_extension_stats_runtime(
    *,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    symbol: str | None,
    tail_pct: float | None,
    percentile_window_years: int | None,
    divergence_window_days: int,
    divergence_min_extension_days: int,
    divergence_min_extension_percentile: float | None,
    divergence_max_extension_percentile: float | None,
    divergence_min_price_delta_pct: float,
    divergence_min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> ExtensionStatsRuntime:
    inputs = _prepare_runtime_inputs(
        df=df,
        cfg=cfg,
        tail_pct=tail_pct,
        percentile_window_years=percentile_window_years,
    )
    min_ext_pct, max_ext_pct = _divergence_thresholds(
        tail_low_pct=inputs.tail_low_pct,
        tail_high_pct=inputs.tail_high_pct,
        divergence_min_extension_percentile=divergence_min_extension_percentile,
        divergence_max_extension_percentile=divergence_max_extension_percentile,
    )
    rsi_divergence_cfg = build_rsi_divergence_cfg(
        series_ctx=inputs.series_ctx,
        divergence_window_days=divergence_window_days,
        divergence_min_extension_days=divergence_min_extension_days,
        min_ext_pct=min_ext_pct,
        max_ext_pct=max_ext_pct,
        divergence_min_price_delta_pct=divergence_min_price_delta_pct,
        divergence_min_rsi_delta=divergence_min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )
    rsi_divergence_daily, rsi_divergence_weekly = _compute_divergence_outputs(
        ext_cfg=inputs.ext_cfg,
        report_daily=inputs.report_daily,
        report_weekly=inputs.report_weekly,
        series_ctx=inputs.series_ctx,
        weekly_ext=inputs.reports.weekly_ext,
        divergence_window_days=divergence_window_days,
        divergence_min_extension_days=divergence_min_extension_days,
        min_ext_pct=min_ext_pct,
        max_ext_pct=max_ext_pct,
        divergence_min_price_delta_pct=divergence_min_price_delta_pct,
        divergence_min_rsi_delta=divergence_min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )
    return _build_runtime_result(
        inputs=inputs,
        symbol=symbol,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        rsi_divergence_cfg=rsi_divergence_cfg,
        rsi_divergence_daily=rsi_divergence_daily,
        rsi_divergence_weekly=rsi_divergence_weekly,
    )


__all__ = ["ExtensionStatsRuntime", "build_extension_stats_runtime"]
