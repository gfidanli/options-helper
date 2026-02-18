from __future__ import annotations

from typing import Callable

import pandas as pd

from options_helper.commands.technicals.extension_stats_runtime_legacy import ExtensionStatsRuntime


def weekly_context_on_daily(
    *,
    runtime: ExtensionStatsRuntime,
    rolling_percentile_rank: Callable[[pd.Series, int], pd.Series],
    rsi_regime_tag: Callable[..., str],
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None]:
    weekly_ext_pct = None
    try:
        tail_years_w = runtime.report_weekly.tail_window_years or (
            max(runtime.report_weekly.current_percentiles.keys())
            if runtime.report_weekly.current_percentiles
            else None
        )
        bars_w = int(tail_years_w * int(runtime.ext_cfg.get("days_per_year", 252) / 5)) if tail_years_w else None
        ext_w = runtime.weekly_ext.dropna()
        if bars_w and bars_w > 1 and not ext_w.empty:
            bars_w = bars_w if len(ext_w) >= bars_w else len(ext_w)
            if bars_w > 1:
                weekly_ext_pct = rolling_percentile_rank(ext_w, bars_w)
    except Exception:  # noqa: BLE001
        weekly_ext_pct = None

    weekly_pct_on_daily = (
        weekly_ext_pct.reindex(runtime.ext_series_daily.index, method="ffill")
        if weekly_ext_pct is not None
        else None
    )
    weekly_rsi_regime_on_daily = None
    if runtime.weekly_rsi_series is not None:
        try:
            weekly_rsi_regime = runtime.weekly_rsi_series.dropna().apply(
                lambda v: rsi_regime_tag(
                    rsi_value=float(v),
                    rsi_overbought=float(runtime.rsi_overbought),
                    rsi_oversold=float(runtime.rsi_oversold),
                )
            )
            weekly_rsi_regime_on_daily = weekly_rsi_regime.reindex(runtime.ext_series_daily.index, method="ffill")
        except Exception:  # noqa: BLE001
            weekly_rsi_regime_on_daily = None

    weekly_div_type_on_daily, weekly_div_rsi_on_daily = _weekly_divergence_on_daily(runtime=runtime)
    return weekly_pct_on_daily, weekly_rsi_regime_on_daily, weekly_div_type_on_daily, weekly_div_rsi_on_daily


def _weekly_divergence_on_daily(
    *,
    runtime: ExtensionStatsRuntime,
) -> tuple[pd.Series | None, pd.Series | None]:
    by_weekly_date = (
        (runtime.rsi_divergence_weekly or {}).get("events_by_date", {})
        if isinstance(runtime.rsi_divergence_weekly, dict)
        else {}
    )
    if not isinstance(by_weekly_date, dict) or not by_weekly_date:
        return None, None

    try:
        s_div = pd.Series(index=runtime.weekly_close_series.index, dtype="object")
        s_tag = pd.Series(index=runtime.weekly_close_series.index, dtype="object")
        for d, ev in by_weekly_date.items():
            try:
                ts = pd.Timestamp(d)
            except Exception:  # noqa: BLE001
                continue
            if ts in s_div.index:
                s_div.loc[ts] = (ev or {}).get("divergence")
                s_tag.loc[ts] = (ev or {}).get("rsi_regime")
        div_daily = s_div.reindex(runtime.ext_series_daily.index, method="ffill")
        rsi_daily = s_tag.reindex(runtime.ext_series_daily.index, method="ffill")
        return div_daily, rsi_daily
    except Exception:  # noqa: BLE001
        return None, None


__all__ = ["weekly_context_on_daily"]
