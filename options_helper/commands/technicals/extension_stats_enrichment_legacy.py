from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from options_helper.commands.technicals.extension_stats_enrichment_summary_legacy import (
    add_divergence_tail_event_summary,
    build_max_move_summary_daily,
)
from options_helper.commands.technicals.extension_stats_enrichment_context_legacy import (
    weekly_context_on_daily,
)
from options_helper.commands.technicals.extension_stats_runtime_legacy import ExtensionStatsRuntime
from options_helper.commands.technicals.extension_stats_runtime_utils_legacy import none_if_nan


def _neg0_to_0(val: float) -> float:
    return 0.0 if float(val) == 0.0 else float(val)


def _date_to_iloc(index: pd.Index) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, idx in enumerate(index):
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        out[d] = i
    return out


def _compute_daily_forward_maps(
    *,
    runtime: ExtensionStatsRuntime,
    event_iloc: int | None,
    forward_max_up_move: Callable[..., float | None],
    forward_max_down_move: Callable[..., float | None],
) -> tuple[dict[int, float | None], dict[int, float | None], dict[str, float | None], dict[str, float | None]]:
    max_up_short = {int(h): None for h in runtime.forward_days_daily}
    max_down_short = {int(h): None for h in runtime.forward_days_daily}
    max_up_long = {k: None for k in runtime.max_return_horizons_days.keys()}
    max_down_long = {k: None for k in runtime.max_return_horizons_days.keys()}
    if event_iloc is None:
        return max_up_short, max_down_short, max_up_long, max_down_long

    if runtime.high_series_daily is not None:
        for h in runtime.forward_days_daily:
            r = forward_max_up_move(
                open_series=runtime.open_series_daily,
                high_series=runtime.high_series_daily,
                start_iloc=event_iloc,
                horizon_bars=int(h),
            )
            max_up_short[int(h)] = None if r is None else float(r)
        for label, h in runtime.max_return_horizons_days.items():
            r = forward_max_up_move(
                open_series=runtime.open_series_daily,
                high_series=runtime.high_series_daily,
                start_iloc=event_iloc,
                horizon_bars=int(h),
            )
            max_up_long[str(label)] = None if r is None else float(r)
    if runtime.low_series_daily is not None:
        for h in runtime.forward_days_daily:
            r = forward_max_down_move(
                open_series=runtime.open_series_daily,
                low_series=runtime.low_series_daily,
                start_iloc=event_iloc,
                horizon_bars=int(h),
            )
            max_down_short[int(h)] = None if r is None else float(r)
        for label, h in runtime.max_return_horizons_days.items():
            r = forward_max_down_move(
                open_series=runtime.open_series_daily,
                low_series=runtime.low_series_daily,
                start_iloc=event_iloc,
                horizon_bars=int(h),
            )
            max_down_long[str(label)] = None if r is None else float(r)
    return max_up_short, max_down_short, max_up_long, max_down_long


def _directional_daily_maps(
    *,
    direction: object,
    runtime: ExtensionStatsRuntime,
    max_up_short: dict[int, float | None],
    max_down_short: dict[int, float | None],
    max_up_long: dict[str, float | None],
    max_down_long: dict[str, float | None],
) -> tuple[dict, dict, dict, dict]:
    if direction == "low":
        max_fav_short = {k: none_if_nan(v) for k, v in max_up_short.items()}
        max_fav_long = {k: none_if_nan(v) for k, v in max_up_long.items()}
        dd_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
        dd_long = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_long.items()}
    elif direction == "high":
        max_fav_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
        max_fav_long = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_long.items()}
        dd_short = {k: none_if_nan(v) for k, v in max_up_short.items()}
        dd_long = {k: none_if_nan(v) for k, v in max_up_long.items()}
    else:
        max_fav_short = {int(h): None for h in runtime.forward_days_daily}
        max_fav_long = {k: None for k in runtime.max_return_horizons_days.keys()}
        dd_short = {int(h): None for h in runtime.forward_days_daily}
        dd_long = {k: None for k in runtime.max_return_horizons_days.keys()}
    return max_fav_short, max_fav_long, dd_short, dd_long


def _enrich_daily_tail_events(
    *,
    runtime: ExtensionStatsRuntime,
    daily_tail_events: list[dict[str, Any]],
    daily_date_to_iloc: dict[str, int],
    by_daily_date: dict[str, Any],
    weekly_pct_on_daily: pd.Series | None,
    weekly_rsi_regime_on_daily: pd.Series | None,
    weekly_div_type_on_daily: pd.Series | None,
    weekly_div_rsi_on_daily: pd.Series | None,
    rsi_regime_tag: Callable[..., str],
    forward_max_up_move: Callable[..., float | None],
    forward_max_down_move: Callable[..., float | None],
) -> None:
    for ev in daily_tail_events:
        d = ev.get("date")
        ev["rsi_divergence"] = by_daily_date.get(d) if isinstance(by_daily_date, dict) else None
        i = daily_date_to_iloc.get(d) if isinstance(d, str) else None
        rsi_val = none_if_nan(runtime.rsi_series_daily.iloc[i]) if i is not None and runtime.rsi_series_daily is not None else None
        ev["rsi"] = rsi_val
        ev["rsi_regime"] = _event_rsi_regime(rsi_val=rsi_val, runtime=runtime, rsi_regime_tag=rsi_regime_tag)
        max_up_short, max_down_short, max_up_long, max_down_long = _compute_daily_forward_maps(
            runtime=runtime,
            event_iloc=i,
            forward_max_up_move=forward_max_up_move,
            forward_max_down_move=forward_max_down_move,
        )
        max_fav_short, max_fav_long, dd_short, dd_long = _directional_daily_maps(
            direction=ev.get("direction"),
            runtime=runtime,
            max_up_short=max_up_short,
            max_down_short=max_down_short,
            max_up_long=max_up_long,
            max_down_long=max_down_long,
        )
        ev["forward_max_up_returns"] = max_up_short
        ev["forward_max_down_returns"] = max_down_short
        ev["forward_max_fav_returns"] = max_fav_short
        ev["max_up_returns"] = max_up_long
        ev["max_down_returns"] = max_down_long
        ev["max_fav_returns"] = max_fav_long
        ev["forward_drawdown_returns"] = dd_short
        ev["drawdown_returns"] = dd_long
        ev["weekly_context"] = _daily_weekly_context(
            event_iloc=i,
            weekly_pct_on_daily=weekly_pct_on_daily,
            weekly_rsi_regime_on_daily=weekly_rsi_regime_on_daily,
            weekly_div_type_on_daily=weekly_div_type_on_daily,
            weekly_div_rsi_on_daily=weekly_div_rsi_on_daily,
        )


def _event_rsi_regime(
    *,
    rsi_val: object,
    runtime: ExtensionStatsRuntime,
    rsi_regime_tag: Callable[..., str],
) -> str | None:
    if rsi_val is None:
        return None
    try:
        return rsi_regime_tag(
            rsi_value=float(rsi_val),
            rsi_overbought=float(runtime.rsi_overbought),
            rsi_oversold=float(runtime.rsi_oversold),
        )
    except Exception:  # noqa: BLE001
        return None


def _daily_weekly_context(
    *,
    event_iloc: int | None,
    weekly_pct_on_daily: pd.Series | None,
    weekly_rsi_regime_on_daily: pd.Series | None,
    weekly_div_type_on_daily: pd.Series | None,
    weekly_div_rsi_on_daily: pd.Series | None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "extension_percentile": None,
        "rsi_regime": None,
        "divergence": None,
        "divergence_rsi_regime": None,
    }
    if event_iloc is None:
        return out
    if weekly_pct_on_daily is not None:
        out["extension_percentile"] = none_if_nan(weekly_pct_on_daily.iloc[event_iloc])
    if weekly_rsi_regime_on_daily is not None:
        out["rsi_regime"] = none_if_nan(weekly_rsi_regime_on_daily.iloc[event_iloc])
    if weekly_div_type_on_daily is not None:
        out["divergence"] = none_if_nan(weekly_div_type_on_daily.iloc[event_iloc])
    if weekly_div_rsi_on_daily is not None:
        out["divergence_rsi_regime"] = none_if_nan(weekly_div_rsi_on_daily.iloc[event_iloc])
    return out


def _enrich_weekly_tail_events(
    *,
    runtime: ExtensionStatsRuntime,
    weekly_tail_events: list[dict[str, Any]],
    weekly_date_to_iloc: dict[str, int],
    by_weekly_date: dict[str, Any],
    rsi_regime_tag: Callable[..., str],
    forward_max_up_move: Callable[..., float | None],
    forward_max_down_move: Callable[..., float | None],
) -> None:
    for ev in weekly_tail_events:
        d = ev.get("date")
        ev["rsi_divergence"] = by_weekly_date.get(d) if isinstance(by_weekly_date, dict) else None
        i = weekly_date_to_iloc.get(d) if isinstance(d, str) else None
        rsi_val = none_if_nan(runtime.weekly_rsi_series.iloc[i]) if i is not None and runtime.weekly_rsi_series is not None else None
        ev["rsi"] = rsi_val
        ev["rsi_regime"] = _event_rsi_regime(rsi_val=rsi_val, runtime=runtime, rsi_regime_tag=rsi_regime_tag)
        max_up_short, max_down_short = _compute_weekly_forward_maps(
            runtime=runtime,
            event_iloc=i,
            forward_max_up_move=forward_max_up_move,
            forward_max_down_move=forward_max_down_move,
        )
        max_fav_short, dd_short = _directional_weekly_maps(
            direction=ev.get("direction"),
            runtime=runtime,
            max_up_short=max_up_short,
            max_down_short=max_down_short,
        )
        ev["forward_max_up_returns"] = max_up_short
        ev["forward_max_down_returns"] = max_down_short
        ev["forward_max_fav_returns"] = max_fav_short
        ev["forward_drawdown_returns"] = dd_short


def _compute_weekly_forward_maps(
    *,
    runtime: ExtensionStatsRuntime,
    event_iloc: int | None,
    forward_max_up_move: Callable[..., float | None],
    forward_max_down_move: Callable[..., float | None],
) -> tuple[dict[int, float | None], dict[int, float | None]]:
    max_up_short = {int(h): None for h in runtime.forward_days_weekly}
    max_down_short = {int(h): None for h in runtime.forward_days_weekly}
    if event_iloc is None:
        return max_up_short, max_down_short
    for h in runtime.forward_days_weekly:
        r_up = forward_max_up_move(
            open_series=runtime.weekly_open_series,
            high_series=runtime.weekly_high_series,
            start_iloc=event_iloc,
            horizon_bars=int(h),
        )
        r_dn = forward_max_down_move(
            open_series=runtime.weekly_open_series,
            low_series=runtime.weekly_low_series,
            start_iloc=event_iloc,
            horizon_bars=int(h),
        )
        max_up_short[int(h)] = None if r_up is None else float(r_up)
        max_down_short[int(h)] = None if r_dn is None else float(r_dn)
    return max_up_short, max_down_short


def _directional_weekly_maps(
    *,
    direction: object,
    runtime: ExtensionStatsRuntime,
    max_up_short: dict[int, float | None],
    max_down_short: dict[int, float | None],
) -> tuple[dict[int, float | None], dict[int, float | None]]:
    if direction == "low":
        max_fav_short = {k: none_if_nan(v) for k, v in max_up_short.items()}
        dd_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
    elif direction == "high":
        max_fav_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
        dd_short = {k: none_if_nan(v) for k, v in max_up_short.items()}
    else:
        max_fav_short = {int(h): None for h in runtime.forward_days_weekly}
        dd_short = {int(h): None for h in runtime.forward_days_weekly}
    return max_fav_short, dd_short


def enrich_extension_payload(runtime: ExtensionStatsRuntime) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from options_helper.technicals_backtesting.extension_percentiles import rolling_percentile_rank
    from options_helper.technicals_backtesting.max_forward_returns import (
        forward_max_down_move,
        forward_max_up_move,
    )
    from options_helper.technicals_backtesting.rsi_divergence import rsi_regime_tag

    payload = runtime.payload
    daily = payload.get("report_daily", {}) or {}
    weekly = payload.get("report_weekly", {}) or {}
    daily_tail_events = daily.get("tail_events", []) or []
    weekly_tail_events = weekly.get("tail_events", []) or []
    daily_date_to_iloc = _date_to_iloc(runtime.ext_series_daily.index)
    weekly_pct_on_daily, weekly_rsi_regime_on_daily, weekly_div_type_on_daily, weekly_div_rsi_on_daily = weekly_context_on_daily(
        runtime=runtime,
        rolling_percentile_rank=rolling_percentile_rank,
        rsi_regime_tag=rsi_regime_tag,
    )
    by_daily_date = (
        (runtime.rsi_divergence_daily or {}).get("events_by_date", {})
        if isinstance(runtime.rsi_divergence_daily, dict)
        else {}
    )
    _enrich_daily_tail_events(
        runtime=runtime,
        daily_tail_events=daily_tail_events,
        daily_date_to_iloc=daily_date_to_iloc,
        by_daily_date=by_daily_date,
        weekly_pct_on_daily=weekly_pct_on_daily,
        weekly_rsi_regime_on_daily=weekly_rsi_regime_on_daily,
        weekly_div_type_on_daily=weekly_div_type_on_daily,
        weekly_div_rsi_on_daily=weekly_div_rsi_on_daily,
        rsi_regime_tag=rsi_regime_tag,
        forward_max_up_move=forward_max_up_move,
        forward_max_down_move=forward_max_down_move,
    )
    weekly_date_to_iloc = _date_to_iloc(runtime.weekly_close_series.index)
    by_weekly_date = (
        (runtime.rsi_divergence_weekly or {}).get("events_by_date", {})
        if isinstance(runtime.rsi_divergence_weekly, dict)
        else {}
    )
    _enrich_weekly_tail_events(
        runtime=runtime,
        weekly_tail_events=weekly_tail_events,
        weekly_date_to_iloc=weekly_date_to_iloc,
        by_weekly_date=by_weekly_date,
        rsi_regime_tag=rsi_regime_tag,
        forward_max_up_move=forward_max_up_move,
        forward_max_down_move=forward_max_down_move,
    )
    max_move_summary_daily = {"horizons_days": runtime.max_return_horizons_days, "buckets": []}
    try:
        max_move_summary_daily = build_max_move_summary_daily(
            daily_tail_events=daily_tail_events,
            max_return_horizons_days=runtime.max_return_horizons_days,
        )
    except Exception:  # noqa: BLE001
        max_move_summary_daily = {"horizons_days": runtime.max_return_horizons_days, "buckets": []}
    payload["max_move_summary_daily"] = max_move_summary_daily
    payload["max_upside_summary_daily"] = max_move_summary_daily
    add_divergence_tail_event_summary(runtime=runtime, daily_tail_events=daily_tail_events)
    daily_tail_events = sorted(daily_tail_events, key=lambda ev: (ev.get("date") or ""), reverse=True)
    weekly_tail_events = sorted(weekly_tail_events, key=lambda ev: (ev.get("date") or ""), reverse=True)
    daily["tail_events"] = daily_tail_events
    weekly["tail_events"] = weekly_tail_events
    payload["report_daily"] = daily
    payload["report_weekly"] = weekly
    return payload, daily_tail_events, weekly_tail_events


__all__ = ["enrich_extension_payload"]
