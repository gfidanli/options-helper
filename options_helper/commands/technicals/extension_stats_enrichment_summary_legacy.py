from __future__ import annotations

from typing import Any

import pandas as pd

from options_helper.commands.technicals.extension_stats_runtime_legacy import ExtensionStatsRuntime


def _quantile(values: list[float], q: float) -> float | None:
    import numpy as np

    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.percentile(vals, q * 100.0))


def _max_move_stats_for_horizon(*, rows: list[dict[str, Any]], key: str, horizon_label: str) -> dict[str, float | None]:
    is_high_tail_bucket = str(key).startswith("high_tail")
    fav_sign = -1.0 if is_high_tail_bucket else 1.0
    dd_sign = 1.0 if is_high_tail_bucket else -1.0
    fav_mags = []
    dd_mags = []
    for ev in rows:
        v = (ev.get("max_fav_returns") or {}).get(horizon_label)
        if v is not None and not pd.isna(v):
            fav_mags.append(abs(float(v)) * 100.0)
        dd = (ev.get("drawdown_returns") or {}).get(horizon_label)
        if dd is not None and not pd.isna(dd):
            dd_mags.append(abs(float(dd)) * 100.0)
    fav_med = _quantile(fav_mags, 0.50)
    fav_p75 = _quantile(fav_mags, 0.75)
    dd_med = _quantile(dd_mags, 0.50)
    dd_p75 = _quantile(dd_mags, 0.75)
    return {
        "fav_median": None if fav_med is None else float(fav_sign) * float(fav_med),
        "fav_p75": None if fav_p75 is None else float(fav_sign) * float(fav_p75),
        "dd_median": None if dd_med is None else float(dd_sign) * float(dd_med),
        "dd_p75": None if dd_p75 is None else float(dd_sign) * float(dd_p75),
    }


def build_max_move_summary_daily(
    *,
    daily_tail_events: list[dict[str, Any]],
    max_return_horizons_days: dict[str, int],
) -> dict[str, object]:
    out: dict[str, object] = {"horizons_days": max_return_horizons_days, "buckets": []}
    buckets = [
        ("low_tail_all", "Low tail (all)", lambda ev: ev.get("direction") == "low"),
        ("low_tail_rsi_oversold", "Low tail + RSI oversold (event)", lambda ev: ev.get("direction") == "low" and ev.get("rsi_regime") == "oversold"),
        ("low_tail_bull_div", "Low tail + bullish divergence", lambda ev: ev.get("direction") == "low" and isinstance(ev.get("rsi_divergence"), dict) and (ev.get("rsi_divergence") or {}).get("divergence") == "bullish"),
        ("low_tail_bull_div_rsi_oversold", "Low tail + bullish divergence + RSI oversold (event)", lambda ev: ev.get("direction") == "low" and ev.get("rsi_regime") == "oversold" and isinstance(ev.get("rsi_divergence"), dict) and (ev.get("rsi_divergence") or {}).get("divergence") == "bullish"),
        ("high_tail_all", "High tail (all)", lambda ev: ev.get("direction") == "high"),
        ("high_tail_rsi_overbought", "High tail + RSI overbought (event)", lambda ev: ev.get("direction") == "high" and ev.get("rsi_regime") == "overbought"),
        ("high_tail_bear_div", "High tail + bearish divergence", lambda ev: ev.get("direction") == "high" and isinstance(ev.get("rsi_divergence"), dict) and (ev.get("rsi_divergence") or {}).get("divergence") == "bearish"),
        ("high_tail_bear_div_rsi_overbought", "High tail + bearish divergence + RSI overbought (event)", lambda ev: ev.get("direction") == "high" and ev.get("rsi_regime") == "overbought" and isinstance(ev.get("rsi_divergence"), dict) and (ev.get("rsi_divergence") or {}).get("divergence") == "bearish"),
    ]
    for key, label, fn in buckets:
        rows = [ev for ev in daily_tail_events if fn(ev)]
        bucket_out: dict[str, object] = {"key": key, "label": label, "n": len(rows), "stats": {}}
        for h_label in max_return_horizons_days.keys():
            bucket_out["stats"][h_label] = _max_move_stats_for_horizon(rows=rows, key=key, horizon_label=h_label)
        out["buckets"].append(bucket_out)
    return out


def add_divergence_tail_event_summary(*, runtime: ExtensionStatsRuntime, daily_tail_events: list[dict[str, Any]]) -> None:
    if runtime.rsi_divergence_daily is None:
        return
    fwd_days_int = [int(d) for d in runtime.forward_days_daily]
    summary: dict[str, dict[str, dict]] = {"high": {}, "low": {}}
    for tail in ("high", "low"):
        want = "bearish" if tail == "high" else "bullish"
        summary[tail]["with_divergence"] = _divergence_bucket(
            daily_tail_events=daily_tail_events,
            tail=tail,
            want=want,
            with_divergence=True,
            fwd_days_int=fwd_days_int,
        )
        summary[tail]["without_divergence"] = _divergence_bucket(
            daily_tail_events=daily_tail_events,
            tail=tail,
            want=want,
            with_divergence=False,
            fwd_days_int=fwd_days_int,
        )
    runtime.rsi_divergence_daily["tail_event_summary"] = summary


def _divergence_bucket(
    *,
    daily_tail_events: list[dict[str, Any]],
    tail: str,
    want: str,
    with_divergence: bool,
    fwd_days_int: list[int],
) -> dict[str, object]:
    rows = []
    for ev in daily_tail_events:
        if ev.get("direction") != tail:
            continue
        div = ev.get("rsi_divergence") or None
        match = bool(div) and div.get("divergence") == want
        if match != with_divergence:
            continue
        rows.append(ev)
    out: dict[str, object] = {"n": len(rows)}
    for d in fwd_days_int:
        maxrets = []
        pcts = []
        for ev in rows:
            r = _get_forward(ev.get("forward_max_fav_returns") or {}, d)
            p = _get_forward(ev.get("forward_extension_percentiles") or {}, d)
            if r is not None and not pd.isna(r):
                maxrets.append(float(r) * 100.0)
            if p is not None and not pd.isna(p):
                pcts.append(float(p))
        out[f"median_fwd_max_fav_move_pct_{d}d"] = _median(maxrets)
        out[f"median_fwd_extension_percentile_{d}d"] = _median(pcts)
    return out


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    m = len(vals) // 2
    if len(vals) % 2:
        return float(vals[m])
    return float((vals[m - 1] + vals[m]) / 2.0)


def _get_forward(dct: dict, day: int) -> object | None:
    if day in dct:
        return dct.get(day)
    return dct.get(str(day))


__all__ = ["add_divergence_tail_event_summary", "build_max_move_summary_daily"]
