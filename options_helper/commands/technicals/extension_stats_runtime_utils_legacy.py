from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def none_if_nan(val: object) -> object | None:
    try:
        if val is None:
            return None
        if isinstance(val, (float, int)) and pd.isna(val):
            return None
        if pd.isna(val):
            return None
        return val
    except Exception:  # noqa: BLE001
        return None


def _latest_divergence_events(
    *,
    flags: pd.DataFrame,
    events_by_date: dict[str, dict[str, Any]],
    divergence_window_days: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    recent = flags.tail(max(1, int(divergence_window_days) + 2))
    last_bearish = None
    last_bullish = None
    for idx, row in reversed(list(recent.iterrows())):
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        if last_bearish is None and bool(row.get("bearish_divergence")):
            last_bearish = events_by_date.get(d)
        if last_bullish is None and bool(row.get("bullish_divergence")):
            last_bullish = events_by_date.get(d)
        if last_bearish is not None and last_bullish is not None:
            break
    return last_bearish, last_bullish


def _divergence_events_by_date(flags: pd.DataFrame) -> dict[str, dict[str, Any]]:
    events = flags[(flags["bearish_divergence"]) | (flags["bullish_divergence"])]
    out: dict[str, dict[str, Any]] = {}
    for idx, row in events.iterrows():
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        out[d] = {
            "date": d,
            "divergence": none_if_nan(row.get("divergence")),
            "rsi_regime": none_if_nan(row.get("rsi_regime")),
            "swing1_date": none_if_nan(row.get("swing1_date")),
            "swing2_date": none_if_nan(row.get("swing2_date")),
            "close1": none_if_nan(row.get("close1")),
            "close2": none_if_nan(row.get("close2")),
            "rsi1": none_if_nan(row.get("rsi1")),
            "rsi2": none_if_nan(row.get("rsi2")),
            "price_delta_pct": none_if_nan(row.get("price_delta_pct")),
            "rsi_delta": none_if_nan(row.get("rsi_delta")),
        }
    return out


def compute_divergence_payload(
    *,
    report: Any,
    ext_series: pd.Series,
    close_series: pd.Series,
    rsi_series: pd.Series | None,
    divergence_window_days: int,
    divergence_min_extension_days: int,
    min_ext_pct: float,
    max_ext_pct: float,
    divergence_min_price_delta_pct: float,
    divergence_min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
    bars_per_year: int,
    rolling_percentile_rank: Callable[[pd.Series, int], pd.Series],
    compute_rsi_divergence_flags: Callable[..., pd.DataFrame],
) -> dict[str, Any] | None:
    try:
        tail_years = report.tail_window_years or (
            max(report.current_percentiles.keys()) if report.current_percentiles else None
        )
        bars = int(tail_years * bars_per_year) if tail_years else None
        if rsi_series is None or not bars or bars <= 1:
            return None

        aligned = pd.concat(
            [ext_series.rename("ext"), close_series.rename("close"), rsi_series.rename("rsi")],
            axis=1,
        ).dropna()
        if aligned.empty:
            return None

        bars = bars if len(aligned) >= bars else len(aligned)
        if bars <= 1:
            return None
        ext_pct = rolling_percentile_rank(aligned["ext"], bars)
        flags = compute_rsi_divergence_flags(
            close_series=aligned["close"],
            rsi_series=aligned["rsi"],
            extension_percentile_series=ext_pct,
            window_days=divergence_window_days,
            min_extension_days=divergence_min_extension_days,
            min_extension_percentile=min_ext_pct,
            max_extension_percentile=max_ext_pct,
            min_price_delta_pct=divergence_min_price_delta_pct,
            min_rsi_delta=divergence_min_rsi_delta,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            require_rsi_extreme=require_rsi_extreme,
        )
        events_by_date = _divergence_events_by_date(flags)
        last_bearish, last_bullish = _latest_divergence_events(
            flags=flags,
            events_by_date=events_by_date,
            divergence_window_days=divergence_window_days,
        )
        return {
            "asof": report.asof,
            "current": {"bearish": last_bearish, "bullish": last_bullish},
            "events_by_date": events_by_date,
        }
    except Exception:  # noqa: BLE001
        return None


__all__ = ["compute_divergence_payload", "none_if_nan"]
