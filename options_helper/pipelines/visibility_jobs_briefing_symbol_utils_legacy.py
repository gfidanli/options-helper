from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.confluence import ConfluenceInputs, score_confluence
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.flow import aggregate_flow_window, compute_flow
from options_helper.commands.common import _spot_from_meta
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot

if TYPE_CHECKING:
    from options_helper.models import Position


def _trend_from_weekly_flag(flag: bool | None) -> str | None:
    if flag is True:
        return "up"
    if flag is False:
        return "down"
    return None


def _extension_percentile_from_snapshot(snapshot: TechnicalSnapshot | None) -> float | None:
    if snapshot is None or snapshot.extension_percentiles is None:
        return None

    daily = snapshot.extension_percentiles.daily
    if daily is None or not daily.current_percentiles:
        return None

    parsed: list[tuple[float, float]] = []
    for key, value in daily.current_percentiles.items():
        try:
            parsed.append((float(key), float(value)))
        except Exception:  # noqa: BLE001
            continue
    if not parsed:
        return None
    return sorted(parsed, key=lambda item: item[0])[-1][1]


def _net_flow_delta_oi_notional(flow_net: pd.DataFrame | None) -> float | None:
    if flow_net is None or flow_net.empty:
        return None
    if "deltaOI_notional" not in flow_net.columns or "optionType" not in flow_net.columns:
        return None

    local = flow_net.copy()
    local["deltaOI_notional"] = pd.to_numeric(local["deltaOI_notional"], errors="coerce")
    local["optionType"] = local["optionType"].astype(str).str.lower()
    calls = local[local["optionType"] == "call"]["deltaOI_notional"].dropna()
    puts = local[local["optionType"] == "put"]["deltaOI_notional"].dropna()
    if calls.empty and puts.empty:
        return None
    return float(calls.sum()) - float(puts.sum())


def _collect_event_warnings(
    *,
    today: date,
    next_earnings_date: date | None,
    risk_profile: Any,
    positions_for_symbol: list[Position],
) -> list[str]:
    event_warnings: set[str] = set()
    base_risk = earnings_event_risk(
        today=today,
        expiry=None,
        next_earnings_date=next_earnings_date,
        warn_days=risk_profile.earnings_warn_days,
        avoid_days=risk_profile.earnings_avoid_days,
    )
    event_warnings.update(base_risk["warnings"])

    for position in positions_for_symbol:
        position_risk = earnings_event_risk(
            today=today,
            expiry=position.expiry,
            next_earnings_date=next_earnings_date,
            warn_days=risk_profile.earnings_warn_days,
            avoid_days=risk_profile.earnings_avoid_days,
        )
        event_warnings.update(position_risk["warnings"])
    return sorted(event_warnings)


def _load_compare_and_flow(
    *,
    compare_enabled: bool,
    compare_norm: str,
    store: Any,
    sym: str,
    to_date: date,
    df_to: pd.DataFrame,
    spot_to: float,
    top: int,
) -> tuple[Any | None, pd.DataFrame | None, list[str]]:
    warnings: list[str] = []
    if not compare_enabled:
        return None, None, warnings

    if compare_norm.startswith("-") and compare_norm[1:].isdigit():
        try:
            from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"compare unavailable: {exc}")
            return None, None, warnings
    else:
        from_date = store.resolve_date(sym, compare_norm)

    if from_date is None or from_date == to_date:
        return None, None, warnings

    df_from = store.load_day(sym, from_date)
    spot_from = _spot_from_meta(store.load_meta(sym, from_date))
    if spot_from is None:
        warnings.append("compare unavailable: missing spot in from-date meta.json")
        return None, None, warnings
    if df_from.empty or df_to.empty:
        warnings.append("compare unavailable: missing snapshot CSVs for from/to date")
        return None, None, warnings

    compare_report, _, _ = compute_compare_report(
        symbol=sym,
        from_date=from_date,
        to_date=to_date,
        from_df=df_from,
        to_df=df_to,
        spot_from=spot_from,
        spot_to=spot_to,
        top=top,
    )
    try:
        flow = compute_flow(df_to, df_from, spot=spot_to)
        flow_net = aggregate_flow_window([flow], group_by="strike")
    except Exception:  # noqa: BLE001
        warnings.append("flow unavailable: compute failed")
        flow_net = None
    return compare_report, flow_net, warnings


def _compute_symbol_confluence(
    *,
    technicals: TechnicalSnapshot | None,
    flow_net: pd.DataFrame | None,
    derived_row: Any,
    confluence_cfg: Any,
) -> tuple[Any | None, str | None]:
    try:
        trend = _trend_from_weekly_flag(technicals.weekly_trend_up if technicals is not None else None)
        ext_pct = _extension_percentile_from_snapshot(technicals)
        inputs = ConfluenceInputs(
            weekly_trend=trend,
            extension_percentile=ext_pct,
            rsi_divergence=None,
            flow_delta_oi_notional=_net_flow_delta_oi_notional(flow_net),
            iv_rv_20d=(derived_row.iv_rv_20d if derived_row is not None else None),
        )
        return score_confluence(inputs, cfg=confluence_cfg), None
    except Exception as exc:  # noqa: BLE001
        return None, f"confluence unavailable: {exc}"
