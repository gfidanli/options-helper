from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, Field

from options_helper.analysis.chain_metrics import ChainReport, compute_chain_report, select_expiries
from options_helper.analysis.flow import compute_flow


def _parse_expiry_dates(df: pd.DataFrame) -> list[date]:
    if df is None or df.empty or "expiry" not in df.columns:
        return []
    out: list[date] = []
    for s in sorted({str(x) for x in df["expiry"].dropna().unique().tolist()}):
        try:
            out.append(date.fromisoformat(s))
        except ValueError:
            continue
    return sorted(set(out))


def _oi_by_strike(df: pd.DataFrame, *, option_type: str, expiry: str | None = None) -> dict[float, float]:
    if df is None or df.empty or "optionType" not in df.columns or "strike" not in df.columns:
        return {}
    sub = df[df["optionType"] == option_type]
    if expiry is not None and "expiry" in sub.columns:
        sub = sub[sub["expiry"] == expiry]
    if sub.empty or "openInterest" not in sub.columns:
        return {}
    strike = pd.to_numeric(sub["strike"], errors="coerce")
    oi = pd.to_numeric(sub["openInterest"], errors="coerce")
    sub = sub.assign(_strike=strike, _oi=oi).dropna(subset=["_strike", "_oi"])
    if sub.empty:
        return {}
    grouped = sub.groupby("_strike", as_index=False)["_oi"].sum()
    return {float(r["_strike"]): float(r["_oi"]) for _, r in grouped.iterrows()}


class WallDeltaLevel(BaseModel):
    strike: float
    oi_from: float = Field(ge=0.0)
    oi_to: float = Field(ge=0.0)
    delta_oi: float


class WallsDelta(BaseModel):
    calls: list[WallDeltaLevel] = Field(default_factory=list)
    puts: list[WallDeltaLevel] = Field(default_factory=list)


class WallsDeltaByExpiry(BaseModel):
    expiry: str
    walls: WallsDelta


class ExpiryCompare(BaseModel):
    expiry: str
    atm_iv_from: float | None = None
    atm_iv_to: float | None = None
    atm_iv_change_pp: float | None = None
    expected_move_from: float | None = None
    expected_move_to: float | None = None
    expected_move_change: float | None = None
    expected_move_pct_from: float | None = None
    expected_move_pct_to: float | None = None
    expected_move_pct_change_pp: float | None = None


class FlowContract(BaseModel):
    contract_symbol: str
    osi: str | None = None
    expiry: str | None = None
    option_type: str | None = None
    strike: float | None = None
    delta_oi: float | None = None
    delta_oi_notional: float | None = None
    flow_class: str | None = None


class FlowExpiryNet(BaseModel):
    expiry: str
    delta_oi_notional: float


class CompareReport(BaseModel):
    schema_version: int = 1
    symbol: str
    from_date: str
    to_date: str
    spot_from: float
    spot_to: float
    spot_change: float
    spot_change_pct: float | None
    pc_oi_ratio_from: float | None = None
    pc_oi_ratio_to: float | None = None
    pc_oi_ratio_change: float | None = None
    pc_volume_ratio_from: float | None = None
    pc_volume_ratio_to: float | None = None
    pc_volume_ratio_change: float | None = None
    expiries: list[ExpiryCompare] = Field(default_factory=list)
    walls_overall: WallsDelta
    walls_by_expiry: list[WallsDeltaByExpiry] = Field(default_factory=list)
    flow_top_contracts: list[FlowContract] = Field(default_factory=list)
    flow_top_expiries: list[FlowExpiryNet] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _wall_deltas(
    *,
    from_oi: dict[float, float],
    to_oi: dict[float, float],
    strikes: list[float],
    top: int,
) -> list[WallDeltaLevel]:
    levels: list[WallDeltaLevel] = []
    for strike in strikes:
        oi_from = float(from_oi.get(strike, 0.0))
        oi_to = float(to_oi.get(strike, 0.0))
        levels.append(WallDeltaLevel(strike=float(strike), oi_from=oi_from, oi_to=oi_to, delta_oi=oi_to - oi_from))
    levels.sort(key=lambda x: (abs(x.delta_oi), x.strike), reverse=True)
    return levels[:top]


def compute_compare_report(
    *,
    symbol: str,
    from_date: date,
    to_date: date,
    from_df: pd.DataFrame,
    to_df: pd.DataFrame,
    spot_from: float,
    spot_to: float,
    top: int = 10,
) -> tuple[CompareReport, ChainReport, ChainReport]:
    warnings: list[str] = []

    from_exp = _parse_expiry_dates(from_df)
    to_exp = _parse_expiry_dates(to_df)
    common_exp = sorted(set(from_exp).intersection(to_exp))
    selected_exp = select_expiries(common_exp, mode="near")

    if not common_exp:
        warnings.append("no_common_expiries")

    report_from = compute_chain_report(
        from_df,
        symbol=symbol,
        as_of=from_date,
        spot=spot_from,
        expiries_mode="all",
        include_expiries=selected_exp,
        top=top,
        best_effort=True,
    )
    report_to = compute_chain_report(
        to_df,
        symbol=symbol,
        as_of=to_date,
        spot=spot_to,
        expiries_mode="all",
        include_expiries=selected_exp,
        top=top,
        best_effort=True,
    )

    spot_change = float(spot_to - spot_from)
    spot_change_pct = (spot_change / spot_from) if spot_from > 0 else None

    pc_oi_from = report_from.totals.pc_oi_ratio
    pc_oi_to = report_to.totals.pc_oi_ratio
    pc_oi_change = (pc_oi_to - pc_oi_from) if (pc_oi_from is not None and pc_oi_to is not None) else None

    pc_vol_from = report_from.totals.pc_volume_ratio
    pc_vol_to = report_to.totals.pc_volume_ratio
    pc_vol_change = (pc_vol_to - pc_vol_from) if (pc_vol_from is not None and pc_vol_to is not None) else None

    expiry_diffs: list[ExpiryCompare] = []
    from_by_exp = {e.expiry: e for e in report_from.expiries}
    to_by_exp = {e.expiry: e for e in report_to.expiries}
    for exp in report_to.included_expiries:
        a = from_by_exp.get(exp)
        b = to_by_exp.get(exp)
        if a is None or b is None:
            continue

        iv_change_pp = None
        if a.atm_iv is not None and b.atm_iv is not None:
            iv_change_pp = (b.atm_iv - a.atm_iv) * 100.0

        em_change = None
        if a.expected_move is not None and b.expected_move is not None:
            em_change = b.expected_move - a.expected_move

        em_pct_change_pp = None
        if a.expected_move_pct is not None and b.expected_move_pct is not None:
            em_pct_change_pp = (b.expected_move_pct - a.expected_move_pct) * 100.0

        expiry_diffs.append(
            ExpiryCompare(
                expiry=exp,
                atm_iv_from=a.atm_iv,
                atm_iv_to=b.atm_iv,
                atm_iv_change_pp=iv_change_pp,
                expected_move_from=a.expected_move,
                expected_move_to=b.expected_move,
                expected_move_change=em_change,
                expected_move_pct_from=a.expected_move_pct,
                expected_move_pct_to=b.expected_move_pct,
                expected_move_pct_change_pp=em_pct_change_pp,
            )
        )

    # Walls: compute deltas at strikes that are top walls on either day (overall + per-expiry).
    from_call_oi = _oi_by_strike(from_df, option_type="call")
    to_call_oi = _oi_by_strike(to_df, option_type="call")
    from_put_oi = _oi_by_strike(from_df, option_type="put")
    to_put_oi = _oi_by_strike(to_df, option_type="put")

    call_strikes = sorted(
        {lvl.strike for lvl in report_from.walls_overall.calls}.union({lvl.strike for lvl in report_to.walls_overall.calls})
    )
    put_strikes = sorted(
        {lvl.strike for lvl in report_from.walls_overall.puts}.union({lvl.strike for lvl in report_to.walls_overall.puts})
    )

    walls_overall = WallsDelta(
        calls=_wall_deltas(from_oi=from_call_oi, to_oi=to_call_oi, strikes=call_strikes, top=top),
        puts=_wall_deltas(from_oi=from_put_oi, to_oi=to_put_oi, strikes=put_strikes, top=top),
    )

    walls_by_expiry: list[WallsDeltaByExpiry] = []
    from_walls_exp = {w.expiry: w.walls for w in report_from.walls_by_expiry}
    to_walls_exp = {w.expiry: w.walls for w in report_to.walls_by_expiry}
    for exp in report_to.included_expiries:
        fw = from_walls_exp.get(exp)
        tw = to_walls_exp.get(exp)
        if fw is None or tw is None:
            continue

        from_call_oi_e = _oi_by_strike(from_df, option_type="call", expiry=exp)
        to_call_oi_e = _oi_by_strike(to_df, option_type="call", expiry=exp)
        from_put_oi_e = _oi_by_strike(from_df, option_type="put", expiry=exp)
        to_put_oi_e = _oi_by_strike(to_df, option_type="put", expiry=exp)

        call_strikes_e = sorted({lvl.strike for lvl in fw.calls}.union({lvl.strike for lvl in tw.calls}))
        put_strikes_e = sorted({lvl.strike for lvl in fw.puts}.union({lvl.strike for lvl in tw.puts}))

        walls_by_expiry.append(
            WallsDeltaByExpiry(
                expiry=exp,
                walls=WallsDelta(
                    calls=_wall_deltas(from_oi=from_call_oi_e, to_oi=to_call_oi_e, strikes=call_strikes_e, top=top),
                    puts=_wall_deltas(from_oi=from_put_oi_e, to_oi=to_put_oi_e, strikes=put_strikes_e, top=top),
                ),
            )
        )

    # Flow (contract deltas) where available.
    flow_top_contracts: list[FlowContract] = []
    flow_top_expiries: list[FlowExpiryNet] = []
    try:
        non_zero_flow = pd.DataFrame()
        flow = compute_flow(to_df, from_df, spot=spot_to)
        if not flow.empty:
            prev_oi = pd.to_numeric(flow.get("openInterest_prev"), errors="coerce") if "openInterest_prev" in flow.columns else None
            prev_matched = int(prev_oi.notna().sum()) if prev_oi is not None else 0
            total_contracts = int(len(flow))
            if total_contracts > 0 and prev_matched == 0:
                warnings.append(f"flow_prev_oi_unmatched (0/{total_contracts} contracts)")
            elif total_contracts > 0 and prev_matched < total_contracts:
                warnings.append(f"flow_prev_oi_partial ({prev_matched}/{total_contracts} contracts)")

            delta_oi = pd.to_numeric(flow.get("deltaOI"), errors="coerce") if "deltaOI" in flow.columns else None
            if delta_oi is not None and prev_matched > 0:
                non_zero = (delta_oi.abs() > 0).any()
                if not bool(non_zero):
                    warnings.append(f"oi_unchanged_between_snapshots ({prev_matched} contracts matched)")

        if not flow.empty and "deltaOI_notional" in flow.columns:
            flow = flow.copy()
            flow["deltaOI_notional"] = pd.to_numeric(flow.get("deltaOI_notional"), errors="coerce")

            non_zero_flow = flow[flow["deltaOI_notional"].abs() > 0].copy()
            if non_zero_flow.empty:
                # Avoid rendering misleading "top" tables when there's no meaningful notional deltas.
                non_zero_flow = pd.DataFrame()

        if not flow.empty and "deltaOI_notional" in flow.columns and not non_zero_flow.empty:
            non_zero_flow = non_zero_flow.assign(_abs=non_zero_flow["deltaOI_notional"].abs()).sort_values(
                "_abs", ascending=False
            )
            for _, row in non_zero_flow.head(top).iterrows():
                strike = row.get("strike")
                strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
                osi_raw = row.get("osi")
                osi_val = None
                if osi_raw is not None and not pd.isna(osi_raw):
                    osi_val = str(osi_raw).strip() or None
                delta_oi = row.get("deltaOI")
                delta_oi_val = float(delta_oi) if delta_oi is not None and not pd.isna(delta_oi) else None
                delta_notional = row.get("deltaOI_notional")
                delta_notional_val = (
                    float(delta_notional) if delta_notional is not None and not pd.isna(delta_notional) else None
                )
                flow_top_contracts.append(
                    FlowContract(
                        contract_symbol=str(row.get("contractSymbol")),
                        osi=osi_val,
                        expiry=str(row.get("expiry")) if row.get("expiry") is not None else None,
                        option_type=str(row.get("optionType")) if row.get("optionType") is not None else None,
                        strike=strike_val,
                        delta_oi=delta_oi_val,
                        delta_oi_notional=delta_notional_val,
                        flow_class=str(row.get("flow_class")) if row.get("flow_class") is not None else None,
                    )
                )

            expiry_net = non_zero_flow.groupby("expiry", as_index=False)["deltaOI_notional"].sum()
            expiry_net = expiry_net.assign(_abs=expiry_net["deltaOI_notional"].abs()).sort_values("_abs", ascending=False)
            for _, row in expiry_net.head(top).iterrows():
                flow_top_expiries.append(
                    FlowExpiryNet(expiry=str(row["expiry"]), delta_oi_notional=float(row["deltaOI_notional"]))
                )
    except Exception:  # noqa: BLE001
        warnings.append("flow_failed")

    report = CompareReport(
        symbol=symbol.upper(),
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        spot_from=float(spot_from),
        spot_to=float(spot_to),
        spot_change=spot_change,
        spot_change_pct=spot_change_pct,
        pc_oi_ratio_from=pc_oi_from,
        pc_oi_ratio_to=pc_oi_to,
        pc_oi_ratio_change=pc_oi_change,
        pc_volume_ratio_from=pc_vol_from,
        pc_volume_ratio_to=pc_vol_to,
        pc_volume_ratio_change=pc_vol_change,
        expiries=expiry_diffs,
        walls_overall=walls_overall,
        walls_by_expiry=walls_by_expiry,
        flow_top_contracts=flow_top_contracts,
        flow_top_expiries=flow_top_expiries,
        warnings=warnings,
    )
    return report, report_from, report_to
