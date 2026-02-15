from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

import pandas as pd

from options_helper.analysis.chain_metrics import execution_quality
from options_helper.models import LegSide, MultiLegPosition, OptionType


@dataclass(frozen=True)
class MultiLegRollLeg:
    side: LegSide
    option_type: OptionType
    expiry: str
    dte: int
    strike: float
    contracts: int
    mark: float | None
    bid: float | None
    ask: float | None
    spread_pct: float | None
    execution_quality: str | None
    open_interest: int | None
    volume: int | None


@dataclass(frozen=True)
class MultiLegRollCandidate:
    rank_score: float
    legs: list[MultiLegRollLeg]
    net_mark: float | None
    roll_debit: float | None
    liquidity_ok: bool
    warnings: list[str]
    rationale: list[str]


@dataclass(frozen=True)
class MultiLegRollPlanReport:
    symbol: str
    as_of: str
    spot: float
    position_id: str
    structure: str
    horizon_months: int
    target_dte: int
    current_net_mark: float | None
    current_net_debit: float | None
    current_legs: list[MultiLegRollLeg]
    candidates: list[MultiLegRollCandidate]
    warnings: list[str]


def _as_float(val) -> float | None:
    try:
        if val is None:
            return None
        if pd.isna(val):
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _as_int(val) -> int | None:
    try:
        if val is None:
            return None
        if pd.isna(val):
            return None
        return int(val)
    except Exception:  # noqa: BLE001
        return None


def _mark_price(*, bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    if ask is not None and ask > 0:
        return ask
    if bid is not None and bid > 0:
        return bid
    return None


def _spread_pct(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def _months_to_target_dte(horizon_months: int) -> int:
    return int(round(horizon_months * (365.0 / 12.0)))


def _row_to_leg(
    *,
    side: LegSide,
    option_type: OptionType,
    expiry: date,
    as_of: date,
    strike: float,
    contracts: int,
    row: pd.Series | dict | None,
) -> MultiLegRollLeg:
    bid = ask = last = None
    oi = vol = None
    if row is not None:
        bid = _as_float(row.get("bid"))
        ask = _as_float(row.get("ask"))
        last = _as_float(row.get("lastPrice"))
        oi = _as_int(row.get("openInterest"))
        vol = _as_int(row.get("volume"))

    mark = _mark_price(bid=bid, ask=ask, last=last)
    spread_pct = _spread_pct(bid, ask)
    exec_quality = execution_quality(spread_pct)
    dte = max(0, (expiry - as_of).days)

    return MultiLegRollLeg(
        side=side,
        option_type=option_type,
        expiry=expiry.isoformat(),
        dte=dte,
        strike=float(strike),
        contracts=int(contracts),
        mark=mark,
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        execution_quality=exec_quality,
        open_interest=oi,
        volume=vol,
    )


def _candidate_expiries(
    df: pd.DataFrame,
    *,
    as_of: date,
    option_type: OptionType,
    target_dte: int,
    current_expiry: date,
    top: int,
) -> list[tuple[date, int]]:
    if df is None or df.empty or "expiry" not in df.columns:
        return []
    expiry_raw = pd.to_datetime(df["expiry"], errors="coerce")
    dte = (expiry_raw - pd.Timestamp(as_of)).dt.days
    mask = expiry_raw.notna() & dte.notna() & (dte >= 1)
    if "optionType" in df.columns:
        mask = mask & (df["optionType"].astype(str).str.lower() == option_type)

    if not mask.any():
        return []

    expiry_dt = expiry_raw.dt.date[mask]
    dte_vals = dte[mask].astype(int)
    exp_df = pd.DataFrame({"expiry": expiry_dt, "dte": dte_vals}).drop_duplicates(subset=["expiry"])
    exp_df = exp_df.sort_values(by="dte")

    exp_df["dist"] = (exp_df["dte"] - target_dte).abs()
    exp_df = exp_df.sort_values(by=["dist", "dte"])

    candidates: list[tuple[date, int]] = []
    for _, row in exp_df.iterrows():
        exp = row["expiry"]
        if exp == current_expiry:
            continue
        candidates.append((exp, int(row["dte"])))
        if len(candidates) >= top:
            break
    if not candidates and current_expiry in exp_df["expiry"].values:
        cur_row = exp_df[exp_df["expiry"] == current_expiry].iloc[0]
        candidates.append((current_expiry, int(cur_row["dte"])))
    return candidates


def _infer_option_type_from_contract_symbol(contract_symbol: str | None) -> str | None:
    if contract_symbol is None:
        return None
    raw = str(contract_symbol).strip().upper()
    if not raw:
        return None
    match = re.search(r"\d{6}([CP])\d{8}$", raw)
    if not match:
        return None
    if match.group(1) == "C":
        return "call"
    if match.group(1) == "P":
        return "put"
    return None


def _find_snapshot_row(
    df: pd.DataFrame,
    *,
    expiry: date,
    strike: float,
    option_type: str,
    contract_symbol: str | None = None,
    strike_tol: float = 1e-6,
) -> pd.Series | None:
    if df is None or df.empty:
        return None

    if contract_symbol and "contractSymbol" in df.columns:
        mask = df["contractSymbol"].astype(str) == str(contract_symbol)
        if mask.any():
            return df.loc[mask].iloc[0]

    sub = df
    expiry_str = expiry.isoformat()
    if "expiry" in sub.columns:
        sub = sub[sub["expiry"].astype(str) == expiry_str]
        if sub.empty:
            return None

    option_type_norm = str(option_type).strip().lower()
    if "optionType" in sub.columns:
        sub = sub[sub["optionType"].astype(str).str.lower() == option_type_norm]
        if sub.empty:
            return None
    elif "contractSymbol" in sub.columns and option_type_norm in {"call", "put"}:
        inferred = sub["contractSymbol"].map(_infer_option_type_from_contract_symbol)
        sub = sub[inferred == option_type_norm]
        if sub.empty:
            return None

    if "strike" not in sub.columns:
        return None

    strike_series = pd.to_numeric(sub["strike"], errors="coerce")
    if strike_series.isna().all():
        return None

    diff = (strike_series.astype(float) - float(strike)).abs()
    match = diff < float(strike_tol)
    if match.any():
        return sub.loc[match].iloc[0]
    return None


def compute_roll_plan_multileg(
    df: pd.DataFrame,
    *,
    symbol: str,
    as_of: date,
    spot: float,
    position: MultiLegPosition,
    horizon_months: int,
    min_open_interest: int,
    min_volume: int,
    top: int = 5,
    max_spread_pct: float = 0.35,
    include_bad_quotes: bool = False,
    max_debit: float | None = None,
    min_credit: float | None = None,
) -> MultiLegRollPlanReport:
    warnings: list[str] = []

    if len(position.legs) != 2:
        raise ValueError("Multi-leg roll planner currently supports 2-leg verticals only.")

    leg_sides = {leg.side for leg in position.legs}
    if leg_sides != {"long", "short"}:
        raise ValueError("Multi-leg roll planner requires one long leg and one short leg.")

    option_types = {leg.option_type for leg in position.legs}
    if len(option_types) != 1:
        raise ValueError("Multi-leg roll planner requires legs with the same option type.")

    expiries = {leg.expiry for leg in position.legs}
    if len(expiries) != 1:
        raise ValueError("Multi-leg roll planner currently supports same-expiry verticals only.")

    option_type = next(iter(option_types))
    current_expiry = next(iter(expiries))
    long_leg = next(leg for leg in position.legs if leg.side == "long")
    short_leg = next(leg for leg in position.legs if leg.side == "short")

    width = short_leg.strike - long_leg.strike
    if abs(width) < 1e-6:
        raise ValueError("Multi-leg roll planner requires distinct strikes.")

    current_legs: list[MultiLegRollLeg] = []
    current_net_mark = 0.0
    current_mark_ready = True

    for leg in position.legs:
        row = _find_snapshot_row(df, expiry=leg.expiry, strike=leg.strike, option_type=leg.option_type)
        leg_metrics = _row_to_leg(
            side=leg.side,
            option_type=leg.option_type,
            expiry=leg.expiry,
            as_of=as_of,
            strike=leg.strike,
            contracts=leg.contracts,
            row=row,
        )
        current_legs.append(leg_metrics)
        if leg_metrics.mark is None:
            current_mark_ready = False
        else:
            signed_contracts = leg_metrics.contracts if leg.side == "long" else -leg_metrics.contracts
            current_net_mark += leg_metrics.mark * float(signed_contracts) * 100.0

    current_net = current_net_mark if current_mark_ready else None
    if current_net is None:
        warnings.append("missing_current_mark")

    target_dte = _months_to_target_dte(horizon_months)
    expiry_candidates = _candidate_expiries(
        df,
        as_of=as_of,
        option_type=option_type,
        target_dte=target_dte,
        current_expiry=current_expiry,
        top=top * 3,
    )

    candidates: list[MultiLegRollCandidate] = []

    for expiry, dte in expiry_candidates:
        strike_long = long_leg.strike
        strike_short = strike_long + width

        row_long = _find_snapshot_row(df, expiry=expiry, strike=strike_long, option_type=option_type)
        row_short = _find_snapshot_row(df, expiry=expiry, strike=strike_short, option_type=option_type)
        if row_long is None or row_short is None:
            continue

        leg_long = _row_to_leg(
            side=long_leg.side,
            option_type=option_type,
            expiry=expiry,
            as_of=as_of,
            strike=strike_long,
            contracts=long_leg.contracts,
            row=row_long,
        )
        leg_short = _row_to_leg(
            side=short_leg.side,
            option_type=option_type,
            expiry=expiry,
            as_of=as_of,
            strike=strike_short,
            contracts=short_leg.contracts,
            row=row_short,
        )

        leg_list = [leg_long, leg_short]
        net_mark_ready = True
        net_mark = 0.0
        warn: list[str] = []
        liquid = True

        for leg_metrics in leg_list:
            if leg_metrics.mark is None:
                net_mark_ready = False
            else:
                signed_contracts = leg_metrics.contracts if leg_metrics.side == "long" else -leg_metrics.contracts
                net_mark += leg_metrics.mark * float(signed_contracts) * 100.0

            if leg_metrics.open_interest is None or leg_metrics.open_interest < min_open_interest:
                liquid = False
                warn.append("low_open_interest")
            if leg_metrics.volume is None or leg_metrics.volume < min_volume:
                liquid = False
                warn.append("low_volume")

            if (
                not include_bad_quotes
                and leg_metrics.spread_pct is not None
                and leg_metrics.spread_pct > max_spread_pct
            ):
                liquid = False
                warn.append("wide_spread")

        net_mark_val = net_mark if net_mark_ready else None
        if net_mark_val is None:
            continue
        roll_debit = None
        if net_mark_val is not None and current_net is not None:
            roll_debit = net_mark_val - current_net

        if roll_debit is not None:
            if max_debit is not None and roll_debit > max_debit:
                continue
            if min_credit is not None and roll_debit > -abs(min_credit):
                continue

        rank_score = -abs(dte - target_dte)
        if liquid:
            rank_score += 1000.0

        rationale = [
            f"Expiry {dte} DTE vs target {target_dte}.",
            "Kept strikes to preserve width.",
        ]

        candidates.append(
            MultiLegRollCandidate(
                rank_score=rank_score,
                legs=leg_list,
                net_mark=net_mark_val,
                roll_debit=roll_debit,
                liquidity_ok=liquid,
                warnings=sorted(set(warn)),
                rationale=rationale,
            )
        )

    if not candidates:
        warnings.append("no_candidates_found")

    candidates = sorted(candidates, key=lambda c: c.rank_score, reverse=True)[:top]

    return MultiLegRollPlanReport(
        symbol=symbol,
        as_of=as_of.isoformat(),
        spot=spot,
        position_id=position.id,
        structure="vertical",
        horizon_months=horizon_months,
        target_dte=target_dte,
        current_net_mark=current_net,
        current_net_debit=position.net_debit,
        current_legs=current_legs,
        candidates=candidates,
        warnings=warnings,
    )
