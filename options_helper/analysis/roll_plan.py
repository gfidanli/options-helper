from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, Field

from options_helper.analysis.chain_metrics import compute_mark_price, execution_quality
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.greeks import black_scholes_greeks
from options_helper.analysis.osi import format_osi, parse_contract_symbol
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.models import OptionType, Position

RollIntent = Literal["max-upside", "reduce-theta", "increase-delta", "de-risk"]
RollShape = Literal["out-same-strike", "out-up", "out-down"]


def _col_as_float(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


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


def _parse_iso_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        s = str(val).strip()
        if not s:
            return None
        return date.fromisoformat(s)
    except Exception:  # noqa: BLE001
        return None


def _osi_from_row(row: dict) -> str | None:
    if row is None:
        return None
    if "osi" in row:
        value = row.get("osi")
        if value is not None and not pd.isna(value):
            text = str(value).strip()
            if text:
                return text
    raw = row.get("contractSymbol")
    parsed = parse_contract_symbol(raw)
    if parsed is None:
        return None
    try:
        return format_osi(parsed)
    except ValueError:
        return None


def _months_to_target_dte(horizon_months: int) -> int:
    # Use 365/12 to avoid overly coarse 30-day months.
    return int(round(horizon_months * (365.0 / 12.0)))


def _best_effort_delta_theta(
    row: dict,
    *,
    option_type: OptionType,
    spot: float,
    as_of: date,
) -> tuple[float | None, float | None]:
    delta = _as_float(row.get("bs_delta"))
    theta = _as_float(row.get("bs_theta_per_day"))
    if delta is not None and theta is not None:
        return (delta, theta)

    iv = _as_float(row.get("impliedVolatility"))
    strike = _as_float(row.get("strike"))
    expiry = _parse_iso_date(row.get("expiry"))
    if iv is None or strike is None or expiry is None:
        return (delta, theta)

    dte = (expiry - as_of).days
    t_years = dte / 365.0 if dte > 0 else None
    if t_years is None or t_years <= 0:
        return (delta, theta)

    g = black_scholes_greeks(option_type=option_type, s=spot, k=strike, t_years=t_years, sigma=iv)
    if g is None:
        return (delta, theta)

    return (delta if delta is not None else g.delta, theta if theta is not None else g.theta_per_day)


class RollContract(BaseModel):
    contract_symbol: str | None = None
    osi: str | None = None
    option_type: OptionType
    expiry: str
    dte: int = Field(ge=0)
    strike: float = Field(gt=0.0)

    mark: float | None = Field(default=None, ge=0.0)
    bid: float | None = Field(default=None, ge=0.0)
    ask: float | None = Field(default=None, ge=0.0)
    spread: float | None = Field(default=None, ge=0.0)
    spread_pct: float | None = Field(default=None, ge=0.0)
    execution_quality: str | None = None
    quality_score: float | None = None
    quality_label: str | None = None
    last_trade_age_days: int | None = None
    quality_warnings: list[str] = Field(default_factory=list)

    implied_vol: float | None = Field(default=None, ge=0.0)
    open_interest: int | None = Field(default=None, ge=0)
    volume: int | None = Field(default=None, ge=0)

    delta: float | None = None
    theta_per_day: float | None = None


class RollCandidate(BaseModel):
    rank_score: float
    contract: RollContract
    roll_debit: float | None = None
    roll_debit_per_contract: float | None = None
    liquidity_ok: bool = True
    issues: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RollPlanReport(BaseModel):
    schema_version: int = 1
    symbol: str
    as_of: str
    spot: float
    position_id: str
    contracts: int
    intent: RollIntent
    shape: RollShape
    horizon_months: int
    target_dte: int
    current: RollContract
    candidates: list[RollCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _CandidateRow:
    row: dict
    expiry: date
    dte: int
    strike: float


def _strike_in_shape(strike: float, *, current_strike: float, shape: RollShape) -> bool:
    if shape == "out-same-strike":
        return abs(strike - current_strike) <= 1e-6
    if shape == "out-up":
        return strike > current_strike + 1e-6
    return strike < current_strike - 1e-6


def _pick_base_strike(sub: pd.DataFrame, *, current_strike: float, shape: RollShape) -> float | None:
    if sub.empty or "strike" not in sub.columns:
        return None
    strikes = pd.to_numeric(sub["strike"], errors="coerce").dropna().astype(float).unique().tolist()
    if not strikes:
        return None
    strikes = sorted(set(strikes))

    if shape == "out-same-strike":
        return min(strikes, key=lambda s: abs(s - current_strike))

    if shape == "out-up":
        above = [s for s in strikes if s > current_strike + 1e-6]
        if above:
            return min(above, key=lambda s: abs(s - current_strike))
        return min(strikes, key=lambda s: abs(s - current_strike))

    # out-down
    below = [s for s in strikes if s < current_strike - 1e-6]
    if below:
        return min(below, key=lambda s: abs(s - current_strike))
    return min(strikes, key=lambda s: abs(s - current_strike))


def _pick_by_delta(
    sub: pd.DataFrame,
    *,
    target_delta: float,
    current_strike: float,
    shape: RollShape,
) -> float | None:
    if sub.empty:
        return None
    if "bs_delta" not in sub.columns or "strike" not in sub.columns:
        return None

    strike = pd.to_numeric(sub["strike"], errors="coerce")
    delta = pd.to_numeric(sub["bs_delta"], errors="coerce")
    cand = sub.assign(_strike=strike, _delta=delta).dropna(subset=["_strike", "_delta"])
    if cand.empty:
        return None

    cand = cand[cand["_strike"].map(lambda s: _strike_in_shape(float(s), current_strike=current_strike, shape=shape))]
    if cand.empty:
        return None

    cand = cand.assign(_dist=(cand["_delta"] - target_delta).abs())
    cand = cand.sort_values(["_dist", "_strike"], ascending=[True, True])
    return float(cand.iloc[0]["_strike"])


def compute_roll_plan(
    df: pd.DataFrame,
    *,
    symbol: str,
    as_of: date,
    spot: float,
    position: Position,
    intent: RollIntent,
    horizon_months: int,
    shape: RollShape,
    min_open_interest: int,
    min_volume: int,
    max_debit: float | None = None,
    min_credit: float | None = None,
    top: int = 10,
    max_spread_pct: float = 0.35,
    next_earnings_date: date | None = None,
    earnings_warn_days: int = 21,
    earnings_avoid_days: int = 0,
    include_bad_quotes: bool = False,
) -> RollPlanReport:
    warnings: list[str] = []
    excluded_warnings: set[str] = set()

    if df is None or df.empty:
        raise ValueError("empty snapshot data")
    if "optionType" not in df.columns or "expiry" not in df.columns or "strike" not in df.columns:
        raise ValueError("snapshot missing required columns (need optionType, expiry, strike)")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be > 0")
    if top <= 0:
        raise ValueError("top must be > 0")

    option_type: OptionType = position.option_type
    target_dte = _months_to_target_dte(horizon_months)

    # Normalize chain for this option type.
    chain = df[df["optionType"] == option_type].copy()
    if chain.empty:
        raise ValueError(f"no {option_type} rows in snapshot")

    chain["_expiry"] = chain["expiry"].map(_parse_iso_date)
    chain["_strike"] = pd.to_numeric(chain["strike"], errors="coerce")
    chain = chain.dropna(subset=["_expiry", "_strike"]).copy()
    if chain.empty:
        raise ValueError("no parseable expiry/strike rows in snapshot")

    chain["_dte"] = chain["_expiry"].map(lambda d: int((d - as_of).days))
    chain = chain[chain["_dte"] >= 0].copy()
    if chain.empty:
        raise ValueError("snapshot contains no non-expired contracts for as-of date")

    # Add a deterministic mark field for pricing.
    chain["mark"] = compute_mark_price(chain)
    quality = compute_quote_quality(chain, min_volume=min_volume, min_open_interest=min_open_interest, as_of=as_of)
    chain["_spread"] = quality["spread"]
    chain["_spread_pct"] = quality["spread_pct"]
    chain["_quality_score"] = quality["quality_score"]
    chain["_quality_label"] = quality["quality_label"]
    chain["_quality_warnings"] = quality["quality_warnings"]
    chain["_last_trade_age_days"] = quality["last_trade_age_days"]

    current_exp = position.expiry
    current_dte = (current_exp - as_of).days
    if current_dte < 0:
        raise ValueError(f"position is expired as-of {as_of.isoformat()} (position expiry {current_exp.isoformat()})")

    # Locate the current contract (prefer exact strike match; else nearest strike on same expiry).
    same_exp = chain[chain["_expiry"] == current_exp].copy()
    if same_exp.empty:
        raise ValueError(
            f"position expiry not in snapshot ({current_exp.isoformat()}); re-run snapshot-options with --all-expiries"
        )

    same_exp["_dist_strike"] = (same_exp["_strike"] - float(position.strike)).abs()
    same_exp = same_exp.sort_values(["_dist_strike", "_strike"], ascending=[True, True])
    cur_row = same_exp.iloc[0].to_dict()
    cur_strike = float(cur_row.get("_strike"))
    if abs(cur_strike - float(position.strike)) > 1e-6:
        warnings.append(f"exact_strike_missing_used_nearest (wanted {position.strike:g}, got {cur_strike:g})")

    cur_bid = _as_float(cur_row.get("bid"))
    cur_ask = _as_float(cur_row.get("ask"))
    cur_mark = _as_float(cur_row.get("mark"))
    if cur_mark is None or cur_mark <= 0:
        raise ValueError("missing current contract mark price in snapshot")

    cur_spread = _as_float(cur_row.get("_spread"))
    cur_spread_pct = _as_float(cur_row.get("_spread_pct"))
    cur_exec_quality = execution_quality(cur_spread_pct)
    cur_quality_score = _as_float(cur_row.get("_quality_score"))
    cur_quality_label = cur_row.get("_quality_label")
    cur_last_trade_age_days = _as_int(cur_row.get("_last_trade_age_days"))
    cur_quality_warnings = cur_row.get("_quality_warnings")
    cur_delta, cur_theta = _best_effort_delta_theta(cur_row, option_type=option_type, spot=spot, as_of=as_of)

    current = RollContract(
        contract_symbol=str(cur_row.get("contractSymbol")) if cur_row.get("contractSymbol") is not None else None,
        osi=_osi_from_row(cur_row),
        option_type=option_type,
        expiry=current_exp.isoformat(),
        dte=current_dte,
        strike=cur_strike,
        mark=cur_mark,
        bid=cur_bid,
        ask=cur_ask,
        spread=cur_spread,
        spread_pct=cur_spread_pct,
        execution_quality=cur_exec_quality,
        quality_score=cur_quality_score,
        quality_label=str(cur_quality_label) if cur_quality_label is not None else None,
        last_trade_age_days=cur_last_trade_age_days,
        quality_warnings=cast(list[str], cur_quality_warnings) if cur_quality_warnings is not None else [],
        implied_vol=_as_float(cur_row.get("impliedVolatility")),
        open_interest=_as_int(cur_row.get("openInterest")),
        volume=_as_int(cur_row.get("volume")),
        delta=cur_delta,
        theta_per_day=cur_theta,
    )
    if current.quality_warnings:
        for w in current.quality_warnings:
            if w not in warnings:
                warnings.append(w)

    # Candidate expiry selection: expiries near target DTE (±90d), preferring later expiries (roll out).
    expiries = sorted({d for d in chain["_expiry"].tolist() if isinstance(d, date)})
    by_dte = {e: int((e - as_of).days) for e in expiries}
    window_days = 90
    near = [e for e in expiries if abs(by_dte[e] - target_dte) <= window_days]
    if not near:
        near = sorted(expiries, key=lambda e: (abs(by_dte[e] - target_dte), e))[:6]
        warnings.append("no_expiries_within_target_window_used_nearest")

    preferred = [e for e in near if e > current_exp]
    if not preferred:
        preferred = [e for e in near if e >= current_exp]
        warnings.append("no_later_expiries_in_window")

    preferred = sorted(preferred, key=lambda e: (abs(by_dte[e] - target_dte), e))[:6]

    # Candidate strike selection by intent (best-effort).
    target_deltas: list[float] = []
    if intent == "max-upside":
        target_deltas = [0.30, 0.40, 0.50] if option_type == "call" else [-0.30, -0.40, -0.50]

    candidate_rows: list[_CandidateRow] = []
    used_keys: set[tuple[date, float]] = set()

    for exp in preferred:
        exp_df = chain[chain["_expiry"] == exp].copy()
        if exp_df.empty:
            continue

        base_strike = _pick_base_strike(exp_df, current_strike=cur_strike, shape=shape)
        if base_strike is not None:
            if _strike_in_shape(base_strike, current_strike=cur_strike, shape=shape):
                key = (exp, float(base_strike))
                if key not in used_keys:
                    used_keys.add(key)
                    pick = exp_df.assign(_dist=(exp_df["_strike"] - base_strike).abs()).sort_values(
                        ["_dist", "_strike"], ascending=[True, True]
                    )
                    row = pick.iloc[0].to_dict()
                    candidate_rows.append(
                        _CandidateRow(row=row, expiry=exp, dte=int(by_dte[exp]), strike=float(base_strike))
                    )

        for td in target_deltas:
            strike = _pick_by_delta(exp_df, target_delta=td, current_strike=cur_strike, shape=shape)
            if strike is None:
                continue
            key = (exp, float(strike))
            if key in used_keys:
                continue
            used_keys.add(key)
            pick = exp_df.assign(_dist=(exp_df["_strike"] - strike).abs()).sort_values(
                ["_dist", "_strike"], ascending=[True, True]
            )
            row = pick.iloc[0].to_dict()
            candidate_rows.append(_CandidateRow(row=row, expiry=exp, dte=int(by_dte[exp]), strike=float(strike)))

    # Build candidate records.
    candidates: list[RollCandidate] = []
    for cr in candidate_rows:
        row = dict(cr.row)
        strike = float(row.get("_strike", cr.strike))
        if cr.expiry == current_exp and abs(strike - cur_strike) <= 1e-6:
            continue

        bid = _as_float(row.get("bid"))
        ask = _as_float(row.get("ask"))
        mark = _as_float(row.get("mark"))
        iv = _as_float(row.get("impliedVolatility"))
        oi = _as_int(row.get("openInterest"))
        vol = _as_int(row.get("volume"))

        spread = _as_float(row.get("_spread"))
        spread_pct = _as_float(row.get("_spread_pct"))
        exec_quality = execution_quality(spread_pct)
        quality_score = _as_float(row.get("_quality_score"))
        quality_label = row.get("_quality_label")
        last_trade_age_days = _as_int(row.get("_last_trade_age_days"))
        quality_warnings = row.get("_quality_warnings")
        spread_ok = None
        if spread_pct is not None:
            spread_ok = (spread_pct >= 0) and (spread_pct <= max_spread_pct)
        delta, theta = _best_effort_delta_theta(row, option_type=option_type, spot=spot, as_of=as_of)

        issues: list[str] = []
        oi_ok = (oi is not None) and (oi >= min_open_interest)
        vol_ok = (vol is not None) and (vol >= min_volume)
        if not (oi_ok or vol_ok):
            issues.append("illiquid_oi_vol")
        if spread_ok is False:
            issues.append("wide_or_invalid_spread")
        if spread_ok is None:
            issues.append("missing_bid_ask")
        if mark is None or mark <= 0:
            issues.append("missing_mark")

        roll_debit_per_contract = None
        roll_debit = None
        if mark is not None and mark > 0 and cur_mark is not None and cur_mark > 0:
            roll_debit_per_contract = (mark - cur_mark) * 100.0
            roll_debit = roll_debit_per_contract * float(position.contracts)
        else:
            issues.append("missing_roll_cost")

        # Debit/credit gating (total $ for the roll).
        if max_debit is not None:
            if roll_debit is None or roll_debit > max_debit:
                issues.append("over_max_debit")
        if min_credit is not None:
            credit = None if roll_debit is None else (-roll_debit)
            if credit is None or credit < min_credit:
                issues.append("below_min_credit")

        liquidity_ok = ("illiquid_oi_vol" not in issues) and ("wide_or_invalid_spread" not in issues)

        # Rank score: higher is better (used only for sorting deterministically).
        dte_penalty = abs(int(cr.dte) - int(target_dte))
        debit_penalty = 0.0 if roll_debit is None else max(0.0, roll_debit) / 1000.0
        delta_target = 0.40 if option_type == "call" else -0.40
        delta_penalty = 0.0 if delta is None else abs(delta - delta_target)
        liq_penalty = 10.0 if not liquidity_ok else 0.0
        score = -(dte_penalty + (10.0 * debit_penalty) + (50.0 * delta_penalty) + liq_penalty)

        rationale: list[str] = []
        rationale.append(f"Horizon fit: DTE {cr.dte} vs target {target_dte} (Δ={dte_penalty}d).")
        if roll_debit is not None:
            rationale.append(f"Roll cost: {roll_debit:+.0f} total (${roll_debit_per_contract:+.0f}/contract).")
        if delta is not None:
            rationale.append(f"Delta: {delta:+.2f}" + ("" if cur_delta is None else f" (current {cur_delta:+.2f})."))
        if theta is not None:
            rationale.append(
                f"Theta/day: {theta:+.4f}" + ("" if cur_theta is None else f" (current {cur_theta:+.4f}).")
            )
        rationale.append(
            "Liquidity: "
            + f"OI={oi if oi is not None else 'n/a'}, Vol={vol if vol is not None else 'n/a'}"
            + ("" if spread_pct is None else f", spread={spread_pct*100.0:.1f}%")
            + "."
        )

        event_risk = earnings_event_risk(
            today=as_of,
            expiry=cr.expiry,
            next_earnings_date=next_earnings_date,
            warn_days=earnings_warn_days,
            avoid_days=earnings_avoid_days,
        )
        risk_warnings = list(event_risk["warnings"])
        if event_risk["exclude"]:
            excluded_warnings.update(risk_warnings)
            continue

        contract = RollContract(
            contract_symbol=str(row.get("contractSymbol")) if row.get("contractSymbol") is not None else None,
            osi=_osi_from_row(row),
            option_type=option_type,
            expiry=cr.expiry.isoformat(),
            dte=int(cr.dte),
            strike=strike,
            mark=mark,
            bid=bid,
            ask=ask,
            spread=spread,
            spread_pct=spread_pct,
            execution_quality=exec_quality,
            quality_score=quality_score,
            quality_label=str(quality_label) if quality_label is not None else None,
            last_trade_age_days=last_trade_age_days,
            quality_warnings=cast(list[str], quality_warnings) if quality_warnings is not None else [],
            implied_vol=iv,
            open_interest=oi,
            volume=vol,
            delta=delta,
            theta_per_day=theta,
        )
        combined_warnings = list(risk_warnings)
        if contract.quality_warnings:
            combined_warnings.extend(contract.quality_warnings)
        combined_warnings = sorted(set(combined_warnings))
        candidates.append(
            RollCandidate(
                rank_score=float(score),
                contract=contract,
                roll_debit=roll_debit,
                roll_debit_per_contract=roll_debit_per_contract,
                liquidity_ok=bool(liquidity_ok),
                issues=issues,
                rationale=rationale,
                warnings=combined_warnings,
            )
        )

    # Prefer candidates that pass liquidity and cost constraints; fall back if none.
    filtered = [
        c
        for c in candidates
        if c.liquidity_ok and ("missing_mark" not in c.issues) and ("missing_roll_cost" not in c.issues)
    ]
    if max_debit is not None:
        filtered = [c for c in filtered if "over_max_debit" not in c.issues]
    if min_credit is not None:
        filtered = [c for c in filtered if "below_min_credit" not in c.issues]
    if not include_bad_quotes:
        filtered = [c for c in filtered if c.contract.quality_label != "bad"]

    if not filtered and candidates:
        warnings.append("no_candidates_passed_gates_showing_best_effort")
        filtered = candidates

    if excluded_warnings:
        warnings.extend(sorted(excluded_warnings))

    # Sort deterministically: best score first, then expiry/strike.
    filtered.sort(
        key=lambda c: (
            -c.rank_score,
            c.contract.expiry,
            c.contract.strike,
        )
    )

    return RollPlanReport(
        symbol=symbol.upper(),
        as_of=as_of.isoformat(),
        spot=float(spot),
        position_id=position.id,
        contracts=int(position.contracts),
        intent=intent,
        shape=shape,
        horizon_months=int(horizon_months),
        target_dte=int(target_dte),
        current=current,
        candidates=filtered[:top],
        warnings=warnings,
    )
