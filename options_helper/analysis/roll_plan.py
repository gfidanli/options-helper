from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping, Sequence, cast

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


@dataclass(frozen=True)
class _CurrentSelection:
    contract: RollContract
    strike: float
    mark: float
    delta: float | None
    theta: float | None


@dataclass(frozen=True)
class _CandidateBuildContext:
    option_type: OptionType
    as_of: date
    spot: float
    current_exp: date
    current_strike: float
    current_mark: float
    current_delta: float | None
    current_theta: float | None
    target_dte: int
    position_contracts: int
    min_open_interest: int
    min_volume: int
    max_spread_pct: float
    max_debit: float | None
    min_credit: float | None
    next_earnings_date: date | None
    earnings_warn_days: int
    earnings_avoid_days: int


@dataclass(frozen=True)
class _CandidateMarketData:
    strike: float
    bid: float | None
    ask: float | None
    mark: float | None
    implied_vol: float | None
    open_interest: int | None
    volume: int | None
    spread: float | None
    spread_pct: float | None
    execution_quality: str | None
    quality_score: float | None
    quality_label: str | None
    last_trade_age_days: int | None
    quality_warnings: list[str]
    spread_ok: bool | None
    delta: float | None
    theta: float | None


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


def _validate_roll_plan_inputs(
    df: pd.DataFrame,
    *,
    horizon_months: int,
    top: int,
) -> None:
    if df is None or df.empty:
        raise ValueError("empty snapshot data")
    if "optionType" not in df.columns or "expiry" not in df.columns or "strike" not in df.columns:
        raise ValueError("snapshot missing required columns (need optionType, expiry, strike)")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be > 0")
    if top <= 0:
        raise ValueError("top must be > 0")


def _prepare_roll_chain(
    df: pd.DataFrame,
    *,
    option_type: OptionType,
    as_of: date,
    min_open_interest: int,
    min_volume: int,
) -> pd.DataFrame:
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

    chain["mark"] = compute_mark_price(chain)
    quality = compute_quote_quality(chain, min_volume=min_volume, min_open_interest=min_open_interest, as_of=as_of)
    chain["_spread"] = quality["spread"]
    chain["_spread_pct"] = quality["spread_pct"]
    chain["_quality_score"] = quality["quality_score"]
    chain["_quality_label"] = quality["quality_label"]
    chain["_quality_warnings"] = quality["quality_warnings"]
    chain["_last_trade_age_days"] = quality["last_trade_age_days"]
    return chain


def _resolve_current_selection(
    chain: pd.DataFrame,
    *,
    position: Position,
    option_type: OptionType,
    as_of: date,
    spot: float,
    warnings: list[str],
) -> _CurrentSelection:
    current_exp = position.expiry
    current_dte = (current_exp - as_of).days
    if current_dte < 0:
        raise ValueError(f"position is expired as-of {as_of.isoformat()} (position expiry {current_exp.isoformat()})")

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

    cur_mark = _as_float(cur_row.get("mark"))
    if cur_mark is None or cur_mark <= 0:
        raise ValueError("missing current contract mark price in snapshot")

    cur_delta, cur_theta = _best_effort_delta_theta(cur_row, option_type=option_type, spot=spot, as_of=as_of)
    contract = _build_current_contract(
        cur_row=cur_row,
        option_type=option_type,
        current_exp=current_exp,
        current_dte=current_dte,
        strike=cur_strike,
        delta=cur_delta,
        theta=cur_theta,
    )
    if contract.quality_warnings:
        for warning in contract.quality_warnings:
            if warning not in warnings:
                warnings.append(warning)
    return _CurrentSelection(contract=contract, strike=cur_strike, mark=cur_mark, delta=cur_delta, theta=cur_theta)


def _build_current_contract(
    *,
    cur_row: dict,
    option_type: OptionType,
    current_exp: date,
    current_dte: int,
    strike: float,
    delta: float | None,
    theta: float | None,
) -> RollContract:
    cur_quality_warnings = cur_row.get("_quality_warnings")
    cur_quality_label = cur_row.get("_quality_label")
    return RollContract(
        contract_symbol=str(cur_row.get("contractSymbol")) if cur_row.get("contractSymbol") is not None else None,
        osi=_osi_from_row(cur_row),
        option_type=option_type,
        expiry=current_exp.isoformat(),
        dte=current_dte,
        strike=strike,
        mark=_as_float(cur_row.get("mark")),
        bid=_as_float(cur_row.get("bid")),
        ask=_as_float(cur_row.get("ask")),
        spread=_as_float(cur_row.get("_spread")),
        spread_pct=_as_float(cur_row.get("_spread_pct")),
        execution_quality=execution_quality(_as_float(cur_row.get("_spread_pct"))),
        quality_score=_as_float(cur_row.get("_quality_score")),
        quality_label=str(cur_quality_label) if cur_quality_label is not None else None,
        last_trade_age_days=_as_int(cur_row.get("_last_trade_age_days")),
        quality_warnings=cast(list[str], cur_quality_warnings) if cur_quality_warnings is not None else [],
        implied_vol=_as_float(cur_row.get("impliedVolatility")),
        open_interest=_as_int(cur_row.get("openInterest")),
        volume=_as_int(cur_row.get("volume")),
        delta=delta,
        theta_per_day=theta,
    )


def _select_candidate_expiries(
    chain: pd.DataFrame,
    *,
    as_of: date,
    target_dte: int,
    current_exp: date,
    warnings: list[str],
) -> tuple[list[date], dict[date, int]]:
    expiries = sorted({d for d in chain["_expiry"].tolist() if isinstance(d, date)})
    by_dte = {expiry: int((expiry - as_of).days) for expiry in expiries}
    near = [expiry for expiry in expiries if abs(by_dte[expiry] - target_dte) <= 90]
    if not near:
        near = sorted(expiries, key=lambda expiry: (abs(by_dte[expiry] - target_dte), expiry))[:6]
        warnings.append("no_expiries_within_target_window_used_nearest")

    preferred = [expiry for expiry in near if expiry > current_exp]
    if not preferred:
        preferred = [expiry for expiry in near if expiry >= current_exp]
        warnings.append("no_later_expiries_in_window")
    preferred = sorted(preferred, key=lambda expiry: (abs(by_dte[expiry] - target_dte), expiry))[:6]
    return preferred, by_dte


def _target_deltas_for_intent(
    *,
    intent: RollIntent,
    option_type: OptionType,
) -> list[float]:
    if intent != "max-upside":
        return []
    return [0.30, 0.40, 0.50] if option_type == "call" else [-0.30, -0.40, -0.50]


def _collect_candidate_rows(
    chain: pd.DataFrame,
    *,
    preferred_expiries: Sequence[date],
    by_dte: Mapping[date, int],
    target_deltas: Sequence[float],
    current_strike: float,
    shape: RollShape,
) -> list[_CandidateRow]:
    candidate_rows: list[_CandidateRow] = []
    used_keys: set[tuple[date, float]] = set()
    for expiry in preferred_expiries:
        exp_df = chain[chain["_expiry"] == expiry].copy()
        if exp_df.empty:
            continue
        base_strike = _pick_base_strike(exp_df, current_strike=current_strike, shape=shape)
        if base_strike is not None:
            _append_candidate_row_for_strike(
                candidate_rows=candidate_rows,
                used_keys=used_keys,
                exp_df=exp_df,
                expiry=expiry,
                by_dte=by_dte,
                strike=float(base_strike),
            )
        for target_delta in target_deltas:
            strike = _pick_by_delta(exp_df, target_delta=target_delta, current_strike=current_strike, shape=shape)
            if strike is None:
                continue
            _append_candidate_row_for_strike(
                candidate_rows=candidate_rows,
                used_keys=used_keys,
                exp_df=exp_df,
                expiry=expiry,
                by_dte=by_dte,
                strike=float(strike),
            )
    return candidate_rows


def _append_candidate_row_for_strike(
    *,
    candidate_rows: list[_CandidateRow],
    used_keys: set[tuple[date, float]],
    exp_df: pd.DataFrame,
    expiry: date,
    by_dte: Mapping[date, int],
    strike: float,
) -> None:
    key = (expiry, strike)
    if key in used_keys:
        return
    used_keys.add(key)
    pick = exp_df.assign(_dist=(exp_df["_strike"] - strike).abs()).sort_values(["_dist", "_strike"], ascending=[True, True])
    candidate_rows.append(_CandidateRow(row=pick.iloc[0].to_dict(), expiry=expiry, dte=int(by_dte[expiry]), strike=strike))


def _candidate_market_data(
    row: Mapping[str, object],
    *,
    context: _CandidateBuildContext,
    fallback_strike: float,
) -> _CandidateMarketData:
    strike = float(row.get("_strike", fallback_strike))
    spread_pct = _as_float(row.get("_spread_pct"))
    spread_ok = None if spread_pct is None else (spread_pct >= 0 and spread_pct <= context.max_spread_pct)
    quality_label = row.get("_quality_label")
    delta, theta = _best_effort_delta_theta(dict(row), option_type=context.option_type, spot=context.spot, as_of=context.as_of)
    quality_warnings = row.get("_quality_warnings")
    return _CandidateMarketData(
        strike=strike,
        bid=_as_float(row.get("bid")),
        ask=_as_float(row.get("ask")),
        mark=_as_float(row.get("mark")),
        implied_vol=_as_float(row.get("impliedVolatility")),
        open_interest=_as_int(row.get("openInterest")),
        volume=_as_int(row.get("volume")),
        spread=_as_float(row.get("_spread")),
        spread_pct=spread_pct,
        execution_quality=execution_quality(spread_pct),
        quality_score=_as_float(row.get("_quality_score")),
        quality_label=str(quality_label) if quality_label is not None else None,
        last_trade_age_days=_as_int(row.get("_last_trade_age_days")),
        quality_warnings=cast(list[str], quality_warnings) if quality_warnings is not None else [],
        spread_ok=spread_ok,
        delta=delta,
        theta=theta,
    )


def _candidate_issues_and_roll_cost(
    market: _CandidateMarketData,
    *,
    context: _CandidateBuildContext,
) -> tuple[list[str], float | None, float | None, bool]:
    issues: list[str] = []
    oi_ok = market.open_interest is not None and market.open_interest >= context.min_open_interest
    vol_ok = market.volume is not None and market.volume >= context.min_volume
    if not (oi_ok or vol_ok):
        issues.append("illiquid_oi_vol")
    if market.spread_ok is False:
        issues.append("wide_or_invalid_spread")
    if market.spread_ok is None:
        issues.append("missing_bid_ask")
    if market.mark is None or market.mark <= 0:
        issues.append("missing_mark")

    roll_debit_per_contract = None
    roll_debit = None
    if market.mark is not None and market.mark > 0 and context.current_mark > 0:
        roll_debit_per_contract = (market.mark - context.current_mark) * 100.0
        roll_debit = roll_debit_per_contract * float(context.position_contracts)
    else:
        issues.append("missing_roll_cost")
    if context.max_debit is not None and (roll_debit is None or roll_debit > context.max_debit):
        issues.append("over_max_debit")
    if context.min_credit is not None:
        credit = None if roll_debit is None else (-roll_debit)
        if credit is None or credit < context.min_credit:
            issues.append("below_min_credit")

    liquidity_ok = ("illiquid_oi_vol" not in issues) and ("wide_or_invalid_spread" not in issues)
    return issues, roll_debit, roll_debit_per_contract, liquidity_ok


def _candidate_rank_score(
    *,
    dte: int,
    roll_debit: float | None,
    delta: float | None,
    liquidity_ok: bool,
    context: _CandidateBuildContext,
) -> float:
    dte_penalty = abs(int(dte) - int(context.target_dte))
    debit_penalty = 0.0 if roll_debit is None else max(0.0, roll_debit) / 1000.0
    delta_target = 0.40 if context.option_type == "call" else -0.40
    delta_penalty = 0.0 if delta is None else abs(delta - delta_target)
    liq_penalty = 10.0 if not liquidity_ok else 0.0
    return -(dte_penalty + (10.0 * debit_penalty) + (50.0 * delta_penalty) + liq_penalty)


def _candidate_rationale(
    *,
    dte: int,
    roll_debit: float | None,
    roll_debit_per_contract: float | None,
    delta: float | None,
    theta: float | None,
    open_interest: int | None,
    volume: int | None,
    spread_pct: float | None,
    context: _CandidateBuildContext,
) -> list[str]:
    dte_penalty = abs(int(dte) - int(context.target_dte))
    rationale: list[str] = [f"Horizon fit: DTE {dte} vs target {context.target_dte} (Δ={dte_penalty}d)."]
    if roll_debit is not None:
        rationale.append(f"Roll cost: {roll_debit:+.0f} total (${roll_debit_per_contract:+.0f}/contract).")
    if delta is not None:
        rationale.append(f"Delta: {delta:+.2f}" + ("" if context.current_delta is None else f" (current {context.current_delta:+.2f})."))
    if theta is not None:
        rationale.append(
            f"Theta/day: {theta:+.4f}" + ("" if context.current_theta is None else f" (current {context.current_theta:+.4f}).")
        )
    rationale.append(
        "Liquidity: "
        + f"OI={open_interest if open_interest is not None else 'n/a'}, Vol={volume if volume is not None else 'n/a'}"
        + ("" if spread_pct is None else f", spread={spread_pct*100.0:.1f}%")
        + "."
    )
    return rationale


def _build_candidate_contract(
    *,
    row: Mapping[str, object],
    market: _CandidateMarketData,
    context: _CandidateBuildContext,
    expiry: date,
    dte: int,
) -> RollContract:
    return RollContract(
        contract_symbol=str(row.get("contractSymbol")) if row.get("contractSymbol") is not None else None,
        osi=_osi_from_row(dict(row)),
        option_type=context.option_type,
        expiry=expiry.isoformat(),
        dte=int(dte),
        strike=market.strike,
        mark=market.mark,
        bid=market.bid,
        ask=market.ask,
        spread=market.spread,
        spread_pct=market.spread_pct,
        execution_quality=market.execution_quality,
        quality_score=market.quality_score,
        quality_label=market.quality_label,
        last_trade_age_days=market.last_trade_age_days,
        quality_warnings=market.quality_warnings,
        implied_vol=market.implied_vol,
        open_interest=market.open_interest,
        volume=market.volume,
        delta=market.delta,
        theta_per_day=market.theta,
    )


def _build_candidate(
    candidate_row: _CandidateRow,
    *,
    context: _CandidateBuildContext,
) -> tuple[RollCandidate | None, list[str]]:
    row = dict(candidate_row.row)
    market = _candidate_market_data(row, context=context, fallback_strike=candidate_row.strike)
    if candidate_row.expiry == context.current_exp and abs(market.strike - context.current_strike) <= 1e-6:
        return None, []

    issues, roll_debit, roll_debit_per_contract, liquidity_ok = _candidate_issues_and_roll_cost(market, context=context)
    score = _candidate_rank_score(
        dte=candidate_row.dte,
        roll_debit=roll_debit,
        delta=market.delta,
        liquidity_ok=liquidity_ok,
        context=context,
    )
    rationale = _candidate_rationale(
        dte=candidate_row.dte,
        roll_debit=roll_debit,
        roll_debit_per_contract=roll_debit_per_contract,
        delta=market.delta,
        theta=market.theta,
        open_interest=market.open_interest,
        volume=market.volume,
        spread_pct=market.spread_pct,
        context=context,
    )
    event_risk = earnings_event_risk(
        today=context.as_of,
        expiry=candidate_row.expiry,
        next_earnings_date=context.next_earnings_date,
        warn_days=context.earnings_warn_days,
        avoid_days=context.earnings_avoid_days,
    )
    risk_warnings = list(event_risk["warnings"])
    if event_risk["exclude"]:
        return None, risk_warnings

    contract = _build_candidate_contract(
        row=row,
        market=market,
        context=context,
        expiry=candidate_row.expiry,
        dte=candidate_row.dte,
    )
    combined_warnings = sorted(set([*risk_warnings, *contract.quality_warnings]))
    return (
        RollCandidate(
            rank_score=float(score),
            contract=contract,
            roll_debit=roll_debit,
            roll_debit_per_contract=roll_debit_per_contract,
            liquidity_ok=bool(liquidity_ok),
            issues=issues,
            rationale=rationale,
            warnings=combined_warnings,
        ),
        [],
    )


def _build_candidates(
    candidate_rows: Sequence[_CandidateRow],
    *,
    context: _CandidateBuildContext,
) -> tuple[list[RollCandidate], set[str]]:
    candidates: list[RollCandidate] = []
    excluded_warnings: set[str] = set()
    for candidate_row in candidate_rows:
        candidate, excluded = _build_candidate(candidate_row, context=context)
        if excluded:
            excluded_warnings.update(excluded)
        if candidate is not None:
            candidates.append(candidate)
    return candidates, excluded_warnings


def _build_candidate_context(
    *,
    option_type: OptionType,
    as_of: date,
    spot: float,
    position: Position,
    current_selection: _CurrentSelection,
    target_dte: int,
    min_open_interest: int,
    min_volume: int,
    max_spread_pct: float,
    max_debit: float | None,
    min_credit: float | None,
    next_earnings_date: date | None,
    earnings_warn_days: int,
    earnings_avoid_days: int,
) -> _CandidateBuildContext:
    return _CandidateBuildContext(
        option_type=option_type,
        as_of=as_of,
        spot=spot,
        current_exp=position.expiry,
        current_strike=current_selection.strike,
        current_mark=current_selection.mark,
        current_delta=current_selection.delta,
        current_theta=current_selection.theta,
        target_dte=target_dte,
        position_contracts=int(position.contracts),
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        max_spread_pct=max_spread_pct,
        max_debit=max_debit,
        min_credit=min_credit,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=earnings_warn_days,
        earnings_avoid_days=earnings_avoid_days,
    )


def _filter_candidates(
    candidates: Sequence[RollCandidate],
    *,
    max_debit: float | None,
    min_credit: float | None,
    include_bad_quotes: bool,
) -> list[RollCandidate]:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.liquidity_ok and ("missing_mark" not in candidate.issues) and ("missing_roll_cost" not in candidate.issues)
    ]
    if max_debit is not None:
        filtered = [candidate for candidate in filtered if "over_max_debit" not in candidate.issues]
    if min_credit is not None:
        filtered = [candidate for candidate in filtered if "below_min_credit" not in candidate.issues]
    if not include_bad_quotes:
        filtered = [candidate for candidate in filtered if candidate.contract.quality_label != "bad"]
    return filtered


def _finalize_filtered_candidates(
    *,
    filtered: list[RollCandidate],
    candidates: Sequence[RollCandidate],
    excluded_warnings: Sequence[str],
    warnings: list[str],
) -> list[RollCandidate]:
    if not filtered and candidates:
        warnings.append("no_candidates_passed_gates_showing_best_effort")
        filtered = list(candidates)
    if excluded_warnings:
        warnings.extend(sorted(excluded_warnings))
    filtered.sort(key=lambda candidate: (-candidate.rank_score, candidate.contract.expiry, candidate.contract.strike))
    return filtered


def _compute_roll_candidates(
    df: pd.DataFrame,
    *,
    as_of: date,
    spot: float,
    position: Position,
    intent: RollIntent,
    horizon_months: int,
    shape: RollShape,
    min_open_interest: int,
    min_volume: int,
    max_spread_pct: float,
    max_debit: float | None,
    min_credit: float | None,
    next_earnings_date: date | None,
    earnings_warn_days: int,
    earnings_avoid_days: int,
    warnings: list[str],
) -> tuple[int, _CurrentSelection, list[RollCandidate], set[str]]:
    option_type: OptionType = position.option_type
    target_dte = _months_to_target_dte(horizon_months)
    chain = _prepare_roll_chain(
        df,
        option_type=option_type,
        as_of=as_of,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
    )
    current_selection = _resolve_current_selection(
        chain,
        position=position,
        option_type=option_type,
        as_of=as_of,
        spot=spot,
        warnings=warnings,
    )
    preferred_expiries, by_dte = _select_candidate_expiries(
        chain,
        as_of=as_of,
        target_dte=target_dte,
        current_exp=position.expiry,
        warnings=warnings,
    )
    candidate_rows = _collect_candidate_rows(
        chain,
        preferred_expiries=preferred_expiries,
        by_dte=by_dte,
        target_deltas=_target_deltas_for_intent(intent=intent, option_type=option_type),
        current_strike=current_selection.strike,
        shape=shape,
    )
    context = _build_candidate_context(
        option_type=option_type,
        as_of=as_of,
        spot=spot,
        position=position,
        current_selection=current_selection,
        target_dte=target_dte,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        max_spread_pct=max_spread_pct,
        max_debit=max_debit,
        min_credit=min_credit,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=earnings_warn_days,
        earnings_avoid_days=earnings_avoid_days,
    )
    candidates, excluded_warnings = _build_candidates(candidate_rows, context=context)
    return target_dte, current_selection, candidates, excluded_warnings


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
    _validate_roll_plan_inputs(df, horizon_months=horizon_months, top=top)
    target_dte, current_selection, candidates, excluded_warnings = _compute_roll_candidates(
        df,
        as_of=as_of,
        spot=spot,
        position=position,
        intent=intent,
        horizon_months=horizon_months,
        shape=shape,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        max_spread_pct=max_spread_pct,
        max_debit=max_debit,
        min_credit=min_credit,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=earnings_warn_days,
        earnings_avoid_days=earnings_avoid_days,
        warnings=warnings,
    )
    filtered = _filter_candidates(
        candidates,
        max_debit=max_debit,
        min_credit=min_credit,
        include_bad_quotes=include_bad_quotes,
    )
    filtered = _finalize_filtered_candidates(
        filtered=filtered,
        candidates=candidates,
        excluded_warnings=tuple(excluded_warnings),
        warnings=warnings,
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
        current=current_selection.contract,
        candidates=filtered[:top],
        warnings=warnings,
    )
