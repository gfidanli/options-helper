from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

from options_helper.analysis.greeks import black_scholes_greeks, black_scholes_price
from options_helper.models import OptionType
from options_helper.schemas.research_metrics_contracts import SCENARIO_GRID_FIELDS, SCENARIO_SUMMARY_FIELDS


DEFAULT_SPOT_MOVES_PCT: tuple[float, ...] = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)
DEFAULT_IV_MOVES_PP: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)
DEFAULT_DAYS_FORWARD: tuple[int, ...] = (0, 7, 14, 30)


@dataclass(frozen=True)
class PositionScenarioResult:
    summary: dict[str, object]
    grid: list[dict[str, object]]


@dataclass(frozen=True)
class _ScenarioContext:
    strike_value: float | None
    spot_value: float | None
    mark_value: float | None
    basis_value: float | None
    iv_value: float | None
    contracts_value: int
    signed_contracts: int
    dte_days: int
    intrinsic: float | None
    extrinsic: float | None


def compute_position_scenarios(
    *,
    symbol: str,
    as_of: date,
    contract_symbol: str,
    option_type: OptionType,
    side: str,
    contracts: int,
    spot: float | None,
    strike: float,
    expiry: date,
    mark: float | None,
    iv: float | None,
    basis: float | None,
    r: float = 0.0,
    spot_moves_pct: Sequence[float] = DEFAULT_SPOT_MOVES_PCT,
    iv_moves_pp: Sequence[float] = DEFAULT_IV_MOVES_PP,
    days_forward: Sequence[int] = DEFAULT_DAYS_FORWARD,
    iv_min: float = 1e-4,
) -> PositionScenarioResult:
    context, warnings = _build_scenario_context(
        as_of=as_of,
        expiry=expiry,
        side=side,
        contracts=contracts,
        option_type=option_type,
        strike=strike,
        spot=spot,
        mark=mark,
        iv=iv,
        basis=basis,
        iv_min=iv_min,
    )
    theta_burn_dollars_day, theta_burn_pct_premium_day = _compute_theta_burn_metrics(
        context=context,
        option_type=option_type,
        r=r,
    )
    grid = _build_scenario_grid(
        symbol=symbol,
        as_of=as_of,
        contract_symbol=contract_symbol,
        option_type=option_type,
        context=context,
        r=r,
        spot_moves_pct=spot_moves_pct,
        iv_moves_pp=iv_moves_pp,
        days_forward=days_forward,
    )
    if context.dte_days < 0:
        warnings.append("past_expiry")

    summary = _project_fields(
        SCENARIO_SUMMARY_FIELDS,
        {
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "contract_symbol": contract_symbol,
            "option_type": option_type,
            "side": side,
            "contracts": context.contracts_value,
            "spot": context.spot_value,
            "strike": context.strike_value,
            "expiry": expiry.isoformat(),
            "mark": context.mark_value,
            "iv": context.iv_value,
            "intrinsic": context.intrinsic,
            "extrinsic": context.extrinsic,
            "theta_burn_dollars_day": theta_burn_dollars_day,
            "theta_burn_pct_premium_day": theta_burn_pct_premium_day,
            "warnings": _dedupe_preserve_order(warnings),
        },
    )

    return PositionScenarioResult(summary=summary, grid=grid)


def _build_scenario_context(
    *,
    as_of: date,
    expiry: date,
    side: str,
    contracts: int,
    option_type: OptionType,
    strike: float,
    spot: float | None,
    mark: float | None,
    iv: float | None,
    basis: float | None,
    iv_min: float,
) -> tuple[_ScenarioContext, list[str]]:
    warnings: list[str] = []
    strike_value = _positive_float(strike)
    if strike_value is None:
        warnings.append("missing_strike")
    spot_value = _positive_float(spot)
    if spot_value is None:
        warnings.append("missing_spot")
    mark_value = _positive_float(mark)
    if mark_value is None:
        warnings.append("missing_mark")
    basis_value = _positive_float(basis)
    iv_value = _coerce_float(iv)
    if iv_value is None or iv_value <= max(0.0, float(iv_min)):
        iv_value = None
        warnings.append("missing_iv")

    intrinsic = (
        _intrinsic(option_type=option_type, spot=spot_value, strike=strike_value)
        if strike_value is not None and spot_value is not None
        else None
    )
    extrinsic = max(0.0, mark_value - intrinsic) if intrinsic is not None and mark_value is not None else None
    contracts_value = _coerce_int(contracts, default=0)
    signed_contracts = contracts_value * _side_sign(side)
    context = _ScenarioContext(
        strike_value=strike_value,
        spot_value=spot_value,
        mark_value=mark_value,
        basis_value=basis_value,
        iv_value=iv_value,
        contracts_value=contracts_value,
        signed_contracts=signed_contracts,
        dte_days=int((expiry - as_of).days),
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    return context, warnings


def _compute_theta_burn_metrics(
    *,
    context: _ScenarioContext,
    option_type: OptionType,
    r: float,
) -> tuple[float | None, float | None]:
    theta_burn_dollars_day: float | None = None
    if context.dte_days == 0:
        theta_burn_dollars_day = 0.0
    elif (
        context.dte_days > 0
        and context.strike_value is not None
        and context.spot_value is not None
        and context.iv_value is not None
    ):
        greeks = black_scholes_greeks(
            option_type=option_type,
            s=context.spot_value,
            k=context.strike_value,
            t_years=context.dte_days / 365.0,
            sigma=context.iv_value,
            r=r,
        )
        if greeks is not None:
            theta_burn_dollars_day = -greeks.theta_per_day * float(context.signed_contracts) * 100.0

    theta_burn_pct_premium_day: float | None = None
    premium_ref = context.mark_value if context.mark_value is not None else context.basis_value
    if theta_burn_dollars_day is not None and premium_ref is not None and context.contracts_value > 0:
        premium_position = premium_ref * float(context.contracts_value) * 100.0
        if premium_position > 0.0:
            theta_burn_pct_premium_day = theta_burn_dollars_day / premium_position
    return theta_burn_dollars_day, theta_burn_pct_premium_day


def _build_scenario_grid(
    *,
    symbol: str,
    as_of: date,
    contract_symbol: str,
    option_type: OptionType,
    context: _ScenarioContext,
    r: float,
    spot_moves_pct: Sequence[float],
    iv_moves_pp: Sequence[float],
    days_forward: Sequence[int],
) -> list[dict[str, object]]:
    if context.dte_days < 0:
        return []
    if context.strike_value is None or context.spot_value is None or context.iv_value is None:
        return []

    spot_axis = _normalize_spot_moves_pct(spot_moves_pct)
    iv_axis = _normalize_iv_moves_pp(iv_moves_pp)
    day_axis = _normalize_days_forward(days_forward)
    pnl_reference = context.mark_value if context.mark_value is not None else context.basis_value
    rows: list[dict[str, object]] = []
    for spot_change_pct in spot_axis:
        scenario_spot = context.spot_value * (1.0 + spot_change_pct)
        for iv_change_pp in iv_axis:
            scenario_iv = context.iv_value + (iv_change_pp / 100.0)
            for forward_days in day_axis:
                rows.append(
                    _scenario_grid_row(
                        symbol=symbol,
                        as_of=as_of,
                        contract_symbol=contract_symbol,
                        option_type=option_type,
                        context=context,
                        r=r,
                        scenario_spot=scenario_spot,
                        spot_change_pct=spot_change_pct,
                        scenario_iv=scenario_iv,
                        iv_change_pp=iv_change_pp,
                        forward_days=forward_days,
                        pnl_reference=pnl_reference,
                    )
                )
    return rows


def _scenario_grid_row(
    *,
    symbol: str,
    as_of: date,
    contract_symbol: str,
    option_type: OptionType,
    context: _ScenarioContext,
    r: float,
    scenario_spot: float,
    spot_change_pct: float,
    scenario_iv: float,
    iv_change_pp: float,
    forward_days: int,
    pnl_reference: float | None,
) -> dict[str, object]:
    days_to_expiry = max(context.dte_days - forward_days, 0)
    theoretical = _scenario_theoretical_price(
        option_type=option_type,
        spot=scenario_spot,
        strike=context.strike_value or 0.0,
        iv=scenario_iv,
        dte_days=days_to_expiry,
        r=r,
    )
    pnl_per_contract = None if (theoretical is None or pnl_reference is None) else theoretical - pnl_reference
    pnl_position = (
        None
        if pnl_per_contract is None
        else pnl_per_contract * float(context.signed_contracts) * 100.0
    )
    return _project_fields(
        SCENARIO_GRID_FIELDS,
        {
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "contract_symbol": contract_symbol,
            "spot_change_pct": spot_change_pct,
            "iv_change_pp": iv_change_pp,
            "days_forward": forward_days,
            "scenario_spot": scenario_spot,
            "scenario_iv": scenario_iv,
            "days_to_expiry": days_to_expiry,
            "theoretical_price": theoretical,
            "pnl_per_contract": pnl_per_contract,
            "pnl_position": pnl_position,
        },
    )


def _scenario_theoretical_price(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    iv: float,
    dte_days: int,
    r: float,
) -> float | None:
    if spot <= 0.0 or strike <= 0.0:
        return None
    if dte_days <= 0:
        return _intrinsic(option_type=option_type, spot=spot, strike=strike)
    if iv <= 0.0:
        return None
    return black_scholes_price(option_type=option_type, s=spot, k=strike, t_years=dte_days / 365.0, sigma=iv, r=r)


def _intrinsic(*, option_type: OptionType, spot: float, strike: float) -> float:
    if option_type == "call":
        return max(0.0, spot - strike)
    return max(0.0, strike - spot)


def _positive_float(value: object) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None or parsed <= 0.0:
        return None
    return parsed


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _coerce_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _side_sign(side: str) -> int:
    return -1 if str(side).strip().lower() == "short" else 1


def _normalize_spot_moves_pct(values: Sequence[float]) -> tuple[float, ...]:
    out: set[float] = {0.0}
    for raw in values:
        value = _coerce_float(raw)
        if value is None:
            continue
        if abs(value) > 1.0:
            value = value / 100.0
        out.add(round(value, 10))
    return tuple(sorted(out))


def _normalize_iv_moves_pp(values: Sequence[float]) -> tuple[float, ...]:
    out: set[float] = {0.0}
    for raw in values:
        value = _coerce_float(raw)
        if value is None:
            continue
        if 0.0 < abs(value) <= 1.0:
            value = value * 100.0
        out.add(round(value, 10))
    return tuple(sorted(out))


def _normalize_days_forward(values: Sequence[int]) -> tuple[int, ...]:
    out: set[int] = {0}
    for raw in values:
        coerced = _coerce_int(raw, default=-1)
        if coerced < 0:
            continue
        out.add(coerced)
    return tuple(sorted(out))


def _project_fields(fields: tuple[str, ...], data: dict[str, object]) -> dict[str, object]:
    return {field: data.get(field) for field in fields}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "DEFAULT_DAYS_FORWARD",
    "DEFAULT_IV_MOVES_PP",
    "DEFAULT_SPOT_MOVES_PCT",
    "PositionScenarioResult",
    "compute_position_scenarios",
]
