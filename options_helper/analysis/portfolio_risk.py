from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from options_helper.analysis.advice import PositionMetrics
from options_helper.analysis.greeks import black_scholes_greeks, black_scholes_price
from options_helper.models import OptionType


@dataclass(frozen=True)
class PositionExposure:
    id: str
    symbol: str
    option_type: OptionType
    expiry: date
    strike: float
    contracts: int
    spot: float | None
    dte: int | None
    implied_vol: float | None
    delta: float | None
    theta_per_day: float | None
    vega: float | None
    delta_shares: float | None
    theta_dollars_per_day: float | None
    vega_dollars_per_iv: float | None
    base_price: float | None
    greek_source: str
    price_source: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioExposure:
    as_of: date | None
    positions: list[PositionExposure]
    total_delta_shares: float | None
    total_theta_dollars_per_day: float | None
    total_vega_dollars_per_iv: float | None
    risk_free_rate: float
    missing_greeks: int
    warnings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class StressScenario:
    name: str
    spot_pct: float = 0.0
    vol_pp: float = 0.0
    days: int = 0


@dataclass(frozen=True)
class StressResult:
    name: str
    spot_pct: float
    vol_pp: float
    days: int
    pnl: float | None
    pnl_pct: float | None
    warnings: tuple[str, ...] = ()


def compute_portfolio_exposure(
    metrics_list: Iterable[PositionMetrics],
    *,
    r: float = 0.0,
    iv_min: float = 1e-4,
) -> PortfolioExposure:
    positions: list[PositionExposure] = []
    delta_vals: list[float] = []
    theta_vals: list[float] = []
    vega_vals: list[float] = []
    as_of_vals: set[date] = set()
    warnings: list[str] = []
    missing_greeks = 0

    for metrics in metrics_list:
        pos = metrics.position
        if metrics.as_of is not None:
            as_of_vals.add(metrics.as_of)

        per_warnings: list[str] = []
        spot = _clean_float(metrics.underlying_price)
        dte = _clean_dte(metrics.dte)
        iv = _clean_float(metrics.implied_vol)

        if iv is not None and iv <= iv_min:
            iv = None
            per_warnings.append("iv_placeholder")

        delta = _clean_float(metrics.delta)
        theta_per_day = _clean_float(metrics.theta_per_day)
        vega = None
        greek_source = "provided" if (delta is not None or theta_per_day is not None) else "missing"

        if spot is not None and iv is not None and dte is not None and dte > 0:
            bs = black_scholes_greeks(
                option_type=pos.option_type,
                s=spot,
                k=pos.strike,
                t_years=dte / 365.0,
                sigma=iv,
                r=r,
            )
            if bs is not None:
                vega = bs.vega
                if delta is None:
                    delta = bs.delta
                if theta_per_day is None:
                    theta_per_day = bs.theta_per_day
                greek_source = "bs"

        if delta is None and theta_per_day is None and vega is None:
            missing_greeks += 1
            per_warnings.append("missing_greeks")
            greek_source = "missing"

        delta_shares = _scale_contracts(delta, pos.contracts)
        theta_dollars = _scale_contracts(theta_per_day, pos.contracts)
        vega_dollars = _scale_contracts(vega, pos.contracts)

        if delta_shares is not None:
            delta_vals.append(delta_shares)
        if theta_dollars is not None:
            theta_vals.append(theta_dollars)
        if vega_dollars is not None:
            vega_vals.append(vega_dollars)

        base_price, price_source = _base_price(
            option_type=pos.option_type,
            spot=spot,
            strike=pos.strike,
            dte=dte,
            iv=iv,
            r=r,
            fallback=metrics.mark,
        )
        if base_price is None:
            per_warnings.append("missing_price")

        positions.append(
            PositionExposure(
                id=pos.id,
                symbol=pos.symbol,
                option_type=pos.option_type,
                expiry=pos.expiry,
                strike=pos.strike,
                contracts=pos.contracts,
                spot=spot,
                dte=dte,
                implied_vol=iv,
                delta=delta,
                theta_per_day=theta_per_day,
                vega=vega,
                delta_shares=delta_shares,
                theta_dollars_per_day=theta_dollars,
                vega_dollars_per_iv=vega_dollars,
                base_price=base_price,
                greek_source=greek_source,
                price_source=price_source,
                warnings=tuple(per_warnings),
            )
        )

    total_delta = _sum_or_none(delta_vals)
    total_theta = _sum_or_none(theta_vals)
    total_vega = _sum_or_none(vega_vals)

    as_of = None
    if len(as_of_vals) == 1:
        as_of = next(iter(as_of_vals))
    elif len(as_of_vals) > 1:
        warnings.append("mixed_as_of_dates")

    if missing_greeks > 0:
        warnings.append(f"missing_greeks={missing_greeks}")

    assumptions = (
        "Delta in shares eq (delta * contracts * 100).",
        "Theta in $/day (theta_per_day * contracts * 100).",
        "Vega in $ per 1.00 IV (vega * contracts * 100).",
        f"Black-Scholes used when greeks missing (r={r:.2%}).",
    )
    if iv_min > 0:
        assumptions += (f"IV <= {iv_min:g} treated as missing.",)

    return PortfolioExposure(
        as_of=as_of,
        positions=positions,
        total_delta_shares=total_delta,
        total_theta_dollars_per_day=total_theta,
        total_vega_dollars_per_iv=total_vega,
        risk_free_rate=float(r),
        missing_greeks=missing_greeks,
        warnings=tuple(warnings),
        assumptions=assumptions,
    )


def run_stress(exposure: PortfolioExposure, scenarios: Iterable[StressScenario]) -> list[StressResult]:
    results: list[StressResult] = []
    for scenario in scenarios:
        base_total = 0.0
        stressed_total = 0.0
        missing = 0
        priced = 0

        for pos in exposure.positions:
            base_price = _stress_price(
                option_type=pos.option_type,
                spot=pos.spot,
                strike=pos.strike,
                dte=pos.dte,
                iv=pos.implied_vol,
                r=exposure.risk_free_rate,
            )
            if base_price is None:
                missing += 1
                continue

            stressed_price = _stress_price(
                option_type=pos.option_type,
                spot=None if pos.spot is None else pos.spot * (1.0 + scenario.spot_pct),
                strike=pos.strike,
                dte=None if pos.dte is None else max(0, pos.dte - scenario.days),
                iv=None if pos.implied_vol is None else max(0.0, pos.implied_vol + scenario.vol_pp),
                r=exposure.risk_free_rate,
            )
            if stressed_price is None:
                missing += 1
                continue

            base_val = base_price * pos.contracts * 100.0
            stressed_val = stressed_price * pos.contracts * 100.0
            base_total += base_val
            stressed_total += stressed_val
            priced += 1

        pnl = None
        pnl_pct = None
        if priced > 0:
            pnl = stressed_total - base_total
            if base_total > 0:
                pnl_pct = pnl / base_total

        warnings: list[str] = []
        if missing > 0:
            warnings.append(f"stress_missing_inputs={missing}")

        results.append(
            StressResult(
                name=scenario.name,
                spot_pct=scenario.spot_pct,
                vol_pp=scenario.vol_pp,
                days=scenario.days,
                pnl=pnl,
                pnl_pct=pnl_pct,
                warnings=tuple(warnings),
            )
        )

    return results


def _clean_float(val: float | None) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _clean_dte(val: int | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val) if int(val) >= 0 else 0
    except Exception:  # noqa: BLE001
        return None


def _sum_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values))


def _scale_contracts(val: float | None, contracts: int) -> float | None:
    if val is None:
        return None
    return float(val) * float(contracts) * 100.0


def _intrinsic(option_type: OptionType, spot: float, strike: float) -> float:
    if option_type == "call":
        return max(0.0, spot - strike)
    return max(0.0, strike - spot)


def _base_price(
    *,
    option_type: OptionType,
    spot: float | None,
    strike: float,
    dte: int | None,
    iv: float | None,
    r: float,
    fallback: float | None,
) -> tuple[float | None, str]:
    if spot is not None and dte is not None:
        if dte <= 0:
            return _intrinsic(option_type, spot, strike), "intrinsic"
        if iv is not None and iv > 0:
            price = black_scholes_price(option_type=option_type, s=spot, k=strike, t_years=dte / 365.0, sigma=iv, r=r)
            if price is not None:
                return price, "bs"
    if fallback is not None and fallback > 0:
        return float(fallback), "mark"
    return None, "missing"


def _stress_price(
    *,
    option_type: OptionType,
    spot: float | None,
    strike: float,
    dte: int | None,
    iv: float | None,
    r: float,
) -> float | None:
    if spot is None or dte is None:
        return None
    if dte <= 0:
        return _intrinsic(option_type, spot, strike)
    if iv is None or iv <= 0:
        return None
    return black_scholes_price(option_type=option_type, s=spot, k=strike, t_years=dte / 365.0, sigma=iv, r=r)
