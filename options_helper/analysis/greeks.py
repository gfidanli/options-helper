from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

from options_helper.models import OptionType


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta_per_day: float
    vega: float


def black_scholes_greeks(
    *,
    option_type: OptionType,
    s: float,
    k: float,
    t_years: float,
    sigma: float,
    r: float = 0.0,
) -> Greeks | None:
    if s <= 0.0 or k <= 0.0 or t_years <= 0.0 or sigma <= 0.0:
        return None

    vol_sqrt_t = sigma * sqrt(t_years)
    d1 = (log(s / k) + (r + 0.5 * sigma * sigma) * t_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    nmd1 = _norm_cdf(-d1)
    nmd2 = _norm_cdf(-d2)
    pdf_d1 = _norm_pdf(d1)

    disc = exp(-r * t_years)

    if option_type == "call":
        price = s * nd1 - k * disc * nd2
        delta = nd1
        theta = -(s * pdf_d1 * sigma) / (2.0 * sqrt(t_years)) - (r * k * disc * nd2)
    else:
        price = k * disc * nmd2 - s * nmd1
        delta = nd1 - 1.0
        theta = -(s * pdf_d1 * sigma) / (2.0 * sqrt(t_years)) + (r * k * disc * nmd2)

    gamma = pdf_d1 / (s * vol_sqrt_t)
    vega = s * pdf_d1 * sqrt(t_years)
    theta_per_day = theta / 365.0

    return Greeks(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        theta_per_day=float(theta_per_day),
        vega=float(vega),
    )

