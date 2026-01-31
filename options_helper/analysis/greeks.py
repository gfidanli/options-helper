from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import erf, exp, log, pi, sqrt

import pandas as pd

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


def add_black_scholes_greeks_to_chain(
    df: pd.DataFrame,
    *,
    spot: float,
    expiry: date,
    as_of: date | None = None,
    r: float = 0.0,
    option_type_col: str = "optionType",
    strike_col: str = "strike",
    iv_col: str = "impliedVolatility",
    prefix: str = "bs_",
) -> pd.DataFrame:
    """
    Add best-effort Black-Scholes Greeks to an options chain DataFrame.

    Greeks are computed from:
    - spot (s)
    - strike (k)
    - time to expiry in years (t_years)
    - implied volatility (sigma)
    - risk-free rate (r)

    This is model-based and should be treated as an approximation.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    as_of = as_of or date.today()
    dte = (expiry - as_of).days
    t_years = dte / 365.0 if dte > 0 else None

    cols = {
        f"{prefix}price": [],
        f"{prefix}delta": [],
        f"{prefix}gamma": [],
        f"{prefix}theta_per_day": [],
        f"{prefix}vega": [],
    }

    # Always add the columns (stable schema), even if we can't compute.
    if (
        t_years is None
        or t_years <= 0
        or option_type_col not in out.columns
        or strike_col not in out.columns
        or iv_col not in out.columns
    ):
        for key in cols:
            out[key] = None
        return out

    def _as_float(val) -> float | None:
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
                return None
            return float(val)
        except Exception:  # noqa: BLE001
            return None

    for opt_type, strike, sigma in zip(
        out[option_type_col].tolist(),
        out[strike_col].tolist(),
        out[iv_col].tolist(),
    ):
        opt = str(opt_type).lower().strip() if opt_type is not None else ""
        k = _as_float(strike)
        iv = _as_float(sigma)
        if opt not in {"call", "put"} or k is None or iv is None:
            for key in cols:
                cols[key].append(None)
            continue

        g = black_scholes_greeks(option_type=opt, s=spot, k=k, t_years=t_years, sigma=iv, r=r)
        if g is None:
            for key in cols:
                cols[key].append(None)
            continue

        cols[f"{prefix}price"].append(g.price)
        cols[f"{prefix}delta"].append(g.delta)
        cols[f"{prefix}gamma"].append(g.gamma)
        cols[f"{prefix}theta_per_day"].append(g.theta_per_day)
        cols[f"{prefix}vega"].append(g.vega)

    for key, values in cols.items():
        out[key] = values

    return out
