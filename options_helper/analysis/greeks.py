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


@dataclass(frozen=True)
class _ChainGreeksConfig:
    spot: float
    expiry: date
    as_of: date | None
    r: float
    option_type_col: str
    strike_col: str
    iv_col: str
    prefix: str
    iv_placeholder_threshold: float
    fill_iv_from_mark: bool


def black_scholes_price(
    *,
    option_type: OptionType,
    s: float,
    k: float,
    t_years: float,
    sigma: float,
    r: float = 0.0,
) -> float | None:
    if s <= 0.0 or k <= 0.0 or t_years <= 0.0 or sigma <= 0.0:
        return None

    vol_sqrt_t = sigma * sqrt(t_years)
    d1 = (log(s / k) + (r + 0.5 * sigma * sigma) * t_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    disc = exp(-r * t_years)

    if option_type == "call":
        price = s * _norm_cdf(d1) - k * disc * _norm_cdf(d2)
    else:
        price = k * disc * _norm_cdf(-d2) - s * _norm_cdf(-d1)
    return float(price)


def implied_volatility_from_price(
    *,
    option_type: OptionType,
    s: float,
    k: float,
    t_years: float,
    price: float,
    r: float = 0.0,
    sigma_min: float = 1e-6,
    sigma_max: float = 10.0,
    max_iter: int = 40,
    tol: float = 1e-4,
) -> float | None:
    """
    Best-effort implied volatility via bisection on Black-Scholes price.

    Returns None when inputs are invalid or violate basic no-arbitrage bounds.
    """
    if option_type not in {"call", "put"}:
        return None
    if s <= 0.0 or k <= 0.0 or t_years <= 0.0 or price <= 0.0:
        return None
    if sigma_min <= 0.0 or sigma_max <= sigma_min:
        return None

    disc = exp(-r * t_years)
    if option_type == "call":
        lower = max(0.0, s - k * disc)
        upper = s
    else:
        lower = max(0.0, k * disc - s)
        upper = k * disc

    if price < lower - 1e-9:
        return None
    if price > upper + 1e-9:
        return None
    if price <= lower + 1e-12:
        return sigma_min

    lo = sigma_min
    hi = 1.0
    hi_price = black_scholes_price(option_type=option_type, s=s, k=k, t_years=t_years, sigma=hi, r=r)
    if hi_price is None:
        return None

    while hi_price < price and hi < sigma_max:
        hi = min(sigma_max, hi * 2.0)
        hi_price = black_scholes_price(option_type=option_type, s=s, k=k, t_years=t_years, sigma=hi, r=r)
        if hi_price is None:
            return None

    if hi_price < price:
        if abs(price - upper) <= 1e-6:
            return sigma_max
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mid_price = black_scholes_price(option_type=option_type, s=s, k=k, t_years=t_years, sigma=mid, r=r)
        if mid_price is None:
            return None
        err = mid_price - price
        if abs(err) <= tol:
            return float(mid)
        if err < 0:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2.0)


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


def _chain_as_float(val: object) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _chain_source(val: object) -> str | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:  # noqa: BLE001
        pass
    raw = str(val).strip()
    return raw or None


def _chain_mark_price(bid: float | None, ask: float | None, last: float | None) -> tuple[float | None, str]:
    if bid is not None and bid > 0 and ask is not None and ask > 0:
        return (bid + ask) / 2.0, "mid"
    if last is not None and last > 0:
        return last, "last"
    if ask is not None and ask > 0:
        return ask, "ask"
    if bid is not None and bid > 0:
        return bid, "bid"
    return None, "none"


def _empty_chain_greeks_columns(prefix: str) -> dict[str, list[float | None]]:
    return {
        f"{prefix}price": [],
        f"{prefix}delta": [],
        f"{prefix}gamma": [],
        f"{prefix}theta_per_day": [],
        f"{prefix}vega": [],
    }


def _append_chain_greek_values(
    cols: dict[str, list[float | None]],
    *,
    prefix: str,
    greeks: Greeks | None,
) -> None:
    if greeks is None:
        for key in cols:
            cols[key].append(None)
        return
    cols[f"{prefix}price"].append(greeks.price)
    cols[f"{prefix}delta"].append(greeks.delta)
    cols[f"{prefix}gamma"].append(greeks.gamma)
    cols[f"{prefix}theta_per_day"].append(greeks.theta_per_day)
    cols[f"{prefix}vega"].append(greeks.vega)


def _compute_chain_row_greeks(
    *,
    option_type: object,
    strike: object,
    sigma: object,
    bid_value: object,
    ask_value: object,
    last_value: object,
    volume_value: object,
    source_in: object,
    spot: float,
    t_years: float,
    r: float,
    iv_placeholder_threshold: float,
    fill_iv_from_mark: bool,
) -> tuple[float | None, str | None, Greeks | None]:
    opt = str(option_type).lower().strip() if option_type is not None else ""
    k = _chain_as_float(strike)
    iv = _chain_as_float(sigma)
    source = _chain_source(source_in)
    if iv is not None and iv <= iv_placeholder_threshold:
        iv = None

    inferred_iv = False
    if iv is None and fill_iv_from_mark:
        bid = _chain_as_float(bid_value)
        ask = _chain_as_float(ask_value)
        last = _chain_as_float(last_value)
        volume = _chain_as_float(volume_value)
        mark, mark_src = _chain_mark_price(bid, ask, last)
        ok_for_iv = mark is not None and mark > 0 and (
            mark_src == "mid" or (mark_src == "last" and volume is not None and volume > 0)
        )
        if ok_for_iv and opt in {"call", "put"} and k is not None:
            iv_calc = implied_volatility_from_price(
                option_type=opt,
                s=spot,
                k=k,
                t_years=t_years,
                price=float(mark),
                r=r,
            )
            if iv_calc is not None and iv_calc > iv_placeholder_threshold:
                iv = iv_calc
                inferred_iv = True

    if opt not in {"call", "put"} or k is None or iv is None:
        final_source = "bs_inferred" if inferred_iv else ("missing" if iv is None else source)
        return iv, final_source, None

    greeks = black_scholes_greeks(option_type=opt, s=spot, k=k, t_years=t_years, sigma=iv, r=r)
    if greeks is None:
        final_source = "bs_inferred" if inferred_iv else ("missing" if iv is None else source)
        return iv, final_source, None
    final_source = "bs_inferred" if inferred_iv else ("missing" if iv is None else source)
    return iv, final_source, greeks


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
    iv_placeholder_threshold: float = 1e-4,
    fill_iv_from_mark: bool = True,
) -> pd.DataFrame:
    config = _ChainGreeksConfig(
        spot=spot,
        expiry=expiry,
        as_of=as_of,
        r=r,
        option_type_col=option_type_col,
        strike_col=strike_col,
        iv_col=iv_col,
        prefix=prefix,
        iv_placeholder_threshold=iv_placeholder_threshold,
        fill_iv_from_mark=fill_iv_from_mark,
    )
    return _add_black_scholes_greeks_to_chain_impl(df=df, config=config)


def _add_black_scholes_greeks_to_chain_impl(
    *,
    df: pd.DataFrame,
    config: _ChainGreeksConfig,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    as_of = config.as_of or date.today()
    dte = (config.expiry - as_of).days
    t_years = dte / 365.0 if dte > 0 else None
    cols = _empty_chain_greeks_columns(config.prefix)
    if (
        t_years is None
        or t_years <= 0
        or config.option_type_col not in out.columns
        or config.strike_col not in out.columns
        or config.iv_col not in out.columns
    ):
        for key in cols:
            out[key] = None
        return out

    bids = out["bid"].tolist() if "bid" in out.columns else [None] * len(out)
    asks = out["ask"].tolist() if "ask" in out.columns else [None] * len(out)
    lasts = out["lastPrice"].tolist() if "lastPrice" in out.columns else [None] * len(out)
    volumes = out["volume"].tolist() if "volume" in out.columns else [None] * len(out)
    iv_out: list[float | None] = []
    iv_source_out: list[str | None] = []
    iv_source_in = out["iv_source"].tolist() if "iv_source" in out.columns else [None] * len(out)
    for opt_type, strike, sigma, bid_v, ask_v, last_v, vol_v, source_in in zip(
        out[config.option_type_col].tolist(),
        out[config.strike_col].tolist(),
        out[config.iv_col].tolist(),
        bids,
        asks,
        lasts,
        volumes,
        iv_source_in,
    ):
        iv, source, greeks = _compute_chain_row_greeks(
            option_type=opt_type,
            strike=strike,
            sigma=sigma,
            bid_value=bid_v,
            ask_value=ask_v,
            last_value=last_v,
            volume_value=vol_v,
            source_in=source_in,
            spot=config.spot,
            t_years=t_years,
            r=config.r,
            iv_placeholder_threshold=config.iv_placeholder_threshold,
            fill_iv_from_mark=config.fill_iv_from_mark,
        )
        iv_out.append(iv)
        iv_source_out.append(source)
        _append_chain_greek_values(cols, prefix=config.prefix, greeks=greeks)
    for key, values in cols.items():
        out[key] = values
    out[config.iv_col] = iv_out
    out["iv_source"] = iv_source_out
    return out
