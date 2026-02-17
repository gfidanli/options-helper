from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


def _col_as_float(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def compute_mark_price(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort mark price:
    - mid if bid/ask are both > 0
    - else lastPrice if > 0
    - else ask if > 0
    - else bid if > 0
    """
    bid = _col_as_float(df, "bid")
    ask = _col_as_float(df, "ask")
    last = _col_as_float(df, "lastPrice")

    mark = pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    mid_mask = (bid > 0) & (ask > 0)
    mark = mark.where(~mid_mask, (bid + ask) / 2.0)

    last_mask = last > 0
    mark = mark.fillna(last.where(last_mask))

    ask_mask = ask > 0
    mark = mark.fillna(ask.where(ask_mask))

    bid_mask = bid > 0
    mark = mark.fillna(bid.where(bid_mask))

    return mark


def compute_spread(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort bid/ask spread (ask - bid) when bid/ask are both > 0.
    Returns NaN when bid/ask are missing or non-positive.
    """
    bid = _col_as_float(df, "bid")
    ask = _col_as_float(df, "ask")
    spread = ask - bid
    valid_mask = (bid > 0) & (ask > 0)
    return spread.where(valid_mask)


def compute_spread_pct(df: pd.DataFrame) -> pd.Series:
    """
    Spread as a fraction of mid price when bid/ask are both > 0 and mid > 0.
    Returns NaN when bid/ask are missing or non-positive.
    """
    bid = _col_as_float(df, "bid")
    ask = _col_as_float(df, "ask")
    mid = (ask + bid) / 2.0
    spread_pct = (ask - bid) / mid
    valid_mask = (bid > 0) & (ask > 0) & (mid > 0)
    return spread_pct.where(valid_mask)


def execution_quality(spread_pct: float | None) -> str:
    if spread_pct is None:
        return "unknown"
    try:
        if pd.isna(spread_pct):
            return "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"

    val = float(spread_pct)
    if val < 0:
        return "bad"
    if val <= 0.15:
        return "good"
    if val <= 0.35:
        return "ok"
    return "bad"


def _safe_sum(series: pd.Series) -> float:
    val = series.dropna()
    if val.empty:
        return 0.0
    return float(val.sum())


def _safe_ratio(numer: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _is_monthly_expiry(expiry: date) -> bool:
    # Heuristic: standard monthly options are the 3rd Friday (day 15–21).
    return expiry.weekday() == 4 and 15 <= expiry.day <= 21


def select_expiries(
    expiries: list[date],
    *,
    mode: Literal["near", "monthly", "all"] = "near",
    include: list[date] | None = None,
    near_n: int = 4,
) -> list[date]:
    if include:
        include_set = set(include)
        return sorted([e for e in expiries if e in include_set])

    if mode == "all":
        return list(expiries)
    if mode == "monthly":
        return [e for e in expiries if _is_monthly_expiry(e)]

    # near
    return list(expiries[:near_n])


class ChainTotals(BaseModel):
    calls_oi: float = 0.0
    puts_oi: float = 0.0
    pc_oi_ratio: float | None = None
    calls_volume: float = 0.0
    puts_volume: float = 0.0
    pc_volume_ratio: float | None = None
    calls_volume_notional: float = 0.0
    puts_volume_notional: float = 0.0


class WallLevel(BaseModel):
    strike: float
    open_interest: float = Field(ge=0.0)


class Walls(BaseModel):
    calls: list[WallLevel] = Field(default_factory=list)
    puts: list[WallLevel] = Field(default_factory=list)


class WallsByExpiry(BaseModel):
    expiry: str
    walls: Walls


class ExpiryStats(BaseModel):
    expiry: str
    calls_oi: float = 0.0
    puts_oi: float = 0.0
    pc_oi_ratio: float | None = None
    calls_volume: float = 0.0
    puts_volume: float = 0.0
    pc_volume_ratio: float | None = None
    atm_strike: float | None = None
    call_mark_atm: float | None = None
    put_mark_atm: float | None = None
    straddle_mark_atm: float | None = None
    expected_move: float | None = None
    expected_move_pct: float | None = None
    atm_iv: float | None = None
    skew_25d_pp: float | None = None


class GammaLevel(BaseModel):
    strike: float
    gamma_1pct: float


class GammaSummary(BaseModel):
    peak_strike: float | None = None
    top: list[GammaLevel] = Field(default_factory=list)


class ChainReport(BaseModel):
    schema_version: int = 1
    symbol: str
    as_of: str
    spot: float
    available_expiries: list[str] = Field(default_factory=list)
    included_expiries: list[str] = Field(default_factory=list)
    totals: ChainTotals
    expiries: list[ExpiryStats] = Field(default_factory=list)
    walls_overall: Walls
    walls_by_expiry: list[WallsByExpiry] = Field(default_factory=list)
    gamma: GammaSummary
    warnings: list[str] = Field(default_factory=list)


def _compute_walls(
    df: pd.DataFrame,
    *,
    spot: float,
    top: int,
) -> Walls:
    if df.empty or "optionType" not in df.columns or "strike" not in df.columns:
        return Walls()

    out: dict[str, list[WallLevel]] = {"call": [], "put": []}
    for opt_type, strike_filter in [
        ("call", lambda s: s >= spot),
        ("put", lambda s: s <= spot),
    ]:
        sub = df[df["optionType"] == opt_type]
        if sub.empty:
            continue

        strike = pd.to_numeric(sub["strike"], errors="coerce")
        oi = _col_as_float(sub, "openInterest")
        sub = sub.assign(_strike=strike, _oi=oi)
        sub = sub.dropna(subset=["_strike", "_oi"])
        if sub.empty:
            continue

        sub = sub[strike_filter(sub["_strike"])]
        if sub.empty:
            continue

        grouped = sub.groupby("_strike", as_index=False)["_oi"].sum()
        grouped = grouped.sort_values(["_oi", "_strike"], ascending=[False, True]).head(top)
        out[opt_type] = [
            WallLevel(strike=float(r["_strike"]), open_interest=float(r["_oi"])) for _, r in grouped.iterrows()
        ]

    return Walls(calls=out["call"], puts=out["put"])


def _compute_expiry_atm(
    df: pd.DataFrame,
    *,
    spot: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Returns: (atm_strike, call_mark, put_mark, straddle_mark)
    """
    if df.empty or "strike" not in df.columns or "optionType" not in df.columns:
        return (None, None, None, None)

    strike = pd.to_numeric(df["strike"], errors="coerce")
    mark = df["mark"] if "mark" in df.columns else compute_mark_price(df)
    sub = df.assign(_strike=strike, _mark=mark).dropna(subset=["_strike", "_mark"])
    if sub.empty:
        return (None, None, None, None)

    piv = sub.pivot_table(index="_strike", columns="optionType", values="_mark", aggfunc="first")
    if "call" not in piv.columns or "put" not in piv.columns:
        return (None, None, None, None)

    piv = piv.dropna(subset=["call", "put"])
    if piv.empty:
        return (None, None, None, None)

    piv = piv.assign(_dist=(piv.index.to_series() - spot).abs())
    piv = piv.sort_values(["_dist"], ascending=True)
    atm_strike = float(piv.index[0])
    call_mark = float(piv.iloc[0]["call"])
    put_mark = float(piv.iloc[0]["put"])
    straddle = call_mark + put_mark
    return (atm_strike, call_mark, put_mark, straddle)


def _compute_atm_iv(df: pd.DataFrame, *, atm_strike: float | None) -> float | None:
    if atm_strike is None or df.empty or "strike" not in df.columns:
        return None
    if "impliedVolatility" not in df.columns:
        return None

    strike = pd.to_numeric(df["strike"], errors="coerce")
    iv = _col_as_float(df, "impliedVolatility")
    sub = df.assign(_strike=strike, _iv=iv)
    sub = sub[sub["_strike"] == atm_strike].dropna(subset=["_iv"])
    if sub.empty:
        return None
    return float(sub["_iv"].mean())


def _compute_skew_25d_pp(df: pd.DataFrame) -> float | None:
    if df.empty or "bs_delta" not in df.columns or "impliedVolatility" not in df.columns:
        return None

    delta = _col_as_float(df, "bs_delta")
    iv = _col_as_float(df, "impliedVolatility")
    sub = df.assign(_delta=delta, _iv=iv).dropna(subset=["_delta", "_iv"])
    if sub.empty or "optionType" not in sub.columns:
        return None

    def _pick(target: float, opt_type: str) -> float | None:
        d = sub[sub["optionType"] == opt_type].copy()
        if d.empty:
            return None
        d["_dist"] = (d["_delta"] - target).abs()
        d = d.sort_values(["_dist"], ascending=True)
        return float(d.iloc[0]["_iv"])

    call_iv = _pick(0.25, "call")
    put_iv = _pick(-0.25, "put")
    if call_iv is None or put_iv is None:
        return None
    return (put_iv - call_iv) * 100.0


def _compute_expiry_stats(
    df: pd.DataFrame,
    *,
    expiry: str,
    spot: float,
) -> ExpiryStats:
    calls = df[df.get("optionType") == "call"] if "optionType" in df.columns else df.iloc[0:0]
    puts = df[df.get("optionType") == "put"] if "optionType" in df.columns else df.iloc[0:0]

    calls_oi = _safe_sum(_col_as_float(calls, "openInterest"))
    puts_oi = _safe_sum(_col_as_float(puts, "openInterest"))
    calls_vol = _safe_sum(_col_as_float(calls, "volume"))
    puts_vol = _safe_sum(_col_as_float(puts, "volume"))

    atm_strike, call_mark, put_mark, straddle = _compute_expiry_atm(df, spot=spot)
    atm_iv = _compute_atm_iv(df, atm_strike=atm_strike)
    skew_25d_pp = _compute_skew_25d_pp(df)

    expected_move = straddle
    expected_move_pct = (straddle / spot) if (straddle is not None and spot > 0) else None

    return ExpiryStats(
        expiry=expiry,
        calls_oi=calls_oi,
        puts_oi=puts_oi,
        pc_oi_ratio=_safe_ratio(puts_oi, calls_oi),
        calls_volume=calls_vol,
        puts_volume=puts_vol,
        pc_volume_ratio=_safe_ratio(puts_vol, calls_vol),
        atm_strike=atm_strike,
        call_mark_atm=call_mark,
        put_mark_atm=put_mark,
        straddle_mark_atm=straddle,
        expected_move=expected_move,
        expected_move_pct=expected_move_pct,
        atm_iv=atm_iv,
        skew_25d_pp=skew_25d_pp,
    )


def _compute_gamma_summary(df: pd.DataFrame, *, spot: float, top: int) -> GammaSummary:
    if df.empty or "strike" not in df.columns or "bs_gamma" not in df.columns or "openInterest" not in df.columns:
        return GammaSummary()

    strike = pd.to_numeric(df["strike"], errors="coerce")
    gamma = _col_as_float(df, "bs_gamma")
    oi = _col_as_float(df, "openInterest")
    sub = df.assign(_strike=strike, _gamma=gamma, _oi=oi).dropna(subset=["_strike", "_gamma", "_oi"])
    if sub.empty:
        return GammaSummary()

    sub["_gamma_1pct"] = sub["_gamma"] * sub["_oi"] * 100.0 * (spot**2) * 0.01
    grouped = sub.groupby("_strike", as_index=False)["_gamma_1pct"].sum()
    grouped = grouped.sort_values(["_gamma_1pct", "_strike"], ascending=[False, True]).head(top)

    levels = [
        GammaLevel(strike=float(r["_strike"]), gamma_1pct=float(r["_gamma_1pct"])) for _, r in grouped.iterrows()
    ]
    peak_strike = levels[0].strike if levels else None
    return GammaSummary(peak_strike=peak_strike, top=levels)


def compute_chain_report(
    df: pd.DataFrame,
    *,
    symbol: str,
    as_of: date,
    spot: float,
    expiries_mode: Literal["near", "monthly", "all"] = "near",
    include_expiries: list[date] | None = None,
    top: int = 10,
    best_effort: bool = False,
) -> ChainReport:
    if spot <= 0:
        raise ValueError("spot must be > 0")
    out = df.copy() if df is not None else pd.DataFrame()
    if out.empty:
        return _empty_chain_report(symbol=symbol, as_of=as_of, spot=spot, best_effort=best_effort)
    warnings, out = _normalize_chain_frame(out)
    out["mark"] = compute_mark_price(out)
    available_dates = _parse_available_expiry_dates(out)
    selected = select_expiries(available_dates, mode=expiries_mode, include=include_expiries)
    included_strs = [d.isoformat() for d in selected]
    available_strs = [d.isoformat() for d in available_dates]
    totals = _compute_chain_totals(out)
    walls_overall = _compute_walls(out, spot=spot, top=top)
    expiries, walls_by_expiry = _compute_expiry_breakdown(out, selected=selected, spot=spot, top=top)
    gamma = _compute_gamma_summary(out, spot=spot, top=top)
    return ChainReport(
        symbol=symbol.upper(),
        as_of=as_of.isoformat(),
        spot=float(spot),
        available_expiries=available_strs,
        included_expiries=included_strs,
        totals=totals,
        expiries=expiries,
        walls_overall=walls_overall,
        walls_by_expiry=walls_by_expiry,
        gamma=gamma,
        warnings=warnings,
    )


def _empty_chain_report(*, symbol: str, as_of: date, spot: float, best_effort: bool) -> ChainReport:
    if best_effort:
        return ChainReport(
            symbol=symbol.upper(),
            as_of=as_of.isoformat(),
            spot=spot,
            totals=ChainTotals(),
            walls_overall=Walls(),
            gamma=GammaSummary(),
            warnings=["empty_snapshot"],
        )
    raise ValueError("empty snapshot")


def _normalize_chain_frame(out: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    warnings: list[str] = []
    if "optionType" in out.columns:
        out["optionType"] = out["optionType"].astype(str).str.lower().str.strip()
    else:
        warnings.append("missing_optionType")
    if "expiry" not in out.columns:
        warnings.append("missing_expiry")
        out["expiry"] = None
    return warnings, out


def _parse_available_expiry_dates(out: pd.DataFrame) -> list[date]:
    available_dates: list[date] = []
    expiry_strs = sorted({str(value) for value in out["expiry"].dropna().unique().tolist()})
    for expiry_str in expiry_strs:
        try:
            available_dates.append(date.fromisoformat(expiry_str))
        except ValueError:
            continue
    return sorted(set(available_dates))


def _compute_chain_totals(out: pd.DataFrame) -> ChainTotals:
    calls = out[out.get("optionType") == "call"] if "optionType" in out.columns else out.iloc[0:0]
    puts = out[out.get("optionType") == "put"] if "optionType" in out.columns else out.iloc[0:0]
    calls_oi = _safe_sum(_col_as_float(calls, "openInterest"))
    puts_oi = _safe_sum(_col_as_float(puts, "openInterest"))
    calls_vol = _safe_sum(_col_as_float(calls, "volume"))
    puts_vol = _safe_sum(_col_as_float(puts, "volume"))
    calls_vol_notional = (
        _safe_sum(_col_as_float(calls, "volume") * calls["mark"] * 100.0) if "mark" in calls.columns else 0.0
    )
    puts_vol_notional = (
        _safe_sum(_col_as_float(puts, "volume") * puts["mark"] * 100.0) if "mark" in puts.columns else 0.0
    )
    return ChainTotals(
        calls_oi=calls_oi,
        puts_oi=puts_oi,
        pc_oi_ratio=_safe_ratio(puts_oi, calls_oi),
        calls_volume=calls_vol,
        puts_volume=puts_vol,
        pc_volume_ratio=_safe_ratio(puts_vol, calls_vol),
        calls_volume_notional=calls_vol_notional,
        puts_volume_notional=puts_vol_notional,
    )


def _compute_expiry_breakdown(
    out: pd.DataFrame,
    *,
    selected: list[date],
    spot: float,
    top: int,
) -> tuple[list[ExpiryStats], list[WallsByExpiry]]:
    expiries: list[ExpiryStats] = []
    walls_by_expiry: list[WallsByExpiry] = []
    for expiry in selected:
        expiry_str = expiry.isoformat()
        sub = out[out["expiry"] == expiry_str]
        expiries.append(_compute_expiry_stats(sub, expiry=expiry_str, spot=spot))
        walls_by_expiry.append(WallsByExpiry(expiry=expiry_str, walls=_compute_walls(sub, spot=spot, top=top)))
    return expiries, walls_by_expiry
