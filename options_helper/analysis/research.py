from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import cast

import pandas as pd

from options_helper.analysis.chain_metrics import execution_quality
from options_helper.analysis.confluence import ConfluenceInputs
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.greeks import black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, stoch_rsi
from options_helper.analysis.volatility import realized_vol
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.models import OptionType, RiskProfile


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class UnderlyingSetup:
    symbol: str
    spot: float | None
    direction: Direction
    reasons: list[str]
    daily_rsi: float | None
    daily_stoch_rsi: float | None
    weekly_rsi: float | None
    weekly_breakout: bool | None


@dataclass(frozen=True)
class OptionCandidate:
    symbol: str
    option_type: OptionType
    expiry: date
    dte: int
    strike: float
    mark: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    spread_pct: float | None
    execution_quality: str | None
    last: float | None
    iv: float | None
    delta: float | None
    open_interest: int | None
    volume: int | None
    quality_score: float | None
    quality_label: str | None
    last_trade_age_days: int | None
    rationale: list[str]
    quality_warnings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exclude: bool = False


@dataclass(frozen=True)
class VolatilityContext:
    rv_20d: float | None
    iv_rv_20d: float | None
    iv_percentile: float | None
    atm_iv: float | None


@dataclass(frozen=True)
class SpreadCandidate:
    symbol: str
    expiry: date
    dte: int
    option_type: OptionType
    long_leg: OptionCandidate
    short_leg: OptionCandidate
    debit: float | None
    rationale: list[str]


@dataclass(frozen=True)
class TradeLevels:
    entry: float | None
    pullback_entry: float | None
    stop: float | None
    notes: list[str]


def build_confluence_inputs(
    setup: UnderlyingSetup,
    *,
    extension_percentile: float | None = None,
    vol_context: VolatilityContext | None = None,
    flow_delta_oi_notional: float | None = None,
    rsi_divergence: str | None = None,
) -> ConfluenceInputs:
    if setup.direction == Direction.BULLISH:
        trend = "up"
    elif setup.direction == Direction.BEARISH:
        trend = "down"
    else:
        trend = "flat"

    return ConfluenceInputs(
        weekly_trend=trend,
        extension_percentile=extension_percentile,
        rsi_divergence=rsi_divergence,
        flow_delta_oi_notional=flow_delta_oi_notional,
        iv_rv_20d=None if vol_context is None else vol_context.iv_rv_20d,
    )


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


def _as_float(val) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _as_int(val) -> int | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
            return None
        return int(val)
    except Exception:  # noqa: BLE001
        return None


def choose_expiry(
    expiry_strs: list[str],
    *,
    min_dte: int,
    max_dte: int,
    target_dte: int,
    today: date | None = None,
) -> date | None:
    today = today or date.today()
    candidates: list[tuple[int, date]] = []
    for s in expiry_strs:
        try:
            exp = date.fromisoformat(s)
        except ValueError:
            continue
        dte = (exp - today).days
        if min_dte <= dte <= max_dte:
            candidates.append((dte, exp))
    if not candidates:
        return None
    # Choose expiry closest to target DTE
    _, best = min(candidates, key=lambda t: abs(t[0] - target_dte))
    return best


def analyze_underlying(
    symbol: str,
    *,
    history: pd.DataFrame,
    risk_profile: RiskProfile,
) -> UnderlyingSetup:
    reasons: list[str] = []

    if history.empty or "Close" not in history.columns:
        return UnderlyingSetup(
            symbol=symbol.upper(),
            spot=None,
            direction=Direction.NEUTRAL,
            reasons=["No candle history available."],
            daily_rsi=None,
            daily_stoch_rsi=None,
            weekly_rsi=None,
            weekly_breakout=None,
        )

    close = history["Close"].dropna()
    if close.empty:
        return UnderlyingSetup(
            symbol=symbol.upper(),
            spot=None,
            direction=Direction.NEUTRAL,
            reasons=["No close prices available."],
            daily_rsi=None,
            daily_stoch_rsi=None,
            weekly_rsi=None,
            weekly_breakout=None,
        )

    spot = float(close.iloc[-1])

    rsi_d = rsi(close, 14)
    stoch_rsi_d = stoch_rsi(close)

    close_w = close.resample("W-FRI").last().dropna()
    ema20_w = ema(close_w, 20) if not close_w.empty else None
    ema50_w = ema(close_w, 50) if not close_w.empty else None
    rsi_w = rsi(close_w, 14) if not close_w.empty else None

    lookback = risk_profile.breakout_lookback_weeks
    breakout_w = breakout_up(close_w, lookback) if not close_w.empty else None

    bullish_weekly = False
    bearish_weekly = False
    if ema20_w is not None and ema50_w is not None and spot is not None:
        bullish_weekly = (spot > ema50_w) and (ema20_w >= ema50_w)
        bearish_weekly = (spot < ema50_w) and (ema20_w <= ema50_w)

    if bullish_weekly:
        reasons.append("Weekly trend bullish (price above EMA50; EMA20 ≥ EMA50).")
    elif bearish_weekly:
        reasons.append("Weekly trend bearish (price below EMA50; EMA20 ≤ EMA50).")
    else:
        reasons.append("Weekly trend neutral/mixed.")

    if breakout_w is True:
        reasons.append(f"Weekly breakout: close exceeded prior {lookback}-week high.")

    if rsi_w is not None:
        if rsi_w >= 55:
            reasons.append(f"Weekly RSI strong ({rsi_w:.0f} ≥ 55).")
        elif rsi_w <= 45:
            reasons.append(f"Weekly RSI weak ({rsi_w:.0f} ≤ 45).")
        else:
            reasons.append(f"Weekly RSI neutral ({rsi_w:.0f}).")

    if stoch_rsi_d is not None:
        # User-friendly banding
        if 40 <= stoch_rsi_d <= 60:
            reasons.append(f"Daily StochRSI near mid-range ({stoch_rsi_d:.0f}), room for momentum expansion.")
        elif stoch_rsi_d > 80:
            reasons.append(f"Daily StochRSI elevated ({stoch_rsi_d:.0f}), momentum already extended.")
        elif stoch_rsi_d < 20:
            reasons.append(f"Daily StochRSI depressed ({stoch_rsi_d:.0f}), potential reset/bounce zone.")
        else:
            reasons.append(f"Daily StochRSI {stoch_rsi_d:.0f}.")

    if rsi_d is not None:
        if rsi_d >= 55:
            reasons.append(f"Daily RSI supportive ({rsi_d:.0f} ≥ 55).")
        elif rsi_d <= 45:
            reasons.append(f"Daily RSI weak ({rsi_d:.0f} ≤ 45).")
        else:
            reasons.append(f"Daily RSI neutral ({rsi_d:.0f}).")

    # Direction decision (simple, LEAPS-oriented)
    direction = Direction.NEUTRAL
    if bullish_weekly and (breakout_w is True or (rsi_w is not None and rsi_w >= 55)):
        direction = Direction.BULLISH
    elif bearish_weekly and (breakout_down(close_w, lookback) is True or (rsi_w is not None and rsi_w <= 45)):
        direction = Direction.BEARISH

    return UnderlyingSetup(
        symbol=symbol.upper(),
        spot=spot,
        direction=direction,
        reasons=reasons,
        daily_rsi=rsi_d,
        daily_stoch_rsi=stoch_rsi_d,
        weekly_rsi=rsi_w,
        weekly_breakout=breakout_w,
    )


def _breakout_up_level(close: pd.Series, lookback: int) -> float | None:
    if lookback <= 1 or close.empty or len(close) < lookback + 1:
        return None
    return float(close.iloc[-(lookback + 1) : -1].max())


def _breakout_down_level(close: pd.Series, lookback: int) -> float | None:
    if lookback <= 1 or close.empty or len(close) < lookback + 1:
        return None
    return float(close.iloc[-(lookback + 1) : -1].min())


def _select_close(history: pd.DataFrame) -> pd.Series:
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    if "Adj Close" in history.columns:
        return pd.to_numeric(history["Adj Close"], errors="coerce")
    if "Close" in history.columns:
        return pd.to_numeric(history["Close"], errors="coerce")
    return pd.Series(dtype="float64")


def _atm_iv_from_chain(calls: pd.DataFrame | None, puts: pd.DataFrame | None, spot: float) -> float | None:
    frames: list[pd.DataFrame] = []
    for df in (calls, puts):
        if df is None or df.empty:
            continue
        strike = pd.to_numeric(df.get("strike"), errors="coerce")
        iv = pd.to_numeric(df.get("impliedVolatility"), errors="coerce")
        sub = pd.DataFrame({"strike": strike, "iv": iv}).dropna(subset=["strike", "iv"])
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return None

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["_dist"] = (all_rows["strike"] - spot).abs()
    if all_rows.empty or all_rows["_dist"].isna().all():
        return None
    min_dist = all_rows["_dist"].min()
    atm = all_rows[all_rows["_dist"] == min_dist]
    if atm.empty:
        return None
    return float(atm["iv"].mean())


def _percentile_rank_last(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    n = int(len(values))
    if n == 1:
        return 100.0
    ranks = values.rank(method="average")
    r = float(ranks.iloc[-1])
    return float((r - 1.0) / (n - 1.0) * 100.0)


def compute_volatility_context(
    *,
    history: pd.DataFrame,
    spot: float,
    calls: pd.DataFrame | None,
    puts: pd.DataFrame | None,
    derived_history: pd.DataFrame | None = None,
) -> VolatilityContext:
    close = _select_close(history)
    rv_20d = None
    if not close.empty:
        rv_series = realized_vol(close, 20)
        rv_clean = rv_series.dropna()
        if not rv_clean.empty:
            rv_20d = float(rv_clean.iloc[-1])

    atm_iv = _atm_iv_from_chain(calls, puts, spot)
    iv_rv_20d = None if (rv_20d is None or rv_20d <= 0 or atm_iv is None) else atm_iv / rv_20d

    iv_percentile = None
    if atm_iv is not None:
        values = pd.Series(dtype="float64")
        if derived_history is not None and not derived_history.empty and "atm_iv_near" in derived_history.columns:
            values = pd.to_numeric(derived_history["atm_iv_near"], errors="coerce").dropna()
        combined = pd.concat([values, pd.Series([atm_iv])], ignore_index=True)
        iv_percentile = _percentile_rank_last(combined)

    return VolatilityContext(rv_20d=rv_20d, iv_rv_20d=iv_rv_20d, iv_percentile=iv_percentile, atm_iv=atm_iv)


def suggest_trade_levels(
    setup: UnderlyingSetup,
    *,
    history: pd.DataFrame,
    risk_profile: RiskProfile,
    swing_lookback_days: int = 20,
) -> TradeLevels:
    """
    Best-effort entry/stop levels on the underlying (not the option premium).

    This is intentionally simple for the MVP; we can iterate into ATR-based stops and explicit S/R later.
    """
    if setup.spot is None or history is None or history.empty:
        return TradeLevels(entry=None, pullback_entry=None, stop=None, notes=["No candle history available."])

    entry = float(setup.spot)
    buffer_pct = float(risk_profile.support_proximity_pct)
    lookback_weeks = int(risk_profile.breakout_lookback_weeks)
    notes: list[str] = []

    close = history.get("Close")
    if close is None:
        return TradeLevels(entry=entry, pullback_entry=None, stop=None, notes=["No close prices available."])
    close = close.dropna()
    if close.empty:
        return TradeLevels(entry=entry, pullback_entry=None, stop=None, notes=["No close prices available."])

    low = history.get("Low", close).dropna()
    high = history.get("High", close).dropna()

    ema20_d = ema(close, 20)
    ema50_d = ema(close, 50)

    close_w = close.resample("W-FRI").last().dropna()
    ema20_w = ema(close_w, 20) if not close_w.empty else None
    ema50_w = ema(close_w, 50) if not close_w.empty else None

    swing_low = None
    swing_high = None
    if not low.empty:
        swing_low = float(low.iloc[-swing_lookback_days:].min())
    if not high.empty:
        swing_high = float(high.iloc[-swing_lookback_days:].max())

    pullback_entry: float | None = None
    stop: float | None = None

    if setup.direction == Direction.BULLISH:
        breakout_level = _breakout_up_level(close_w, lookback_weeks)
        breakout_now = breakout_up(close_w, lookback_weeks) is True

        if breakout_now and breakout_level is not None:
            pullback_entry = breakout_level
            stop = breakout_level * (1.0 - buffer_pct)
            notes.append(f"Stop below weekly breakout level (~{breakout_level:.2f}) with {buffer_pct:.0%} buffer.")
        else:
            pullback_entry = ema20_d or ema20_w

            supports: list[tuple[str, float]] = []
            for label, value in (
                ("weekly EMA50", ema50_w),
                ("weekly EMA20", ema20_w),
                ("daily EMA50", ema50_d),
                ("daily EMA20", ema20_d),
                (f"{swing_lookback_days}d swing low", swing_low),
            ):
                if value is None:
                    continue
                supports.append((label, float(value)))

            below = [(label, value) for label, value in supports if value < entry]
            candidates = below if below else supports

            if candidates:
                if risk_profile.tolerance == "high":
                    stop_label, stop_base = min(candidates, key=lambda t: t[1])
                elif risk_profile.tolerance == "low":
                    stop_label, stop_base = max(candidates, key=lambda t: t[1])
                else:
                    stop_label, stop_base = max(candidates, key=lambda t: t[1])
                stop = stop_base * (1.0 - buffer_pct)
                notes.append(f"Stop below {stop_label} (~{stop_base:.2f}) with {buffer_pct:.0%} buffer.")

    elif setup.direction == Direction.BEARISH:
        breakdown_level = _breakout_down_level(close_w, lookback_weeks)
        breakdown_now = breakout_down(close_w, lookback_weeks) is True

        if breakdown_now and breakdown_level is not None:
            pullback_entry = breakdown_level
            stop = breakdown_level * (1.0 + buffer_pct)
            notes.append(f"Stop above weekly breakdown level (~{breakdown_level:.2f}) with {buffer_pct:.0%} buffer.")
        else:
            pullback_entry = ema20_d or ema20_w

            resistances: list[tuple[str, float]] = []
            for label, value in (
                ("weekly EMA50", ema50_w),
                ("weekly EMA20", ema20_w),
                ("daily EMA50", ema50_d),
                ("daily EMA20", ema20_d),
                (f"{swing_lookback_days}d swing high", swing_high),
            ):
                if value is None:
                    continue
                resistances.append((label, float(value)))

            above = [(label, value) for label, value in resistances if value > entry]
            candidates = above if above else resistances

            if candidates:
                if risk_profile.tolerance == "high":
                    stop_label, stop_base = max(candidates, key=lambda t: t[1])
                elif risk_profile.tolerance == "low":
                    stop_label, stop_base = min(candidates, key=lambda t: t[1])
                else:
                    stop_label, stop_base = min(candidates, key=lambda t: t[1])
                stop = stop_base * (1.0 + buffer_pct)
                notes.append(f"Stop above {stop_label} (~{stop_base:.2f}) with {buffer_pct:.0%} buffer.")

    return TradeLevels(entry=entry, pullback_entry=pullback_entry, stop=stop, notes=notes)


def _filter_strike_window(df: pd.DataFrame, *, spot: float, window_pct: float) -> pd.DataFrame:
    if df.empty or "strike" not in df.columns:
        return df
    strike_min = spot * (1.0 - window_pct)
    strike_max = spot * (1.0 + window_pct)
    return df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]


def select_option_candidate(
    df: pd.DataFrame,
    *,
    symbol: str,
    option_type: OptionType,
    expiry: date,
    spot: float,
    target_delta: float,
    window_pct: float,
    min_open_interest: int,
    min_volume: int,
    max_spread_pct: float = 0.35,
    as_of: date | None = None,
    next_earnings_date: date | None = None,
    earnings_warn_days: int = 21,
    earnings_avoid_days: int = 0,
    include_bad_quotes: bool = False,
) -> OptionCandidate | None:
    if df is None or df.empty:
        return None

    df = df.copy()
    df = _filter_strike_window(df, spot=spot, window_pct=window_pct)
    if df.empty:
        return None

    # Normalize fields
    df["bid_f"] = df.get("bid").map(_as_float) if "bid" in df.columns else None
    df["ask_f"] = df.get("ask").map(_as_float) if "ask" in df.columns else None
    df["last_f"] = df.get("lastPrice").map(_as_float) if "lastPrice" in df.columns else None
    df["iv_f"] = df.get("impliedVolatility").map(_as_float) if "impliedVolatility" in df.columns else None
    df["oi_i"] = df.get("openInterest").map(_as_int) if "openInterest" in df.columns else None
    df["vol_i"] = df.get("volume").map(_as_int) if "volume" in df.columns else None
    today = as_of or date.today()
    quality = compute_quote_quality(df, min_volume=min_volume, min_open_interest=min_open_interest, as_of=today)
    df["spread"] = quality["spread"]
    df["spread_pct"] = quality["spread_pct"]
    df["quality_score"] = quality["quality_score"]
    df["quality_label"] = quality["quality_label"]
    df["quality_warnings"] = quality["quality_warnings"]
    df["last_trade_age_days"] = quality["last_trade_age_days"]
    df["exec_quality"] = df["spread_pct"].map(execution_quality)
    dte = max((expiry - today).days, 0)
    t_years = dte / 365.0 if dte > 0 else None

    def _row_mark(row) -> float | None:
        return _mark_price(bid=row["bid_f"], ask=row["ask_f"], last=row["last_f"])

    df["mark"] = df.apply(_row_mark, axis=1)

    # Compute delta (best-effort)
    deltas: list[float | None] = []
    for _, row in df.iterrows():
        sigma = row["iv_f"]
        strike = _as_float(row.get("strike"))
        if sigma is None or strike is None or t_years is None or t_years <= 0:
            deltas.append(None)
            continue
        g = black_scholes_greeks(option_type=option_type, s=spot, k=strike, t_years=t_years, sigma=sigma)
        deltas.append(g.delta if g else None)
    df["delta"] = deltas

    # Prefer liquid strikes, but fall back if needed.
    liquid = df
    if "oi_i" in df.columns and "vol_i" in df.columns:
        liquid = df[(df["oi_i"].fillna(0) >= min_open_interest) | (df["vol_i"].fillna(0) >= min_volume)]
        if liquid.empty:
            liquid = df

    spread_gate_fallback = False
    if "spread_pct" in liquid.columns:
        spread_pct = liquid["spread_pct"]
        bad_mask = spread_pct.notna() & ((spread_pct < 0) | (spread_pct > max_spread_pct))
        spread_filtered = liquid[~bad_mask]
        if not spread_filtered.empty:
            liquid = spread_filtered
        else:
            spread_gate_fallback = True

    quality_gate_fallback = False
    if "quality_label" in liquid.columns and not include_bad_quotes:
        quality_filtered = liquid[liquid["quality_label"].fillna("unknown") != "bad"]
        if not quality_filtered.empty:
            liquid = quality_filtered
        else:
            quality_gate_fallback = True

    # Choose by delta if available, otherwise by moneyness.
    if liquid["delta"].notna().any():
        liquid = liquid[liquid["delta"].notna()]
        pick = liquid.iloc[(liquid["delta"] - target_delta).abs().argsort()].iloc[0]
    else:
        # Fallback: closest strike to spot (ATM)
        pick = liquid.iloc[(liquid["strike"].astype(float) - spot).abs().argsort()].iloc[0]

    strike = float(pick["strike"])
    bid = pick["bid_f"]
    ask = pick["ask_f"]
    last = pick["last_f"]
    mark = pick["mark"]
    spread = _as_float(pick.get("spread"))
    spread_pct = _as_float(pick.get("spread_pct"))
    exec_quality = pick.get("exec_quality")
    quality_score = _as_float(pick.get("quality_score"))
    quality_label = pick.get("quality_label")
    last_trade_age_days = _as_int(pick.get("last_trade_age_days"))
    quality_warnings = pick.get("quality_warnings")
    iv = pick["iv_f"]
    oi = pick["oi_i"]
    vol = pick["vol_i"]
    delta = pick["delta"] if pick["delta"] is not None and not pd.isna(pick["delta"]) else None

    rationale = [
        f"Selected strike nearest target delta {target_delta:.2f} (best-effort BS delta)." if delta is not None else "Selected nearest ATM strike (delta unavailable)."
    ]
    if oi is not None or vol is not None:
        rationale.append(f"Liquidity (OI={oi if oi is not None else 'n/a'}, Vol={vol if vol is not None else 'n/a'}).")
    if spread_pct is not None:
        rationale.append(f"Execution: {exec_quality} (spread {spread_pct:.1%}).")
    else:
        rationale.append(f"Execution: {exec_quality}.")
    if spread_gate_fallback:
        rationale.append("Spread quality was poor across candidates; used best-effort pick.")
    if quality_gate_fallback:
        rationale.append("Quote quality was poor across candidates; used best-effort pick.")

    risk = earnings_event_risk(
        today=today,
        expiry=expiry,
        next_earnings_date=next_earnings_date,
        warn_days=earnings_warn_days,
        avoid_days=earnings_avoid_days,
    )

    return OptionCandidate(
        symbol=symbol.upper(),
        option_type=option_type,
        expiry=expiry,
        dte=dte,
        strike=strike,
        mark=mark,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_pct=spread_pct,
        execution_quality=str(exec_quality) if exec_quality is not None else None,
        last=last,
        iv=iv,
        delta=delta,
        open_interest=oi,
        volume=vol,
        quality_score=quality_score,
        quality_label=str(quality_label) if quality_label is not None else None,
        last_trade_age_days=last_trade_age_days,
        quality_warnings=cast(list[str], quality_warnings) if quality_warnings is not None else [],
        rationale=rationale,
        warnings=cast(list[str], risk["warnings"]),
        exclude=bool(risk["exclude"]),
    )


def build_call_debit_spread(
    calls_df: pd.DataFrame,
    *,
    symbol: str,
    expiry: date,
    spot: float,
    window_pct: float,
    min_open_interest: int,
    min_volume: int,
    long_delta: float = 0.40,
    short_delta: float = 0.25,
    as_of: date | None = None,
    next_earnings_date: date | None = None,
    earnings_warn_days: int = 21,
    earnings_avoid_days: int = 0,
) -> SpreadCandidate | None:
    long_leg = select_option_candidate(
        calls_df,
        symbol=symbol,
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=long_delta,
        window_pct=window_pct,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        as_of=as_of,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=earnings_warn_days,
        earnings_avoid_days=earnings_avoid_days,
    )
    if long_leg is None:
        return None

    # Choose a higher strike for the short leg (cap upside, reduce premium).
    higher = calls_df[calls_df["strike"].astype(float) > long_leg.strike]
    if higher.empty:
        return None

    short_leg = select_option_candidate(
        higher,
        symbol=symbol,
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=short_delta,
        window_pct=window_pct,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        as_of=as_of,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=earnings_warn_days,
        earnings_avoid_days=earnings_avoid_days,
    )
    if short_leg is None:
        return None

    debit = None
    if long_leg.mark is not None and short_leg.mark is not None:
        debit = long_leg.mark - short_leg.mark

    rationale = [
        "Call debit spread reduces premium paid vs outright long call.",
        "Short leg caps upside; suitable when you have a near-term target or want defined risk/reward.",
    ]

    return SpreadCandidate(
        symbol=symbol.upper(),
        expiry=expiry,
        dte=long_leg.dte,
        option_type="call",
        long_leg=long_leg,
        short_leg=short_leg,
        debit=debit,
        rationale=rationale,
    )
