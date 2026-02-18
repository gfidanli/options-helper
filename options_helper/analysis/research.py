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
        return _empty_underlying_setup(symbol, reason="No candle history available.")
    close = history["Close"].dropna()
    if close.empty:
        return _empty_underlying_setup(symbol, reason="No close prices available.")
    spot = float(close.iloc[-1])
    rsi_d = rsi(close, 14)
    stoch_rsi_d = stoch_rsi(close)
    weekly_context = _weekly_underlying_context(close=close, spot=spot, risk_profile=risk_profile)
    _append_weekly_trend_reasons(reasons, weekly_context)
    _append_momentum_reasons(reasons, weekly_context=weekly_context, daily_rsi=rsi_d, daily_stoch_rsi=stoch_rsi_d)
    direction = _choose_underlying_direction(close=weekly_context["close_w"], weekly_context=weekly_context)
    return UnderlyingSetup(
        symbol=symbol.upper(),
        spot=spot,
        direction=direction,
        reasons=reasons,
        daily_rsi=rsi_d,
        daily_stoch_rsi=stoch_rsi_d,
        weekly_rsi=cast(float | None, weekly_context["weekly_rsi"]),
        weekly_breakout=cast(bool | None, weekly_context["breakout"]),
    )


def _empty_underlying_setup(symbol: str, *, reason: str) -> UnderlyingSetup:
    return UnderlyingSetup(
        symbol=symbol.upper(),
        spot=None,
        direction=Direction.NEUTRAL,
        reasons=[reason],
        daily_rsi=None,
        daily_stoch_rsi=None,
        weekly_rsi=None,
        weekly_breakout=None,
    )


def _weekly_underlying_context(
    *,
    close: pd.Series,
    spot: float,
    risk_profile: RiskProfile,
) -> dict[str, object]:
    close_w = close.resample("W-FRI").last().dropna()
    ema20_w = ema(close_w, 20) if not close_w.empty else None
    ema50_w = ema(close_w, 50) if not close_w.empty else None
    weekly_rsi = rsi(close_w, 14) if not close_w.empty else None
    lookback = int(risk_profile.breakout_lookback_weeks)
    breakout = breakout_up(close_w, lookback) if not close_w.empty else None
    bullish_weekly = bool(ema20_w is not None and ema50_w is not None and spot > ema50_w and ema20_w >= ema50_w)
    bearish_weekly = bool(ema20_w is not None and ema50_w is not None and spot < ema50_w and ema20_w <= ema50_w)
    return {
        "close_w": close_w,
        "lookback": lookback,
        "weekly_rsi": weekly_rsi,
        "breakout": breakout,
        "bullish_weekly": bullish_weekly,
        "bearish_weekly": bearish_weekly,
    }


def _append_weekly_trend_reasons(reasons: list[str], weekly_context: dict[str, object]) -> None:
    if cast(bool, weekly_context["bullish_weekly"]):
        reasons.append("Weekly trend bullish (price above EMA50; EMA20 ≥ EMA50).")
    elif cast(bool, weekly_context["bearish_weekly"]):
        reasons.append("Weekly trend bearish (price below EMA50; EMA20 ≤ EMA50).")
    else:
        reasons.append("Weekly trend neutral/mixed.")
    if weekly_context["breakout"] is True:
        reasons.append(f"Weekly breakout: close exceeded prior {weekly_context['lookback']}-week high.")


def _append_momentum_reasons(
    reasons: list[str],
    *,
    weekly_context: dict[str, object],
    daily_rsi: float | None,
    daily_stoch_rsi: float | None,
) -> None:
    weekly_rsi = cast(float | None, weekly_context["weekly_rsi"])
    if weekly_rsi is not None:
        if weekly_rsi >= 55:
            reasons.append(f"Weekly RSI strong ({weekly_rsi:.0f} ≥ 55).")
        elif weekly_rsi <= 45:
            reasons.append(f"Weekly RSI weak ({weekly_rsi:.0f} ≤ 45).")
        else:
            reasons.append(f"Weekly RSI neutral ({weekly_rsi:.0f}).")
    if daily_stoch_rsi is not None:
        if 40 <= daily_stoch_rsi <= 60:
            reasons.append(f"Daily StochRSI near mid-range ({daily_stoch_rsi:.0f}), room for momentum expansion.")
        elif daily_stoch_rsi > 80:
            reasons.append(f"Daily StochRSI elevated ({daily_stoch_rsi:.0f}), momentum already extended.")
        elif daily_stoch_rsi < 20:
            reasons.append(f"Daily StochRSI depressed ({daily_stoch_rsi:.0f}), potential reset/bounce zone.")
        else:
            reasons.append(f"Daily StochRSI {daily_stoch_rsi:.0f}.")
    if daily_rsi is not None:
        if daily_rsi >= 55:
            reasons.append(f"Daily RSI supportive ({daily_rsi:.0f} ≥ 55).")
        elif daily_rsi <= 45:
            reasons.append(f"Daily RSI weak ({daily_rsi:.0f} ≤ 45).")
        else:
            reasons.append(f"Daily RSI neutral ({daily_rsi:.0f}).")


def _choose_underlying_direction(*, close: pd.Series, weekly_context: dict[str, object]) -> Direction:
    bullish_weekly = cast(bool, weekly_context["bullish_weekly"])
    bearish_weekly = cast(bool, weekly_context["bearish_weekly"])
    weekly_rsi = cast(float | None, weekly_context["weekly_rsi"])
    lookback = cast(int, weekly_context["lookback"])
    breakout = weekly_context["breakout"] is True
    if bullish_weekly and (breakout or (weekly_rsi is not None and weekly_rsi >= 55)):
        return Direction.BULLISH
    if bearish_weekly and (
        breakout_down(close, lookback) is True or (weekly_rsi is not None and weekly_rsi <= 45)
    ):
        return Direction.BEARISH
    return Direction.NEUTRAL


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
    close = history.get("Close")
    if close is None:
        return TradeLevels(entry=entry, pullback_entry=None, stop=None, notes=["No close prices available."])
    close = close.dropna()
    if close.empty:
        return TradeLevels(entry=entry, pullback_entry=None, stop=None, notes=["No close prices available."])
    inputs = _trade_level_inputs(
        history=history,
        close=close,
        swing_lookback_days=swing_lookback_days,
    )
    pullback_entry: float | None = None
    stop: float | None = None
    notes: list[str] = []
    if setup.direction == Direction.BULLISH:
        pullback_entry, stop, notes = _suggest_bullish_trade_levels(
            entry=entry,
            risk_profile=risk_profile,
            swing_lookback_days=swing_lookback_days,
            inputs=inputs,
        )
    elif setup.direction == Direction.BEARISH:
        pullback_entry, stop, notes = _suggest_bearish_trade_levels(
            entry=entry,
            risk_profile=risk_profile,
            swing_lookback_days=swing_lookback_days,
            inputs=inputs,
        )
    return TradeLevels(entry=entry, pullback_entry=pullback_entry, stop=stop, notes=notes)


def _trade_level_inputs(
    *,
    history: pd.DataFrame,
    close: pd.Series,
    swing_lookback_days: int,
) -> dict[str, float | pd.Series | None]:
    low = history.get("Low", close).dropna()
    high = history.get("High", close).dropna()
    close_w = close.resample("W-FRI").last().dropna()
    return {
        "close_w": close_w,
        "ema20_d": ema(close, 20),
        "ema50_d": ema(close, 50),
        "ema20_w": ema(close_w, 20) if not close_w.empty else None,
        "ema50_w": ema(close_w, 50) if not close_w.empty else None,
        "swing_low": float(low.iloc[-swing_lookback_days:].min()) if not low.empty else None,
        "swing_high": float(high.iloc[-swing_lookback_days:].max()) if not high.empty else None,
    }


def _suggest_bullish_trade_levels(
    *,
    entry: float,
    risk_profile: RiskProfile,
    swing_lookback_days: int,
    inputs: dict[str, float | pd.Series | None],
) -> tuple[float | None, float | None, list[str]]:
    buffer_pct = float(risk_profile.support_proximity_pct)
    lookback_weeks = int(risk_profile.breakout_lookback_weeks)
    close_w = cast(pd.Series, inputs["close_w"])
    breakout_level = _breakout_up_level(close_w, lookback_weeks)
    breakout_now = breakout_up(close_w, lookback_weeks) is True
    if breakout_now and breakout_level is not None:
        return (
            breakout_level,
            breakout_level * (1.0 - buffer_pct),
            [f"Stop below weekly breakout level (~{breakout_level:.2f}) with {buffer_pct:.0%} buffer."],
        )
    pullback_entry = cast(float | None, inputs["ema20_d"]) or cast(float | None, inputs["ema20_w"])
    supports = _collect_trade_levels(
        (
            ("weekly EMA50", cast(float | None, inputs["ema50_w"])),
            ("weekly EMA20", cast(float | None, inputs["ema20_w"])),
            ("daily EMA50", cast(float | None, inputs["ema50_d"])),
            ("daily EMA20", cast(float | None, inputs["ema20_d"])),
            (f"{swing_lookback_days}d swing low", cast(float | None, inputs["swing_low"])),
        )
    )
    stop_base = _select_stop_level(
        levels=supports,
        entry=entry,
        tolerance=risk_profile.tolerance,
        bullish=True,
    )
    if stop_base is None:
        return pullback_entry, None, []
    stop_label, stop_value = stop_base
    return (
        pullback_entry,
        stop_value * (1.0 - buffer_pct),
        [f"Stop below {stop_label} (~{stop_value:.2f}) with {buffer_pct:.0%} buffer."],
    )


def _suggest_bearish_trade_levels(
    *,
    entry: float,
    risk_profile: RiskProfile,
    swing_lookback_days: int,
    inputs: dict[str, float | pd.Series | None],
) -> tuple[float | None, float | None, list[str]]:
    buffer_pct = float(risk_profile.support_proximity_pct)
    lookback_weeks = int(risk_profile.breakout_lookback_weeks)
    close_w = cast(pd.Series, inputs["close_w"])
    breakdown_level = _breakout_down_level(close_w, lookback_weeks)
    breakdown_now = breakout_down(close_w, lookback_weeks) is True
    if breakdown_now and breakdown_level is not None:
        return (
            breakdown_level,
            breakdown_level * (1.0 + buffer_pct),
            [f"Stop above weekly breakdown level (~{breakdown_level:.2f}) with {buffer_pct:.0%} buffer."],
        )
    pullback_entry = cast(float | None, inputs["ema20_d"]) or cast(float | None, inputs["ema20_w"])
    resistances = _collect_trade_levels(
        (
            ("weekly EMA50", cast(float | None, inputs["ema50_w"])),
            ("weekly EMA20", cast(float | None, inputs["ema20_w"])),
            ("daily EMA50", cast(float | None, inputs["ema50_d"])),
            ("daily EMA20", cast(float | None, inputs["ema20_d"])),
            (f"{swing_lookback_days}d swing high", cast(float | None, inputs["swing_high"])),
        )
    )
    stop_base = _select_stop_level(
        levels=resistances,
        entry=entry,
        tolerance=risk_profile.tolerance,
        bullish=False,
    )
    if stop_base is None:
        return pullback_entry, None, []
    stop_label, stop_value = stop_base
    return (
        pullback_entry,
        stop_value * (1.0 + buffer_pct),
        [f"Stop above {stop_label} (~{stop_value:.2f}) with {buffer_pct:.0%} buffer."],
    )


def _collect_trade_levels(items: tuple[tuple[str, float | None], ...]) -> list[tuple[str, float]]:
    levels: list[tuple[str, float]] = []
    for label, value in items:
        if value is None:
            continue
        levels.append((label, float(value)))
    return levels


def _select_stop_level(
    *,
    levels: list[tuple[str, float]],
    entry: float,
    tolerance: str,
    bullish: bool,
) -> tuple[str, float] | None:
    if not levels:
        return None
    side_levels = (
        [(label, value) for label, value in levels if value < entry]
        if bullish
        else [(label, value) for label, value in levels if value > entry]
    )
    candidates = side_levels if side_levels else levels
    if bullish:
        if tolerance == "high":
            return min(candidates, key=lambda item: item[1])
        return max(candidates, key=lambda item: item[1])
    if tolerance == "high":
        return max(candidates, key=lambda item: item[1])
    return min(candidates, key=lambda item: item[1])


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
    prepared = _prepare_option_candidate_frame(df, spot=spot, window_pct=window_pct)
    if prepared is None:
        return None
    today = as_of or date.today()
    enriched, dte = _enrich_option_candidate_frame(
        prepared,
        option_type=option_type,
        spot=spot,
        expiry=expiry,
        today=today,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
    )
    liquid, spread_gate_fallback, quality_gate_fallback = _apply_option_candidate_filters(
        enriched,
        min_open_interest=min_open_interest,
        min_volume=min_volume,
        max_spread_pct=max_spread_pct,
        include_bad_quotes=include_bad_quotes,
    )
    pick = _pick_option_candidate_row(liquid, target_delta=target_delta, spot=spot)
    return _build_option_candidate_from_pick(
        pick=pick,
        symbol=symbol,
        option_type=option_type,
        expiry=expiry,
        dte=dte,
        target_delta=target_delta,
        today=today,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=earnings_warn_days,
        earnings_avoid_days=earnings_avoid_days,
        spread_gate_fallback=spread_gate_fallback,
        quality_gate_fallback=quality_gate_fallback,
    )


def _prepare_option_candidate_frame(
    df: pd.DataFrame,
    *,
    spot: float,
    window_pct: float,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    prepared = _filter_strike_window(df.copy(), spot=spot, window_pct=window_pct)
    if prepared.empty:
        return None
    return prepared


def _enrich_option_candidate_frame(
    df: pd.DataFrame,
    *,
    option_type: OptionType,
    spot: float,
    expiry: date,
    today: date,
    min_open_interest: int,
    min_volume: int,
) -> tuple[pd.DataFrame, int]:
    df["bid_f"] = df.get("bid").map(_as_float) if "bid" in df.columns else None
    df["ask_f"] = df.get("ask").map(_as_float) if "ask" in df.columns else None
    df["last_f"] = df.get("lastPrice").map(_as_float) if "lastPrice" in df.columns else None
    df["iv_f"] = df.get("impliedVolatility").map(_as_float) if "impliedVolatility" in df.columns else None
    df["oi_i"] = df.get("openInterest").map(_as_int) if "openInterest" in df.columns else None
    df["vol_i"] = df.get("volume").map(_as_int) if "volume" in df.columns else None
    quality = compute_quote_quality(df, min_volume=min_volume, min_open_interest=min_open_interest, as_of=today)
    df["spread"] = quality["spread"]
    df["spread_pct"] = quality["spread_pct"]
    df["quality_score"] = quality["quality_score"]
    df["quality_label"] = quality["quality_label"]
    df["quality_warnings"] = quality["quality_warnings"]
    df["last_trade_age_days"] = quality["last_trade_age_days"]
    df["exec_quality"] = df["spread_pct"].map(execution_quality)
    df["mark"] = df.apply(lambda row: _mark_price(bid=row["bid_f"], ask=row["ask_f"], last=row["last_f"]), axis=1)

    dte = max((expiry - today).days, 0)
    df["delta"] = _compute_candidate_deltas(
        df=df,
        option_type=option_type,
        spot=spot,
        dte=dte,
    )
    return df, dte


def _compute_candidate_deltas(
    *,
    df: pd.DataFrame,
    option_type: OptionType,
    spot: float,
    dte: int,
) -> list[float | None]:
    t_years = dte / 365.0 if dte > 0 else None
    deltas: list[float | None] = []
    for _, row in df.iterrows():
        sigma = _as_float(row.get("iv_f"))
        strike = _as_float(row.get("strike"))
        if sigma is None or strike is None or t_years is None or t_years <= 0:
            deltas.append(None)
            continue
        greeks = black_scholes_greeks(option_type=option_type, s=spot, k=strike, t_years=t_years, sigma=sigma)
        deltas.append(greeks.delta if greeks else None)
    return deltas


def _apply_option_candidate_filters(
    df: pd.DataFrame,
    *,
    min_open_interest: int,
    min_volume: int,
    max_spread_pct: float,
    include_bad_quotes: bool,
) -> tuple[pd.DataFrame, bool, bool]:
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
    return liquid, spread_gate_fallback, quality_gate_fallback


def _pick_option_candidate_row(
    liquid: pd.DataFrame,
    *,
    target_delta: float,
    spot: float,
) -> pd.Series:
    if liquid["delta"].notna().any():
        filtered = liquid[liquid["delta"].notna()]
        return filtered.iloc[(filtered["delta"] - target_delta).abs().argsort()].iloc[0]
    return liquid.iloc[(liquid["strike"].astype(float) - spot).abs().argsort()].iloc[0]


def _build_option_candidate_rationale(
    *,
    delta: float | None,
    target_delta: float,
    open_interest: int | None,
    volume: int | None,
    spread_pct: float | None,
    execution_quality: object,
    spread_gate_fallback: bool,
    quality_gate_fallback: bool,
) -> list[str]:
    rationale = [
        (
            f"Selected strike nearest target delta {target_delta:.2f} (best-effort BS delta)."
            if delta is not None
            else "Selected nearest ATM strike (delta unavailable)."
        )
    ]
    if open_interest is not None or volume is not None:
        rationale.append(
            f"Liquidity (OI={open_interest if open_interest is not None else 'n/a'}, Vol={volume if volume is not None else 'n/a'})."
        )
    if spread_pct is not None:
        rationale.append(f"Execution: {execution_quality} (spread {spread_pct:.1%}).")
    else:
        rationale.append(f"Execution: {execution_quality}.")
    if spread_gate_fallback:
        rationale.append("Spread quality was poor across candidates; used best-effort pick.")
    if quality_gate_fallback:
        rationale.append("Quote quality was poor across candidates; used best-effort pick.")
    return rationale


def _build_option_candidate_from_pick(
    *,
    pick: pd.Series,
    symbol: str,
    option_type: OptionType,
    expiry: date,
    dte: int,
    target_delta: float,
    today: date,
    next_earnings_date: date | None,
    earnings_warn_days: int,
    earnings_avoid_days: int,
    spread_gate_fallback: bool,
    quality_gate_fallback: bool,
) -> OptionCandidate:
    spread_pct = _as_float(pick.get("spread_pct"))
    exec_quality = pick.get("exec_quality")
    delta = _as_float(pick.get("delta"))
    oi = _as_int(pick.get("oi_i"))
    vol = _as_int(pick.get("vol_i"))
    rationale = _build_option_candidate_rationale(
        delta=delta,
        target_delta=target_delta,
        open_interest=oi,
        volume=vol,
        spread_pct=spread_pct,
        execution_quality=exec_quality,
        spread_gate_fallback=spread_gate_fallback,
        quality_gate_fallback=quality_gate_fallback,
    )
    risk = earnings_event_risk(
        today=today,
        expiry=expiry,
        next_earnings_date=next_earnings_date,
        warn_days=earnings_warn_days,
        avoid_days=earnings_avoid_days,
    )
    quality_warnings = pick.get("quality_warnings")
    quality_label = pick.get("quality_label")
    return OptionCandidate(
        symbol=symbol.upper(),
        option_type=option_type,
        expiry=expiry,
        dte=dte,
        strike=float(pick["strike"]),
        mark=_as_float(pick.get("mark")),
        bid=_as_float(pick.get("bid_f")),
        ask=_as_float(pick.get("ask_f")),
        spread=_as_float(pick.get("spread")),
        spread_pct=spread_pct,
        execution_quality=str(exec_quality) if exec_quality is not None else None,
        last=_as_float(pick.get("last_f")),
        iv=_as_float(pick.get("iv_f")),
        delta=delta,
        open_interest=oi,
        volume=vol,
        quality_score=_as_float(pick.get("quality_score")),
        quality_label=str(quality_label) if quality_label is not None else None,
        last_trade_age_days=_as_int(pick.get("last_trade_age_days")),
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
