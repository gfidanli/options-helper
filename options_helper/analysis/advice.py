from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from options_helper.analysis.events import earnings_event_risk, format_next_earnings_line
from options_helper.models import Portfolio, Position, RiskProfile


class Action(str, Enum):
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    ADD = "ADD"
    REDUCE = "REDUCE"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ThesisState(str, Enum):
    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    ADVERSE = "adverse"


@dataclass(frozen=True)
class PositionMetrics:
    position: Position

    underlying_price: float | None
    mark: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    spread_pct: float | None
    execution_quality: str | None
    last: float | None
    implied_vol: float | None
    open_interest: int | None
    volume: int | None
    quality_label: str | None
    last_trade_age_days: int | None
    quality_warnings: list[str]

    dte: int | None
    moneyness: float | None

    pnl_abs: float | None
    pnl_pct: float | None

    # Daily technicals (derived from daily closes)
    sma20: float | None
    sma50: float | None
    rsi14: float | None
    ema20: float | None = None
    ema50: float | None = None

    # 3-business-day technicals (derived from daily closes)
    close_3d: float | None = None
    rsi14_3d: float | None = None
    ema20_3d: float | None = None
    ema50_3d: float | None = None

    # Weekly technicals (derived from daily closes)
    close_w: float | None = None
    rsi14_w: float | None = None
    ema20_w: float | None = None
    ema50_w: float | None = None
    near_support_w: bool | None = None
    breakout_w: bool | None = None

    delta: float | None = None
    theta_per_day: float | None = None
    as_of: date | None = None
    next_earnings_date: date | None = None
    contract_sign: int = 1


@dataclass(frozen=True)
class Advice:
    action: Action
    confidence: Confidence
    reasons: list[str]
    warnings: list[str]


def _thesis_state(
    *,
    option_type: str,
    close: float | None,
    ema20: float | None,
    ema50: float | None,
) -> ThesisState | None:
    if close is None or ema20 is None or ema50 is None:
        return None

    # Underlying direction:
    bullish = (close > ema50) and (ema20 >= ema50)
    bearish = (close < ema50) and (ema20 <= ema50)

    if option_type == "call":
        if bullish:
            return ThesisState.SUPPORTIVE
        if bearish:
            return ThesisState.ADVERSE
        return ThesisState.NEUTRAL

    # put
    if bearish:
        return ThesisState.SUPPORTIVE
    if bullish:
        return ThesisState.ADVERSE
    return ThesisState.NEUTRAL


def _momentum_state(*, option_type: str, rsi_val: float | None) -> ThesisState | None:
    if rsi_val is None:
        return None

    # Calls prefer higher RSI (positive momentum); puts prefer lower RSI.
    if option_type == "call":
        if rsi_val >= 55.0:
            return ThesisState.SUPPORTIVE
        if rsi_val <= 45.0:
            return ThesisState.ADVERSE
        return ThesisState.NEUTRAL

    # put
    if rsi_val <= 45.0:
        return ThesisState.SUPPORTIVE
    if rsi_val >= 55.0:
        return ThesisState.ADVERSE
    return ThesisState.NEUTRAL


def _state_score(state: ThesisState | None) -> int:
    if state is None:
        return 0
    if state == ThesisState.SUPPORTIVE:
        return 1
    if state == ThesisState.ADVERSE:
        return -1
    return 0


def _risk_budget(portfolio: Portfolio) -> tuple[float, float]:
    capital = portfolio.capital_cost_basis()
    total_risk = portfolio.premium_at_risk()
    return capital, total_risk


def advise(metrics: PositionMetrics, portfolio: Portfolio) -> Advice:
    rp: RiskProfile = portfolio.risk_profile
    reasons: list[str] = []
    warnings: list[str] = []

    as_of = metrics.as_of or date.today()
    reasons.append(format_next_earnings_line(as_of, metrics.next_earnings_date))
    event_risk = earnings_event_risk(
        today=as_of,
        expiry=metrics.position.expiry,
        next_earnings_date=metrics.next_earnings_date,
        warn_days=rp.earnings_warn_days,
        avoid_days=rp.earnings_avoid_days,
    )
    warnings.extend(event_risk["warnings"])

    if metrics.mark is None:
        warnings.append("Missing option mark price; advice is limited.")
    if metrics.execution_quality == "unknown":
        warnings.append("Missing bid/ask quotes; quote quality low.")
    if metrics.execution_quality == "bad" and metrics.spread_pct is not None:
        warnings.append(f"Wide spread ({metrics.spread_pct:.1%}); fills may be poor.")
    if metrics.quality_warnings:
        for w in metrics.quality_warnings:
            if w not in warnings:
                warnings.append(w)

    if metrics.open_interest is not None and metrics.open_interest < rp.min_open_interest:
        warnings.append(f"Low open interest ({metrics.open_interest} < {rp.min_open_interest}).")
    if metrics.volume is not None and metrics.volume < rp.min_volume:
        warnings.append(f"Low volume ({metrics.volume} < {rp.min_volume}).")

    capital, total_risk = _risk_budget(portfolio)
    pos_risk = metrics.position.premium_paid

    action = Action.HOLD
    confidence = Confidence.MEDIUM

    # Risk gating
    if capital > 0:
        if rp.max_single_position_risk_pct is not None:
            if pos_risk > rp.max_single_position_risk_pct * capital:
                reasons.append("Premium at risk exceeds single-position risk limit.")
                action = Action.CLOSE if metrics.position.contracts == 1 else Action.REDUCE
                confidence = Confidence.HIGH
        if rp.max_portfolio_risk_pct is not None:
            if total_risk > rp.max_portfolio_risk_pct * capital:
                warnings.append("Total premium at risk exceeds portfolio risk limit; avoid adding risk.")

    option_type = metrics.position.option_type

    daily_trend = _thesis_state(
        option_type=option_type, close=metrics.underlying_price, ema20=metrics.ema20, ema50=metrics.ema50
    )
    three_trend = _thesis_state(
        option_type=option_type, close=metrics.close_3d, ema20=metrics.ema20_3d, ema50=metrics.ema50_3d
    )
    weekly_trend = _thesis_state(
        option_type=option_type, close=metrics.close_w, ema20=metrics.ema20_w, ema50=metrics.ema50_w
    )

    daily_mom = _momentum_state(option_type=option_type, rsi_val=metrics.rsi14)
    weekly_mom = _momentum_state(option_type=option_type, rsi_val=metrics.rsi14_w)

    tech_line = f"Technicals: W={weekly_trend.value if weekly_trend else 'n/a'}"
    tech_line += f" (RSI {metrics.rsi14_w:.0f})" if metrics.rsi14_w is not None else " (RSI n/a)"
    tech_line += f", 3D={three_trend.value if three_trend else 'n/a'}"
    tech_line += f", D={daily_trend.value if daily_trend else 'n/a'}"
    if metrics.near_support_w is True:
        tech_line += ", W_support=yes"
    if metrics.breakout_w is True:
        tech_line += ", W_breakout=yes"
    reasons.append(tech_line)

    # Roll / close near expiry (time-based rule still matters for long options)
    if metrics.dte is not None and metrics.dte <= rp.roll_dte_threshold:
        if weekly_trend == ThesisState.SUPPORTIVE:
            reasons.append(f"DTE is low ({metrics.dte} <= {rp.roll_dte_threshold}); trend still supports thesis.")
            action = Action.ROLL
            confidence = Confidence.MEDIUM if confidence != Confidence.HIGH else confidence
        elif weekly_trend == ThesisState.ADVERSE:
            reasons.append(f"DTE is low ({metrics.dte} <= {rp.roll_dte_threshold}); weekly trend is against thesis.")
            action = Action.CLOSE
            confidence = Confidence.HIGH
        else:
            reasons.append(f"DTE is low ({metrics.dte} <= {rp.roll_dte_threshold}); insufficient trend data.")
            action = Action.ROLL
            confidence = Confidence.LOW

    # Multi-timeframe, long-term oriented guidance:
    # - Weekly controls the thesis.
    # - 3D/Daily confirm (or warn) on timing.
    if action == Action.HOLD:
        score = (3 * _state_score(weekly_trend)) + (2 * _state_score(three_trend)) + (1 * _state_score(daily_trend))

        # PnL-based exits are only used when configured, and only when the thesis looks broken.
        if rp.stop_loss_pct is not None and metrics.pnl_pct is not None and metrics.pnl_pct <= -rp.stop_loss_pct:
            if weekly_trend == ThesisState.ADVERSE and three_trend == ThesisState.ADVERSE:
                reasons.append(
                    f"PnL breached stop-loss ({metrics.pnl_pct:.0%} <= -{rp.stop_loss_pct:.0%}) and trends broke down."
                )
                action = Action.CLOSE if metrics.position.contracts == 1 else Action.REDUCE
                confidence = Confidence.HIGH

        if action == Action.HOLD:
            if weekly_trend == ThesisState.ADVERSE and (three_trend == ThesisState.ADVERSE or daily_trend == ThesisState.ADVERSE):
                if metrics.near_support_w:
                    reasons.append("Weekly trend is weak, but price is near a weekly MA support zone; monitor closely.")
                    action = Action.HOLD
                    confidence = Confidence.LOW
                else:
                    reasons.append("Weekly trend is against thesis and lower timeframes confirm weakness.")
                    action = Action.CLOSE if metrics.position.contracts == 1 else Action.REDUCE
                    confidence = Confidence.HIGH
            elif weekly_trend == ThesisState.SUPPORTIVE:
                strong_momentum = weekly_mom == ThesisState.SUPPORTIVE or daily_mom == ThesisState.SUPPORTIVE
                if metrics.breakout_w is True and strong_momentum and score >= 3:
                    reasons.append("Weekly breakout with momentum; adding risk is acceptable per settings.")
                    action = Action.ADD
                    confidence = Confidence.MEDIUM
                else:
                    reasons.append("Weekly trend supports thesis; prefer holding through drawdowns (add on breakout/momentum).")
                    action = Action.HOLD
                    confidence = Confidence.MEDIUM
            else:
                # Weekly neutral/unknown
                if metrics.dte is not None and metrics.dte <= 120 and score <= -2:
                    reasons.append("Shorter-dated contract and multi-timeframe weakness; consider trimming risk.")
                    action = Action.CLOSE if metrics.position.contracts == 1 else Action.REDUCE
                    confidence = Confidence.MEDIUM
                else:
                    reasons.append("Weekly thesis unclear; let higher timeframe decide, monitor daily/3D for deterioration.")
                    action = Action.HOLD
                    confidence = Confidence.LOW

    # Take profit (optional) — primarily for shorter-term moves or momentum exhaustion.
    if (
        action == Action.HOLD
        and rp.take_profit_pct is not None
        and metrics.pnl_pct is not None
        and metrics.pnl_pct >= rp.take_profit_pct
    ):
        if weekly_mom == ThesisState.ADVERSE or daily_mom == ThesisState.ADVERSE:
            reasons.append(f"PnL exceeded take-profit ({metrics.pnl_pct:.0%} >= {rp.take_profit_pct:.0%}) with weakening momentum.")
            action = Action.CLOSE if metrics.position.contracts == 1 else Action.REDUCE
            confidence = Confidence.MEDIUM

    # If we recommend adding, optionally enforce risk budgets (when configured).
    if action == Action.ADD and capital > 0:
        if rp.max_portfolio_risk_pct is not None and total_risk > rp.max_portfolio_risk_pct * capital:
            warnings.append("Risk budgets are exceeded; suppressing ADD.")
            action = Action.HOLD
            confidence = Confidence.LOW
        if rp.max_single_position_risk_pct is not None and pos_risk > rp.max_single_position_risk_pct * capital:
            warnings.append("Single-position risk budget exceeded; suppressing ADD.")
            action = Action.HOLD
            confidence = Confidence.LOW

    if metrics.quality_label in {"bad", "unknown"}:
        if confidence == Confidence.HIGH:
            confidence = Confidence.MEDIUM
        elif confidence == Confidence.MEDIUM:
            confidence = Confidence.LOW
    elif metrics.execution_quality == "bad":
        if confidence == Confidence.HIGH:
            confidence = Confidence.MEDIUM
        elif confidence == Confidence.MEDIUM:
            confidence = Confidence.LOW

    if not reasons:
        reasons.append("No strong triggers; continue monitoring.")

    return Advice(action=action, confidence=confidence, reasons=reasons, warnings=warnings)
