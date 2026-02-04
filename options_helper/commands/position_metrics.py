from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from options_helper.analysis.advice import PositionMetrics
from options_helper.analysis.chain_metrics import execution_quality
from options_helper.analysis.greeks import black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, sma
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.data.providers.base import MarketDataProvider
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.models import Position, RiskProfile

if TYPE_CHECKING:
    import pandas as pd


pd: object | None = None


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def _extract_float(row, key: str) -> float | None:  # noqa: ANN001
    if key not in row:
        return None
    val = row[key]
    try:
        if val is None:
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _extract_int(row, key: str) -> int | None:  # noqa: ANN001
    if key not in row:
        return None
    val = row[key]
    try:
        if val is None:
            return None
        return int(val)
    except Exception:  # noqa: BLE001
        return None


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


def _position_metrics(
    provider: MarketDataProvider | None,
    position: Position,
    *,
    risk_profile: RiskProfile,
    underlying_history: pd.DataFrame,
    underlying_last_price: float | None,
    as_of: date | None = None,
    next_earnings_date: date | None = None,
    snapshot_row: pd.Series | dict | None = None,
    cost_basis_override: float | None = None,
    include_pnl: bool = True,
    contract_sign: int = 1,
) -> PositionMetrics:
    _ensure_pandas()
    today = as_of or date.today()

    row = snapshot_row
    if row is None:
        if provider is None:
            raise ValueError("provider is required when snapshot_row is not provided")
        chain = provider.get_options_chain(position.symbol, position.expiry)
        df = chain.calls if position.option_type == "call" else chain.puts
        row = contract_row_by_strike(df, position.strike)

    bid = ask = last = iv = None
    oi = vol = None
    if row is not None:
        bid = _extract_float(row, "bid")
        ask = _extract_float(row, "ask")
        last = _extract_float(row, "lastPrice")
        iv = _extract_float(row, "impliedVolatility")
        oi = _extract_int(row, "openInterest")
        vol = _extract_int(row, "volume")

    quality_label = None
    last_trade_age_days = None
    quality_warnings: list[str] = []
    if row is not None:
        quality_df = compute_quote_quality(
            pd.DataFrame([row]),
            min_volume=risk_profile.min_volume,
            min_open_interest=risk_profile.min_open_interest,
            as_of=today,
        )
        if not quality_df.empty:
            q = quality_df.iloc[0]
            label = q.get("quality_label")
            if label is not None and not pd.isna(label):
                quality_label = str(label)
            age = q.get("last_trade_age_days")
            if age is not None and not pd.isna(age):
                try:
                    last_trade_age_days = int(age)
                except Exception:  # noqa: BLE001
                    last_trade_age_days = None
            warnings_val = q.get("quality_warnings")
            if isinstance(warnings_val, list):
                quality_warnings = [str(w) for w in warnings_val if w]

    mark = _mark_price(bid=bid, ask=ask, last=last)
    spread = spread_pct = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        spread = ask - bid
        mid = (ask + bid) / 2.0
        if mid > 0:
            spread_pct = spread / mid
    exec_quality = execution_quality(spread_pct)

    dte = (position.expiry - today).days
    dte_val = dte if dte >= 0 else 0

    underlying_price = underlying_last_price
    moneyness = None
    if underlying_price is not None:
        moneyness = underlying_price / position.strike

    pnl_abs = pnl_pct = None
    if mark is not None and include_pnl:
        basis = position.cost_basis if cost_basis_override is None else cost_basis_override
        pnl_abs = (mark - basis) * 100.0 * position.contracts
        pnl_pct = None if basis <= 0 else (mark - basis) / basis

    close_series: pd.Series | None = None
    volume_series: pd.Series | None = None
    if not underlying_history.empty and "Close" in underlying_history.columns:
        close_series = underlying_history["Close"].dropna()
    if not underlying_history.empty and "Volume" in underlying_history.columns:
        volume_series = underlying_history["Volume"].dropna()

    sma20 = sma(close_series, 20) if close_series is not None else None
    sma50 = sma(close_series, 50) if close_series is not None else None
    rsi14 = rsi(close_series, 14) if close_series is not None else None
    ema20 = ema(close_series, 20) if close_series is not None else None
    ema50 = ema(close_series, 50) if close_series is not None else None

    close_3d = rsi14_3d = ema20_3d = ema50_3d = None
    close_w = rsi14_w = ema20_w = ema50_w = None
    breakout_w = None
    near_support_w = None

    if close_series is not None and isinstance(close_series.index, pd.DatetimeIndex):
        close_3d_series = close_series.resample("3B").last().dropna()
        close_w_series = close_series.resample("W-FRI").last().dropna()

        close_3d = float(close_3d_series.iloc[-1]) if not close_3d_series.empty else None
        close_w = float(close_w_series.iloc[-1]) if not close_w_series.empty else None

        rsi14_3d = rsi(close_3d_series, 14) if not close_3d_series.empty else None
        ema20_3d = ema(close_3d_series, 20) if not close_3d_series.empty else None
        ema50_3d = ema(close_3d_series, 50) if not close_3d_series.empty else None

        rsi14_w = rsi(close_w_series, 14) if not close_w_series.empty else None
        ema20_w = ema(close_w_series, 20) if not close_w_series.empty else None
        ema50_w = ema(close_w_series, 50) if not close_w_series.empty else None

        if close_w is not None and ema50_w is not None and ema50_w != 0:
            near_support_w = abs(close_w - ema50_w) / abs(ema50_w) <= risk_profile.support_proximity_pct

        lookback = risk_profile.breakout_lookback_weeks
        breakout_price = (
            breakout_up(close_w_series, lookback)
            if position.option_type == "call"
            else breakout_down(close_w_series, lookback)
        )

        breakout_vol_ok = True
        if (
            breakout_price is True
            and volume_series is not None
            and isinstance(volume_series.index, pd.DatetimeIndex)
            and risk_profile.breakout_volume_mult > 0
        ):
            vol_w_series = volume_series.resample("W-FRI").sum().dropna()
            if len(vol_w_series) >= lookback + 1:
                last_vol = float(vol_w_series.iloc[-1])
                prev_avg = float(vol_w_series.iloc[-(lookback + 1) : -1].mean())
                if prev_avg > 0:
                    breakout_vol_ok = last_vol >= prev_avg * risk_profile.breakout_volume_mult

        if breakout_price is not None:
            breakout_w = bool(breakout_price and breakout_vol_ok)

    delta = theta_per_day = None
    if underlying_price is not None and iv is not None and dte_val > 0:
        greeks = black_scholes_greeks(
            option_type=position.option_type,
            s=underlying_price,
            k=position.strike,
            t_years=dte_val / 365.0,
            sigma=iv,
        )
        if greeks is not None:
            delta = greeks.delta
            theta_per_day = greeks.theta_per_day

    return PositionMetrics(
        position=position,
        contract_sign=contract_sign,
        underlying_price=underlying_price,
        mark=mark,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_pct=spread_pct,
        execution_quality=exec_quality,
        last=last,
        implied_vol=iv,
        open_interest=oi,
        volume=vol,
        quality_label=quality_label,
        last_trade_age_days=last_trade_age_days,
        quality_warnings=quality_warnings,
        dte=dte_val,
        moneyness=moneyness,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        sma20=sma20,
        sma50=sma50,
        rsi14=rsi14,
        ema20=ema20,
        ema50=ema50,
        close_3d=close_3d,
        rsi14_3d=rsi14_3d,
        ema20_3d=ema20_3d,
        ema50_3d=ema50_3d,
        close_w=close_w,
        rsi14_w=rsi14_w,
        ema20_w=ema20_w,
        ema50_w=ema50_w,
        near_support_w=near_support_w,
        breakout_w=breakout_w,
        delta=delta,
        theta_per_day=theta_per_day,
        as_of=today,
        next_earnings_date=next_earnings_date,
    )
