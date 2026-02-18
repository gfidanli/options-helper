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


def _resolve_snapshot_row(
    *,
    provider: MarketDataProvider | None,
    position: Position,
    snapshot_row: pd.Series | dict | None,
):
    row = snapshot_row
    if row is not None:
        return row
    if provider is None:
        raise ValueError("provider is required when snapshot_row is not provided")
    chain = provider.get_options_chain(position.symbol, position.expiry)
    df = chain.calls if position.option_type == "call" else chain.puts
    return contract_row_by_strike(df, position.strike)


def _extract_quote_fields(row) -> tuple[float | None, float | None, float | None, float | None, int | None, int | None]:  # noqa: ANN001
    if row is None:
        return None, None, None, None, None, None
    return (
        _extract_float(row, "bid"),
        _extract_float(row, "ask"),
        _extract_float(row, "lastPrice"),
        _extract_float(row, "impliedVolatility"),
        _extract_int(row, "openInterest"),
        _extract_int(row, "volume"),
    )


def _extract_quote_quality(
    *,
    row,
    risk_profile: RiskProfile,
    today: date,
) -> tuple[str | None, int | None, list[str]]:  # noqa: ANN001
    if row is None:
        return None, None, []
    quality_df = compute_quote_quality(
        pd.DataFrame([row]),
        min_volume=risk_profile.min_volume,
        min_open_interest=risk_profile.min_open_interest,
        as_of=today,
    )
    if quality_df.empty:
        return None, None, []
    q = quality_df.iloc[0]
    label = q.get("quality_label")
    quality_label = None if label is None or pd.isna(label) else str(label)
    last_trade_age_days = None
    age = q.get("last_trade_age_days")
    if age is not None and not pd.isna(age):
        try:
            last_trade_age_days = int(age)
        except Exception:  # noqa: BLE001
            last_trade_age_days = None
    warnings_val = q.get("quality_warnings")
    quality_warnings = [str(w) for w in warnings_val if w] if isinstance(warnings_val, list) else []
    return quality_label, last_trade_age_days, quality_warnings


def _close_volume_series(underlying_history: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    close_series = None
    volume_series = None
    if not underlying_history.empty and "Close" in underlying_history.columns:
        close_series = underlying_history["Close"].dropna()
    if not underlying_history.empty and "Volume" in underlying_history.columns:
        volume_series = underlying_history["Volume"].dropna()
    return close_series, volume_series


def _valuation_metrics(
    *,
    bid: float | None,
    ask: float | None,
    last: float | None,
    position: Position,
    underlying_last_price: float | None,
    today: date,
    include_pnl: bool,
    cost_basis_override: float | None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    str,
    int,
    float | None,
    float | None,
    float | None,
]:
    mark = _mark_price(bid=bid, ask=ask, last=last)
    spread = spread_pct = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        spread = ask - bid
        mid = (ask + bid) / 2.0
        if mid > 0:
            spread_pct = spread / mid
    exec_quality = execution_quality(spread_pct)
    dte_val = max((position.expiry - today).days, 0)
    underlying_price = underlying_last_price
    moneyness = None if underlying_price is None else underlying_price / position.strike
    pnl_abs = pnl_pct = None
    if mark is not None and include_pnl:
        basis = position.cost_basis if cost_basis_override is None else cost_basis_override
        pnl_abs = (mark - basis) * 100.0 * position.contracts
        pnl_pct = None if basis <= 0 else (mark - basis) / basis
    return mark, spread, spread_pct, exec_quality, dte_val, underlying_price, moneyness, pnl_abs, pnl_pct


def _weekly_breakout(
    *,
    close_w_series: pd.Series,
    volume_series: pd.Series | None,
    option_type: str,
    risk_profile: RiskProfile,
) -> bool | None:
    lookback = risk_profile.breakout_lookback_weeks
    breakout_price = breakout_up(close_w_series, lookback) if option_type == "call" else breakout_down(close_w_series, lookback)
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
    return bool(breakout_price and breakout_vol_ok) if breakout_price is not None else None


def _multi_timeframe_metrics(
    *,
    close_series: pd.Series | None,
    volume_series: pd.Series | None,
    option_type: str,
    risk_profile: RiskProfile,
) -> dict[str, float | bool | None]:
    out: dict[str, float | bool | None] = {
        "close_3d": None,
        "rsi14_3d": None,
        "ema20_3d": None,
        "ema50_3d": None,
        "close_w": None,
        "rsi14_w": None,
        "ema20_w": None,
        "ema50_w": None,
        "near_support_w": None,
        "breakout_w": None,
    }
    if close_series is None or not isinstance(close_series.index, pd.DatetimeIndex):
        return out

    close_3d_series = close_series.resample("3B").last().dropna()
    close_w_series = close_series.resample("W-FRI").last().dropna()
    out["close_3d"] = float(close_3d_series.iloc[-1]) if not close_3d_series.empty else None
    out["close_w"] = float(close_w_series.iloc[-1]) if not close_w_series.empty else None
    out["rsi14_3d"] = rsi(close_3d_series, 14) if not close_3d_series.empty else None
    out["ema20_3d"] = ema(close_3d_series, 20) if not close_3d_series.empty else None
    out["ema50_3d"] = ema(close_3d_series, 50) if not close_3d_series.empty else None
    out["rsi14_w"] = rsi(close_w_series, 14) if not close_w_series.empty else None
    out["ema20_w"] = ema(close_w_series, 20) if not close_w_series.empty else None
    out["ema50_w"] = ema(close_w_series, 50) if not close_w_series.empty else None
    if out["close_w"] is not None and out["ema50_w"] is not None and float(out["ema50_w"]) != 0:
        out["near_support_w"] = abs(float(out["close_w"]) - float(out["ema50_w"])) / abs(float(out["ema50_w"])) <= risk_profile.support_proximity_pct
    out["breakout_w"] = _weekly_breakout(
        close_w_series=close_w_series,
        volume_series=volume_series,
        option_type=option_type,
        risk_profile=risk_profile,
    )
    return out


def _greeks_if_available(
    *,
    option_type: str,
    underlying_price: float | None,
    strike: float,
    dte_val: int,
    iv: float | None,
) -> tuple[float | None, float | None]:
    if underlying_price is None or iv is None or dte_val <= 0:
        return None, None
    greeks = black_scholes_greeks(
        option_type=option_type,
        s=underlying_price,
        k=strike,
        t_years=dte_val / 365.0,
        sigma=iv,
    )
    if greeks is None:
        return None, None
    return greeks.delta, greeks.theta_per_day


def _build_position_metrics_result(
    *,
    position: Position,
    contract_sign: int,
    underlying_price: float | None,
    mark: float | None,
    bid: float | None,
    ask: float | None,
    spread: float | None,
    spread_pct: float | None,
    exec_quality: str,
    last: float | None,
    iv: float | None,
    oi: int | None,
    vol: int | None,
    quality_label: str | None,
    last_trade_age_days: int | None,
    quality_warnings: list[str],
    dte_val: int,
    moneyness: float | None,
    pnl_abs: float | None,
    pnl_pct: float | None,
    sma20: float | None,
    sma50: float | None,
    rsi14: float | None,
    ema20: float | None,
    ema50: float | None,
    mtf: dict[str, float | bool | None],
    delta: float | None,
    theta_per_day: float | None,
    today: date,
    next_earnings_date: date | None,
) -> PositionMetrics:
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
        close_3d=mtf["close_3d"],
        rsi14_3d=mtf["rsi14_3d"],
        ema20_3d=mtf["ema20_3d"],
        ema50_3d=mtf["ema50_3d"],
        close_w=mtf["close_w"],
        rsi14_w=mtf["rsi14_w"],
        ema20_w=mtf["ema20_w"],
        ema50_w=mtf["ema50_w"],
        near_support_w=mtf["near_support_w"],
        breakout_w=mtf["breakout_w"],
        delta=delta,
        theta_per_day=theta_per_day,
        as_of=today,
        next_earnings_date=next_earnings_date,
    )


def _build_position_metrics_from_components(
    *,
    position: Position,
    contract_sign: int,
    today: date,
    next_earnings_date: date | None,
    valuation: tuple[float | None, float | None, float | None, float | None, float | None, str, float | None, float | None, int | None, int | None, str | None, int | None, list[str], int, float | None, float | None, float | None],
    indicators: tuple[float | None, float | None, float | None, float | None, float | None],
    mtf: dict[str, float | bool | None],
    greeks: tuple[float | None, float | None],
) -> PositionMetrics:
    (
        underlying_price,
        mark,
        bid,
        ask,
        spread,
        exec_quality,
        spread_pct,
        last,
        iv,
        oi,
        vol,
        quality_label,
        last_trade_age_days,
        quality_warnings,
        dte_val,
        moneyness,
        pnl_abs,
        pnl_pct,
    ) = valuation
    sma20, sma50, rsi14, ema20, ema50 = indicators
    delta, theta_per_day = greeks
    return _build_position_metrics_result(
        position=position,
        contract_sign=contract_sign,
        underlying_price=underlying_price,
        mark=mark,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_pct=spread_pct,
        exec_quality=exec_quality,
        last=last,
        iv=iv,
        oi=oi,
        vol=vol,
        quality_label=quality_label,
        last_trade_age_days=last_trade_age_days,
        quality_warnings=quality_warnings,
        dte_val=dte_val,
        moneyness=moneyness,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        sma20=sma20,
        sma50=sma50,
        rsi14=rsi14,
        ema20=ema20,
        ema50=ema50,
        mtf=mtf,
        delta=delta,
        theta_per_day=theta_per_day,
        today=today,
        next_earnings_date=next_earnings_date,
    )


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
    row = _resolve_snapshot_row(provider=provider, position=position, snapshot_row=snapshot_row)
    bid, ask, last, iv, oi, vol = _extract_quote_fields(row)
    quality_label, last_trade_age_days, quality_warnings = _extract_quote_quality(row=row, risk_profile=risk_profile, today=today)
    mark, spread, spread_pct, exec_quality, dte_val, underlying_price, moneyness, pnl_abs, pnl_pct = _valuation_metrics(
        bid=bid,
        ask=ask,
        last=last,
        position=position,
        underlying_last_price=underlying_last_price,
        today=today,
        include_pnl=include_pnl,
        cost_basis_override=cost_basis_override,
    )
    close_series, volume_series = _close_volume_series(underlying_history)

    sma20 = sma(close_series, 20) if close_series is not None else None
    sma50 = sma(close_series, 50) if close_series is not None else None
    rsi14 = rsi(close_series, 14) if close_series is not None else None
    ema20 = ema(close_series, 20) if close_series is not None else None
    ema50 = ema(close_series, 50) if close_series is not None else None

    mtf = _multi_timeframe_metrics(
        close_series=close_series,
        volume_series=volume_series,
        option_type=position.option_type,
        risk_profile=risk_profile,
    )
    delta, theta_per_day = _greeks_if_available(
        option_type=position.option_type,
        underlying_price=underlying_price,
        strike=position.strike,
        dte_val=dte_val,
        iv=iv,
    )

    return _build_position_metrics_from_components(
        position=position,
        contract_sign=contract_sign,
        today=today,
        next_earnings_date=next_earnings_date,
        valuation=(
            underlying_price,
            mark,
            bid,
            ask,
            spread,
            exec_quality,
            spread_pct,
            last,
            iv,
            oi,
            vol,
            quality_label,
            last_trade_age_days,
            quality_warnings,
            dte_val,
            moneyness,
            pnl_abs,
            pnl_pct,
        ),
        indicators=(sma20, sma50, rsi14, ema20, ema50),
        mtf=mtf,
        greeks=(delta, theta_per_day),
    )
