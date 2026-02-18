from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from options_helper.analysis.indicators import ema
from options_helper.backtesting.data_source import BacktestDataSource
from options_helper.backtesting.execution import FillMode, fill_price
from options_helper.backtesting.ledger import BacktestLedger, PositionState, TradeLogRow
from options_helper.backtesting.roll import RollPolicy, select_roll_candidate, should_roll
from options_helper.backtesting.strategy import DayContext, PositionContext, Strategy
from options_helper.data.options_snapshots import find_snapshot_row
from options_helper.models import Position
from options_helper.technicals_backtesting.extension_percentiles import rolling_percentile_rank


@dataclass(frozen=True)
class SkipEvent:
    as_of: date
    action: str
    reason: str


@dataclass(frozen=True)
class BacktestRun:
    symbol: str
    contract_symbol: str | None
    start: date | None
    end: date | None
    fill_mode: FillMode
    slippage_factor: float
    initial_cash: float
    trades: list[TradeLogRow]
    skips: list[SkipEvent]
    open_position: PositionState | None
    rolls: list["RollEvent"]


@dataclass(frozen=True)
class RollEvent:
    as_of: date
    from_contract_symbol: str | None
    to_contract_symbol: str | None
    reason: str | None


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _positive_float(value: object) -> float | None:
    val = _as_float(value)
    if val is None or val <= 0:
        return None
    return val


@dataclass(frozen=True)
class _Quote:
    bid: float | None
    ask: float | None
    last: float | None
    mark: float | None
    spread_pct: float | None


def _best_effort_mark(*, bid: float | None, ask: float | None, last: float | None, mark: float | None) -> float | None:
    if mark is not None:
        return mark
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if last is not None:
        return last
    if ask is not None:
        return ask
    if bid is not None:
        return bid
    return None


def _spread_pct_from_bid_ask(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def _quote_from_row(row: dict | None) -> _Quote | None:
    if row is None:
        return None
    bid = _positive_float(row.get("bid"))
    ask = _positive_float(row.get("ask"))
    last = _positive_float(row.get("lastPrice") or row.get("last"))
    mark = _positive_float(row.get("mark"))
    mark = _best_effort_mark(bid=bid, ask=ask, last=last, mark=mark)
    spread_pct = _as_float(row.get("spread_pct"))
    if spread_pct is None:
        spread_pct = _spread_pct_from_bid_ask(bid, ask)
    return _Quote(bid=bid, ask=ask, last=last, mark=mark, spread_pct=spread_pct)


def _slice_candles(candles: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if candles is None or candles.empty:
        return pd.DataFrame()
    if not isinstance(candles.index, pd.DatetimeIndex):
        return pd.DataFrame()
    idx = candles.index
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    cutoff = pd.Timestamp(as_of)
    subset = candles.loc[idx <= cutoff].copy()
    return subset


def _weekly_trend_up(
    candles: pd.DataFrame,
    *,
    resample_rule: str = "W-FRI",
    fast_span: int = 9,
    slow_span: int = 21,
    logic: str = "close_above_fast",
) -> bool | None:
    if candles is None or candles.empty or "Close" not in candles.columns:
        return None
    weekly = candles[["Close"]].resample(resample_rule).last().dropna()
    if weekly.empty:
        return None
    close = weekly["Close"]
    last = float(close.iloc[-1])
    fast = ema(close, fast_span)
    slow = ema(close, slow_span)
    if logic == "close_above_fast_and_fast_above_slow":
        if fast is None or slow is None:
            return None
        return last > fast and fast > slow
    if logic == "fast_above_slow":
        if fast is None or slow is None:
            return None
        return fast > slow
    if logic == "close_above_fast":
        if fast is None:
            return None
        return last > fast
    return None


def _extension_percentile(
    candles: pd.DataFrame,
    *,
    sma_window: int = 20,
    atr_window: int = 14,
    window_days: int = 252,
) -> float | None:
    if candles is None or candles.empty:
        return None
    if not {"High", "Low", "Close"}.issubset(candles.columns):
        return None
    close = candles["Close"].astype(float)
    if len(close) < max(sma_window, atr_window, window_days):
        return None

    sma = close.rolling(window=sma_window).mean()
    high = candles["High"].astype(float)
    low = candles["Low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_window).mean()
    extension = (close - sma) / atr
    extension = extension.dropna()
    if extension.empty:
        return None
    pct_series = rolling_percentile_rank(extension, window=window_days)
    if pct_series.empty:
        return None
    val = pct_series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def _default_day_context_builder(candles: pd.DataFrame, as_of: date) -> DayContext:
    subset = _slice_candles(candles, as_of)
    spot = None
    if not subset.empty and "Close" in subset.columns:
        try:
            spot = float(subset["Close"].iloc[-1])
        except Exception:  # noqa: BLE001
            spot = None
    weekly_trend = _weekly_trend_up(subset)
    extension_pct = _extension_percentile(subset)
    return DayContext(
        as_of=as_of,
        spot=spot,
        weekly_trend_up=weekly_trend,
        extension_percentile=extension_pct,
    )


def _position_id_from_contract(
    symbol: str,
    *,
    contract_symbol: str | None,
    expiry: date | None,
    strike: float | None,
    option_type: str | None,
) -> str:
    if contract_symbol:
        return contract_symbol
    if expiry is None or strike is None or option_type is None:
        return f"{symbol}-unknown"
    suffix = "c" if str(option_type).lower().startswith("c") else "p"
    strike_str = f"{float(strike):g}".replace(".", "p")
    return f"{symbol.lower()}-{expiry.isoformat()}-{strike_str}{suffix}"


def _parse_iso_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        return date.fromisoformat(text)
    except Exception:  # noqa: BLE001
        return None


def _build_position_context(pos: PositionState, *, as_of: date, mark: float | None) -> PositionContext:
    days_held = (as_of - pos.entry_date).days
    pnl_pct = None
    if mark is not None and pos.entry_price > 0:
        pnl_pct = (mark / pos.entry_price) - 1.0
    return PositionContext(
        entry_date=pos.entry_date,
        days_held=days_held,
        entry_price=pos.entry_price,
        mark=mark,
        pnl_pct=pnl_pct,
        max_favorable=pos.max_favorable,
        max_adverse=pos.max_adverse,
    )


@dataclass
class _ContractState:
    position_id: str
    contract_symbol: str | None
    expiry: date | None
    strike: float | None
    option_type: str | None


def _apply_row_contract_updates(state: _ContractState, row_dict: dict | None) -> None:
    if row_dict is None:
        return
    if state.expiry is None:
        state.expiry = _parse_iso_date(row_dict.get("expiry"))
    if state.strike is None:
        state.strike = _as_float(row_dict.get("strike"))
    if state.option_type is None:
        opt = row_dict.get("optionType") or row_dict.get("option_type")
        if opt is not None:
            state.option_type = str(opt).strip().lower()


def _fill_from_quote(
    *,
    side: str,
    quote: _Quote,
    fill_mode: FillMode,
    slippage_factor: float,
    allow_worst_case_mark_fallback: bool,
):
    return fill_price(
        side=side,
        fill_mode=fill_mode,
        bid=quote.bid,
        ask=quote.ask,
        last=quote.last,
        mark=quote.mark,
        spread_pct=quote.spread_pct,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
    )


def _process_entry(
    *,
    ledger: BacktestLedger,
    skips: list[SkipEvent],
    state: _ContractState,
    strategy: Strategy,
    day_ctx: DayContext,
    quote: _Quote | None,
    symbol: str,
    as_of: date,
    fill_mode: FillMode,
    slippage_factor: float,
    allow_worst_case_mark_fallback: bool,
    quantity: int,
) -> None:
    if not strategy.should_enter(day_ctx):
        return
    if quote is None:
        skips.append(SkipEvent(as_of=as_of, action="entry", reason="missing_contract"))
        return
    fill = _fill_from_quote(
        side="buy",
        quote=quote,
        fill_mode=fill_mode,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
    )
    if fill.price is None:
        skips.append(SkipEvent(as_of=as_of, action="entry", reason=fill.reason))
        return
    ledger.open_long(
        state.position_id,
        symbol=symbol,
        contract_symbol=state.contract_symbol,
        expiry=state.expiry or date.min,
        strike=float(state.strike or 0.0),
        option_type=str(state.option_type or "call"),
        quantity=quantity,
        entry_date=as_of,
        entry_price=fill.price,
    )


def _maybe_roll_position(
    *,
    ledger: BacktestLedger,
    state: _ContractState,
    position: PositionState,
    strategy: Strategy,
    day_ctx: DayContext,
    quote: _Quote | None,
    skips: list[SkipEvent],
    rolls: list[RollEvent],
    roll_policy: RollPolicy | None,
    data_source: BacktestDataSource,
    symbol: str,
    as_of: date,
    fill_mode: FillMode,
    slippage_factor: float,
    allow_worst_case_mark_fallback: bool,
) -> bool:
    if roll_policy is None:
        return False
    thesis_ok = strategy.should_enter(day_ctx)
    if not should_roll(as_of=as_of, expiry=position.expiry, dte_threshold=roll_policy.dte_threshold, thesis_ok=thesis_ok):
        return False
    if day_ctx.spot is None:
        skips.append(SkipEvent(as_of=as_of, action="roll", reason="missing_spot"))
        return False
    roll_decision = select_roll_candidate(
        data_source.snapshot_store,
        symbol=symbol,
        as_of=as_of,
        spot=day_ctx.spot,
        position=Position(
            id=state.position_id,
            symbol=symbol,
            option_type=str(position.option_type),
            expiry=position.expiry,
            strike=position.strike,
            contracts=position.quantity,
            cost_basis=position.entry_price,
            opened_at=position.entry_date,
        ),
        policy=roll_policy,
    )
    if roll_decision.candidate is None:
        skips.append(SkipEvent(as_of=as_of, action="roll", reason=roll_decision.reason or "no_candidate"))
        return False
    return _execute_roll(
        ledger=ledger,
        state=state,
        position=position,
        quote=quote,
        skips=skips,
        rolls=rolls,
        roll_decision=roll_decision,
        as_of=as_of,
        symbol=symbol,
        fill_mode=fill_mode,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
    )


def _execute_roll(
    *,
    ledger: BacktestLedger,
    state: _ContractState,
    position: PositionState,
    quote: _Quote | None,
    skips: list[SkipEvent],
    rolls: list[RollEvent],
    roll_decision,
    as_of: date,
    symbol: str,
    fill_mode: FillMode,
    slippage_factor: float,
    allow_worst_case_mark_fallback: bool,
) -> bool:
    if quote is None:
        skips.append(SkipEvent(as_of=as_of, action="roll", reason="missing_contract"))
        return False
    exit_fill = _fill_from_quote(
        side="sell",
        quote=quote,
        fill_mode=fill_mode,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
    )
    candidate = roll_decision.candidate.contract
    candidate_expiry = _parse_iso_date(candidate.expiry)
    entry_fill = fill_price(
        side="buy",
        fill_mode=fill_mode,
        bid=candidate.bid,
        ask=candidate.ask,
        last=None,
        mark=candidate.mark,
        spread_pct=candidate.spread_pct,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
    )
    if candidate_expiry is None:
        skips.append(SkipEvent(as_of=as_of, action="roll", reason="invalid_candidate_expiry"))
        return False
    if exit_fill.price is None:
        skips.append(SkipEvent(as_of=as_of, action="roll", reason=exit_fill.reason))
        return False
    if entry_fill.price is None:
        skips.append(SkipEvent(as_of=as_of, action="roll", reason=entry_fill.reason))
        return False
    ledger.close(state.position_id, exit_date=as_of, exit_price=exit_fill.price)
    state.contract_symbol = candidate.contract_symbol
    state.expiry = candidate_expiry
    state.strike = candidate.strike
    state.option_type = candidate.option_type
    state.position_id = _position_id_from_contract(
        symbol,
        contract_symbol=state.contract_symbol,
        expiry=state.expiry,
        strike=state.strike,
        option_type=state.option_type,
    )
    ledger.open_long(
        state.position_id,
        symbol=symbol,
        contract_symbol=state.contract_symbol,
        expiry=candidate_expiry,
        strike=float(candidate.strike),
        option_type=str(candidate.option_type),
        quantity=position.quantity,
        entry_date=as_of,
        entry_price=entry_fill.price,
    )
    rolls.append(
        RollEvent(
            as_of=as_of,
            from_contract_symbol=position.contract_symbol,
            to_contract_symbol=state.contract_symbol,
            reason=roll_decision.reason,
        )
    )
    return True


def _process_exit(
    *,
    ledger: BacktestLedger,
    skips: list[SkipEvent],
    position_id: str,
    strategy: Strategy,
    pos_ctx: PositionContext,
    quote: _Quote | None,
    as_of: date,
    fill_mode: FillMode,
    slippage_factor: float,
    allow_worst_case_mark_fallback: bool,
) -> None:
    if not strategy.should_exit(pos_ctx):
        return
    if quote is None:
        skips.append(SkipEvent(as_of=as_of, action="exit", reason="missing_contract"))
        return
    fill = _fill_from_quote(
        side="sell",
        quote=quote,
        fill_mode=fill_mode,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
    )
    if fill.price is None:
        skips.append(SkipEvent(as_of=as_of, action="exit", reason=fill.reason))
        return
    ledger.close(position_id, exit_date=as_of, exit_price=fill.price)


def _run_backtest_days(
    *,
    data_source: BacktestDataSource,
    symbol: str,
    start: date | None,
    end: date | None,
    state: _ContractState,
    ledger: BacktestLedger,
    candles: pd.DataFrame,
    ctx_builder: Callable[[pd.DataFrame, date], DayContext],
    strategy: Strategy,
    skips: list[SkipEvent],
    rolls: list[RollEvent],
    fill_mode: FillMode,
    slippage_factor: float,
    allow_worst_case_mark_fallback: bool,
    quantity: int,
    roll_policy: RollPolicy | None,
) -> None:
    for as_of, day_df in data_source.iter_snapshot_days(symbol, start=start, end=end):
        day_ctx = ctx_builder(candles, as_of)
        row = find_snapshot_row(
            day_df,
            expiry=state.expiry or date.min,
            strike=state.strike or 0.0,
            option_type=state.option_type or "call",
            contract_symbol=state.contract_symbol,
        )
        row_dict = row.to_dict() if row is not None else None
        _apply_row_contract_updates(state, row_dict)
        quote = _quote_from_row(row_dict)
        if state.position_id not in ledger.positions:
            _process_entry(
                ledger=ledger,
                skips=skips,
                state=state,
                strategy=strategy,
                day_ctx=day_ctx,
                quote=quote,
                symbol=symbol,
                as_of=as_of,
                fill_mode=fill_mode,
                slippage_factor=slippage_factor,
                allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
                quantity=quantity,
            )
            continue
        position = ledger.positions[state.position_id]
        ledger.update_mark(state.position_id, mark=quote.mark if quote is not None else None)
        pos_ctx = _build_position_context(position, as_of=as_of, mark=quote.mark if quote else None)
        if _maybe_roll_position(
            ledger=ledger,
            state=state,
            position=position,
            strategy=strategy,
            day_ctx=day_ctx,
            quote=quote,
            skips=skips,
            rolls=rolls,
            roll_policy=roll_policy,
            data_source=data_source,
            symbol=symbol,
            as_of=as_of,
            fill_mode=fill_mode,
            slippage_factor=slippage_factor,
            allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
        ):
            continue
        _process_exit(
            ledger=ledger,
            skips=skips,
            position_id=state.position_id,
            strategy=strategy,
            pos_ctx=pos_ctx,
            quote=quote,
            as_of=as_of,
            fill_mode=fill_mode,
            slippage_factor=slippage_factor,
            allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
        )


def run_backtest(
    data_source: BacktestDataSource,
    *,
    symbol: str,
    contract_symbol: str | None = None,
    expiry: date | None = None,
    strike: float | None = None,
    option_type: str | None = None,
    strategy: Strategy,
    start: date | None = None,
    end: date | None = None,
    fill_mode: FillMode = "worst_case",
    slippage_factor: float = 0.5,
    allow_worst_case_mark_fallback: bool = False,
    initial_cash: float = 10000.0,
    quantity: int = 1,
    day_context_builder: Callable[[pd.DataFrame, date], DayContext] | None = None,
    roll_policy: RollPolicy | None = None,
) -> BacktestRun:
    if contract_symbol is None and (expiry is None or strike is None or option_type is None):
        raise ValueError("contract_symbol or (expiry, strike, option_type) is required")

    candles = data_source.load_candles(symbol)
    ctx_builder = day_context_builder or _default_day_context_builder

    ledger = BacktestLedger(cash=initial_cash)
    skips: list[SkipEvent] = []
    rolls: list[RollEvent] = []

    state = _ContractState(
        position_id=_position_id_from_contract(
            symbol,
            contract_symbol=contract_symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
        ),
        contract_symbol=contract_symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    )
    _run_backtest_days(
        data_source=data_source,
        symbol=symbol,
        start=start,
        end=end,
        state=state,
        ledger=ledger,
        candles=candles,
        ctx_builder=ctx_builder,
        strategy=strategy,
        skips=skips,
        rolls=rolls,
        fill_mode=fill_mode,
        slippage_factor=slippage_factor,
        allow_worst_case_mark_fallback=allow_worst_case_mark_fallback,
        quantity=quantity,
        roll_policy=roll_policy,
    )

    open_position = ledger.positions.get(state.position_id)
    return BacktestRun(
        symbol=symbol,
        contract_symbol=contract_symbol,
        start=start,
        end=end,
        fill_mode=fill_mode,
        slippage_factor=slippage_factor,
        initial_cash=initial_cash,
        trades=list(ledger.trades),
        skips=skips,
        open_position=open_position,
        rolls=rolls,
    )
