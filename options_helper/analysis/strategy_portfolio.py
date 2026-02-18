from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from typing import Any, Iterable, Literal, Mapping

import pandas as pd

from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig

_EPSILON = 1e-12
_FAR_FUTURE_TS = pd.Timestamp("2262-04-11 23:47:16.854775+00:00")
_TARGET_R_TOLERANCE = 1e-6

try:  # T1 contract models may not be present on intermediate branches.
    from options_helper.schemas.strategy_modeling_contracts import (
        StrategyEquityPoint as _ContractStrategyEquityPoint,
        StrategyTradeSimulation as _ContractStrategyTradeSimulation,
    )

    _HAS_CONTRACT_MODELS = True
except Exception:  # noqa: BLE001
    _HAS_CONTRACT_MODELS = False
    _ContractStrategyEquityPoint = None
    _ContractStrategyTradeSimulation = None

PortfolioLedgerEvent = Literal["entry", "exit", "skip"]
PortfolioSkipReason = Literal[
    "invalid_trade_fill",
    "insufficient_cash",
    "max_concurrent_positions",
    "non_closed_trade_status",
    "one_open_per_symbol",
    "risk_budget_too_small",
]


@dataclass(frozen=True)
class StrategyPortfolioLedgerRow:
    ts: datetime
    trade_id: str
    symbol: str
    event: PortfolioLedgerEvent
    quantity: int
    price: float | None
    cash_after: float
    equity_after: float
    open_trade_count: int
    closed_trade_count: int
    risk_budget: float | None = None
    risk_amount: float | None = None
    realized_pnl: float | None = None
    skip_reason: PortfolioSkipReason | None = None


@dataclass(frozen=True)
class StrategyPortfolioLedgerResult:
    starting_capital: float
    ending_cash: float
    ending_equity: float
    ledger: tuple[StrategyPortfolioLedgerRow, ...]
    equity_curve: tuple[Any, ...]
    accepted_trade_ids: tuple[str, ...]
    skipped_trade_ids: tuple[str, ...]


@dataclass(frozen=True)
class StrategyPortfolioTargetSubset:
    target_label: str | None
    target_r: float | None
    selection_source: str
    trade_ids: tuple[str, ...]


@dataclass(frozen=True)
class _OpenPosition:
    trade_id: str
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    quantity: int
    entry_price: float
    exit_price: float
    reserved_notional: float
    pnl: float


@dataclass(frozen=True)
class _TradeRecord:
    trade_id: str
    symbol: str
    direction: str
    signal_confirmed_ts: object
    entry_ts: object
    exit_ts: object | None
    status: str
    entry_price: object
    target_price: object | None
    exit_price: object | None
    initial_risk: object


@dataclass(frozen=True)
class StrategyPortfolioEquityPoint:
    ts: datetime
    equity: float
    cash: float | None = None
    drawdown_pct: float | None = None
    open_trade_count: int = 0
    closed_trade_count: int = 0


@dataclass
class _PortfolioLedgerState:
    cash: float
    closed_trade_count: int = 0
    open_positions: dict[str, _OpenPosition] = field(default_factory=dict)
    accepted_trade_ids: list[str] = field(default_factory=list)
    skipped_trade_ids: list[str] = field(default_factory=list)
    ledger: list[StrategyPortfolioLedgerRow] = field(default_factory=list)
    equity_curve: list[Any] = field(default_factory=list)
    equity_peak: float = 0.0


def build_strategy_portfolio_ledger(
    trades: Iterable[Mapping[str, Any] | object],
    *,
    starting_capital: float,
    policy: StrategyModelingPolicyConfig | None = None,
    max_concurrent_positions: int | None = None,
) -> StrategyPortfolioLedgerResult:
    """Construct deterministic portfolio ledger/equity curve from simulated trades."""
    capital, cfg, max_positions = _resolve_portfolio_config(
        starting_capital=starting_capital,
        policy=policy,
        max_concurrent_positions=max_concurrent_positions,
    )
    sorted_trades = sorted(_parse_trade_simulations(trades), key=_trade_sort_key)
    state = _PortfolioLedgerState(cash=capital, equity_peak=capital)
    risk_fraction = float(cfg.risk_per_trade_pct) / 100.0
    for trade in sorted_trades:
        _process_portfolio_trade(
            trade=trade,
            state=state,
            cfg=cfg,
            max_positions=max_positions,
            risk_fraction=risk_fraction,
        )
    _close_positions_through(state, ts=_FAR_FUTURE_TS)
    return _build_portfolio_ledger_result(state=state, capital=capital)


def _resolve_portfolio_config(
    *,
    starting_capital: float,
    policy: StrategyModelingPolicyConfig | None,
    max_concurrent_positions: int | None,
) -> tuple[float, StrategyModelingPolicyConfig, int | None]:
    capital = float(starting_capital)
    if not math.isfinite(capital) or capital <= 0.0:
        raise ValueError("starting_capital must be > 0")

    cfg = policy or StrategyModelingPolicyConfig()
    if cfg.sizing_rule != "risk_pct_of_equity":
        raise ValueError(f"Unsupported sizing_rule: {cfg.sizing_rule}")

    max_positions: int | None = None
    if max_concurrent_positions is not None:
        max_positions = int(max_concurrent_positions)
        if max_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")
    return capital, cfg, max_positions


def _state_current_equity(state: _PortfolioLedgerState) -> float:
    return state.cash + sum(position.reserved_notional for position in state.open_positions.values())


def _append_state_point(state: _PortfolioLedgerState, ts: pd.Timestamp) -> None:
    equity_value = _state_current_equity(state)
    state.equity_peak = max(state.equity_peak, equity_value)
    drawdown_pct = 0.0
    if state.equity_peak > _EPSILON:
        drawdown_pct = (equity_value / state.equity_peak) - 1.0

    state.equity_curve.append(
        _build_equity_point(
            ts=ts.to_pydatetime(),
            equity=equity_value,
            cash=state.cash,
            drawdown_pct=drawdown_pct,
            open_trade_count=len(state.open_positions),
            closed_trade_count=state.closed_trade_count,
        )
    )


def _append_ledger_row(
    state: _PortfolioLedgerState,
    *,
    ts: pd.Timestamp,
    trade_id: str,
    symbol: str,
    event: PortfolioLedgerEvent,
    quantity: int,
    price: float | None,
    risk_budget: float | None = None,
    risk_amount: float | None = None,
    realized_pnl: float | None = None,
    skip_reason: PortfolioSkipReason | None = None,
) -> None:
    state.ledger.append(
        StrategyPortfolioLedgerRow(
            ts=ts.to_pydatetime(),
            trade_id=trade_id,
            symbol=symbol,
            event=event,
            quantity=int(quantity),
            price=price,
            cash_after=state.cash,
            equity_after=_state_current_equity(state),
            open_trade_count=len(state.open_positions),
            closed_trade_count=state.closed_trade_count,
            risk_budget=risk_budget,
            risk_amount=risk_amount,
            realized_pnl=realized_pnl,
            skip_reason=skip_reason,
        )
    )
    _append_state_point(state, ts)


def _append_skip_row(
    state: _PortfolioLedgerState,
    *,
    ts: pd.Timestamp,
    trade: _TradeRecord,
    symbol: str,
    skip_reason: PortfolioSkipReason,
    risk_budget: float | None = None,
) -> None:
    state.skipped_trade_ids.append(trade.trade_id)
    _append_ledger_row(
        state,
        ts=ts,
        trade_id=trade.trade_id,
        symbol=symbol,
        event="skip",
        quantity=0,
        price=None,
        risk_budget=risk_budget,
        skip_reason=skip_reason,
    )


def _close_positions_through(state: _PortfolioLedgerState, *, ts: pd.Timestamp) -> None:
    to_close = sorted(
        (position for position in state.open_positions.values() if position.exit_ts <= ts),
        key=lambda position: (position.exit_ts.value, position.entry_ts.value, position.symbol, position.trade_id),
    )
    for position in to_close:
        state.open_positions.pop(position.trade_id, None)
        state.cash += position.reserved_notional + position.pnl
        state.closed_trade_count += 1
        _append_ledger_row(
            state,
            ts=position.exit_ts,
            trade_id=position.trade_id,
            symbol=position.symbol,
            event="exit",
            quantity=position.quantity,
            price=position.exit_price,
            realized_pnl=position.pnl,
        )


def _process_portfolio_trade(
    *,
    trade: _TradeRecord,
    state: _PortfolioLedgerState,
    cfg: StrategyModelingPolicyConfig,
    max_positions: int | None,
    risk_fraction: float,
) -> None:
    precheck = _precheck_trade_for_entry(
        trade=trade,
        state=state,
        cfg=cfg,
        max_positions=max_positions,
    )
    if precheck is None:
        return
    entry_ts, symbol, skip_reason = precheck
    if skip_reason is not None:
        _append_skip_row(state, ts=entry_ts, trade=trade, symbol=symbol, skip_reason=skip_reason)
        return

    entry_price, risk_per_unit, risk_budget, quantity_by_risk, quantity_by_cash = _trade_sizing_inputs(
        trade=trade,
        state=state,
        risk_fraction=risk_fraction,
    )
    if quantity_by_risk <= 0:
        _append_skip_row(state, ts=entry_ts, trade=trade, symbol=symbol, skip_reason="risk_budget_too_small", risk_budget=risk_budget)
        return
    if quantity_by_cash <= 0:
        _append_skip_row(state, ts=entry_ts, trade=trade, symbol=symbol, skip_reason="insufficient_cash", risk_budget=risk_budget)
        return

    quantity = min(quantity_by_risk, quantity_by_cash)
    reserved_notional = float(quantity) * entry_price
    if reserved_notional > state.cash + _EPSILON:
        _append_skip_row(state, ts=entry_ts, trade=trade, symbol=symbol, skip_reason="insufficient_cash", risk_budget=risk_budget)
        return
    _open_trade_position(
        trade=trade,
        state=state,
        entry_ts=entry_ts,
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        risk_per_unit=risk_per_unit,
        risk_budget=risk_budget,
        reserved_notional=reserved_notional,
    )


def _precheck_trade_for_entry(
    *,
    trade: _TradeRecord,
    state: _PortfolioLedgerState,
    cfg: StrategyModelingPolicyConfig,
    max_positions: int | None,
) -> tuple[pd.Timestamp, str, PortfolioSkipReason | None] | None:
    entry_ts = _to_utc_timestamp(trade.entry_ts)
    if entry_ts is None:
        return None
    _close_positions_through(state, ts=entry_ts)
    symbol = _normalize_symbol(trade.symbol)
    skip_reason = _skip_reason_for_trade(trade)
    if skip_reason is not None:
        return entry_ts, symbol, skip_reason
    if cfg.one_open_per_symbol and any(position.symbol == symbol for position in state.open_positions.values()):
        return entry_ts, symbol, "one_open_per_symbol"
    if max_positions is not None and len(state.open_positions) >= max_positions:
        return entry_ts, symbol, "max_concurrent_positions"
    return entry_ts, symbol, None


def _trade_sizing_inputs(
    *,
    trade: _TradeRecord,
    state: _PortfolioLedgerState,
    risk_fraction: float,
) -> tuple[float, float, float, int, int]:
    entry_price = float(trade.entry_price)
    assert trade.initial_risk is not None
    risk_per_unit = float(trade.initial_risk)
    risk_budget = _state_current_equity(state) * risk_fraction
    quantity_by_risk = int(math.floor((risk_budget + _EPSILON) / risk_per_unit))
    quantity_by_cash = int(math.floor((state.cash + _EPSILON) / entry_price))
    return entry_price, risk_per_unit, risk_budget, quantity_by_risk, quantity_by_cash


def _open_trade_position(
    *,
    trade: _TradeRecord,
    state: _PortfolioLedgerState,
    entry_ts: pd.Timestamp,
    symbol: str,
    quantity: int,
    entry_price: float,
    risk_per_unit: float,
    risk_budget: float,
    reserved_notional: float,
) -> None:
    assert trade.exit_price is not None
    assert trade.exit_ts is not None
    exit_ts = _to_utc_timestamp(trade.exit_ts)
    assert exit_ts is not None
    exit_price = float(trade.exit_price)
    state.cash -= reserved_notional
    risk_amount = float(quantity) * risk_per_unit
    pnl = float(quantity) * _per_unit_pnl(
        direction=trade.direction,
        entry_price=entry_price,
        exit_price=exit_price,
    )
    state.open_positions[trade.trade_id] = _OpenPosition(
        trade_id=trade.trade_id,
        symbol=symbol,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        quantity=quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        reserved_notional=reserved_notional,
        pnl=pnl,
    )
    state.accepted_trade_ids.append(trade.trade_id)
    _append_ledger_row(
        state,
        ts=entry_ts,
        trade_id=trade.trade_id,
        symbol=symbol,
        event="entry",
        quantity=quantity,
        price=entry_price,
        risk_budget=risk_budget,
        risk_amount=risk_amount,
    )


def _build_portfolio_ledger_result(
    *,
    state: _PortfolioLedgerState,
    capital: float,
) -> StrategyPortfolioLedgerResult:
    return StrategyPortfolioLedgerResult(
        starting_capital=capital,
        ending_cash=state.cash,
        ending_equity=_state_current_equity(state),
        ledger=tuple(state.ledger),
        equity_curve=tuple(state.equity_curve),
        accepted_trade_ids=tuple(state.accepted_trade_ids),
        skipped_trade_ids=tuple(state.skipped_trade_ids),
    )


def select_portfolio_trade_subset(
    trades: Iterable[Mapping[str, Any] | object],
    *,
    preferred_target_label: str | None = None,
    preferred_target_r: float | None = None,
) -> StrategyPortfolioTargetSubset:
    """Select one deterministic target subset for portfolio construction."""

    parsed = _parse_trade_simulations(trades)
    all_ids = tuple(trade.trade_id for trade in parsed)
    if not parsed:
        return StrategyPortfolioTargetSubset(
            target_label=_normalize_target_label(preferred_target_label),
            target_r=_normalize_target_r(preferred_target_r),
            selection_source="empty",
            trade_ids=(),
        )

    normalized_label = _normalize_target_label(preferred_target_label)
    normalized_target_r = _normalize_target_r(preferred_target_r)
    preferred_label_match = _select_subset_by_preferred_label(parsed, normalized_label, normalized_target_r)
    if preferred_label_match is not None:
        return preferred_label_match
    preferred_target_r_match = _select_subset_by_preferred_target_r(parsed, normalized_target_r)
    if preferred_target_r_match is not None:
        return preferred_target_r_match

    inferred_label, inferred_target_r = _infer_first_target(parsed)
    if inferred_label is None and inferred_target_r is None:
        return StrategyPortfolioTargetSubset(
            target_label=normalized_label,
            target_r=normalized_target_r,
            selection_source="all_trades",
            trade_ids=all_ids,
        )

    matched = tuple(
        trade.trade_id
        for trade in parsed
        if _trade_matches_target(
            trade,
            target_label=inferred_label,
            target_r=inferred_target_r,
        )
    )
    if not matched:
        return StrategyPortfolioTargetSubset(
            target_label=normalized_label,
            target_r=normalized_target_r,
            selection_source="all_trades",
            trade_ids=all_ids,
        )
    return StrategyPortfolioTargetSubset(
        target_label=inferred_label,
        target_r=inferred_target_r,
        selection_source="inferred_first_target",
        trade_ids=matched,
    )


def _select_subset_by_preferred_label(
    parsed: list[_TradeRecord],
    normalized_label: str | None,
    normalized_target_r: float | None,
) -> StrategyPortfolioTargetSubset | None:
    if normalized_label is None:
        return None
    matched = tuple(
        trade.trade_id for trade in parsed if _trade_matches_target(trade, target_label=normalized_label, target_r=None)
    )
    if not matched:
        return None
    matched_ids = set(matched)
    inferred_target_r = normalized_target_r
    if inferred_target_r is None:
        for trade in parsed:
            if trade.trade_id not in matched_ids:
                continue
            inferred_target_r = _trade_target_r(trade)
            if inferred_target_r is not None:
                break
    return StrategyPortfolioTargetSubset(
        target_label=normalized_label,
        target_r=inferred_target_r,
        selection_source="preferred_target_label",
        trade_ids=matched,
    )


def _select_subset_by_preferred_target_r(
    parsed: list[_TradeRecord],
    normalized_target_r: float | None,
) -> StrategyPortfolioTargetSubset | None:
    if normalized_target_r is None:
        return None
    matched = tuple(
        trade.trade_id for trade in parsed if _trade_matches_target(trade, target_label=None, target_r=normalized_target_r)
    )
    if not matched:
        return None
    matched_ids = set(matched)
    inferred_label: str | None = None
    for trade in parsed:
        if trade.trade_id not in matched_ids:
            continue
        inferred_label = _trade_target_label(trade)
        if inferred_label is not None:
            break
    return StrategyPortfolioTargetSubset(
        target_label=inferred_label,
        target_r=normalized_target_r,
        selection_source="preferred_target_r",
        trade_ids=matched,
    )


def _skip_reason_for_trade(trade: _TradeRecord) -> PortfolioSkipReason | None:
    if trade.status != "closed":
        return "non_closed_trade_status"

    entry_ts = _to_utc_timestamp(trade.entry_ts)
    exit_ts = _to_utc_timestamp(trade.exit_ts)
    if entry_ts is None or exit_ts is None or exit_ts < entry_ts:
        return "invalid_trade_fill"

    if not _is_finite_positive(trade.entry_price):
        return "invalid_trade_fill"
    if not _is_finite_positive(trade.exit_price):
        return "invalid_trade_fill"
    if not _is_finite_positive(trade.initial_risk):
        return "invalid_trade_fill"

    return None


def _parse_trade_simulations(
    payloads: Iterable[Mapping[str, Any] | object],
) -> list[_TradeRecord]:
    rows: list[_TradeRecord] = []
    for payload in payloads:
        if _HAS_CONTRACT_MODELS and _ContractStrategyTradeSimulation is not None and isinstance(
            payload, _ContractStrategyTradeSimulation
        ):
            rows.append(
                _trade_record_from_mapping(
                    {
                        "trade_id": payload.trade_id,
                        "symbol": payload.symbol,
                        "direction": payload.direction,
                        "signal_confirmed_ts": payload.signal_confirmed_ts,
                        "entry_ts": payload.entry_ts,
                        "exit_ts": payload.exit_ts,
                        "status": payload.status,
                        "entry_price": payload.entry_price,
                        "target_price": payload.target_price,
                        "exit_price": payload.exit_price,
                        "initial_risk": payload.initial_risk,
                    }
                )
            )
            continue
        if _HAS_CONTRACT_MODELS and _ContractStrategyTradeSimulation is not None:
            model = _ContractStrategyTradeSimulation.model_validate(dict(payload))
            rows.append(
                _trade_record_from_mapping(
                    {
                        "trade_id": model.trade_id,
                        "symbol": model.symbol,
                        "direction": model.direction,
                        "signal_confirmed_ts": model.signal_confirmed_ts,
                        "entry_ts": model.entry_ts,
                        "exit_ts": model.exit_ts,
                        "status": model.status,
                        "entry_price": model.entry_price,
                        "target_price": model.target_price,
                        "exit_price": model.exit_price,
                        "initial_risk": model.initial_risk,
                    }
                )
            )
            continue
        rows.append(_trade_record_from_mapping(payload))
    return rows


def _trade_record_from_mapping(payload: object) -> _TradeRecord:
    row = dict(payload) if isinstance(payload, Mapping) else {}
    return _TradeRecord(
        trade_id=str(row.get("trade_id", "")).strip(),
        symbol=_normalize_symbol(row.get("symbol", "")),
        direction=str(row.get("direction", "long")).strip().lower(),
        signal_confirmed_ts=row.get("signal_confirmed_ts"),
        entry_ts=row.get("entry_ts"),
        exit_ts=row.get("exit_ts"),
        status=str(row.get("status", "")).strip().lower(),
        entry_price=row.get("entry_price"),
        target_price=row.get("target_price"),
        exit_price=row.get("exit_price"),
        initial_risk=row.get("initial_risk"),
    )


def _trade_sort_key(trade: _TradeRecord) -> tuple[int, int, str, str]:
    entry_ts = _to_utc_timestamp(trade.entry_ts)
    signal_ts = _to_utc_timestamp(trade.signal_confirmed_ts)
    return (
        -1 if entry_ts is None else int(entry_ts.value),
        -1 if signal_ts is None else int(signal_ts.value),
        _normalize_symbol(trade.symbol),
        trade.trade_id,
    )


def _to_utc_timestamp(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _is_finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return False
    return math.isfinite(number) and number > 0.0


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _normalize_target_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*[Rr]$", text)
    if match is None:
        return None
    target_r = _normalize_target_r(match.group(1))
    if target_r is None:
        return None
    tenths = round(target_r * 10.0)
    if abs(target_r - (tenths / 10.0)) <= _TARGET_R_TOLERANCE:
        return f"{tenths / 10.0:.1f}R"
    return f"{target_r:.4f}".rstrip("0").rstrip(".") + "R"


def _normalize_target_r(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _trade_target_label(trade: _TradeRecord) -> str | None:
    _, separator, suffix = str(trade.trade_id).rpartition(":")
    if not separator:
        return None
    return _normalize_target_label(suffix)


def _trade_target_r(trade: _TradeRecord) -> float | None:
    entry_price = _normalize_target_r(trade.entry_price)
    target_price = _normalize_target_r(trade.target_price)
    initial_risk = _normalize_target_r(trade.initial_risk)
    if entry_price is None or target_price is None or initial_risk is None:
        return None
    if initial_risk <= _EPSILON:
        return None

    direction = str(trade.direction).strip().lower()
    if direction == "long":
        target_r = (target_price - entry_price) / initial_risk
    elif direction == "short":
        target_r = (entry_price - target_price) / initial_risk
    else:
        return None

    if not math.isfinite(target_r) or target_r <= _EPSILON:
        return None
    return target_r


def _trade_matches_target(
    trade: _TradeRecord,
    *,
    target_label: str | None,
    target_r: float | None,
) -> bool:
    trade_label = _trade_target_label(trade)
    if target_label is not None and trade_label is not None and trade_label != target_label:
        return False

    trade_target_r = _trade_target_r(trade)
    if target_r is not None and trade_target_r is not None:
        if abs(trade_target_r - target_r) > _TARGET_R_TOLERANCE:
            return False
    elif target_r is not None and target_label is None:
        return False

    if target_label is not None and trade_label is None and target_r is None:
        return False
    return True


def _infer_first_target(trades: list[_TradeRecord]) -> tuple[str | None, float | None]:
    candidates: list[tuple[float, str | None]] = []
    for trade in trades:
        target_r = _trade_target_r(trade)
        if target_r is None:
            continue
        candidates.append((target_r, _trade_target_label(trade)))

    if candidates:
        min_target_r = min(target_r for target_r, _ in candidates)
        labels = sorted(
            {
                label
                for target_r, label in candidates
                if label is not None and abs(target_r - min_target_r) <= _TARGET_R_TOLERANCE
            }
        )
        target_label = labels[0] if labels else None
        return (target_label, min_target_r)

    labels = sorted({label for label in (_trade_target_label(trade) for trade in trades) if label is not None})
    if labels:
        inferred_label = labels[0]
        inferred_target_r = _normalize_target_r(inferred_label.rstrip("Rr"))
        return (inferred_label, inferred_target_r)
    return (None, None)


def _build_equity_point(
    *,
    ts: datetime,
    equity: float,
    cash: float,
    drawdown_pct: float,
    open_trade_count: int,
    closed_trade_count: int,
) -> Any:
    if _HAS_CONTRACT_MODELS and _ContractStrategyEquityPoint is not None:
        return _ContractStrategyEquityPoint(
            ts=ts,
            equity=equity,
            cash=cash,
            drawdown_pct=drawdown_pct,
            open_trade_count=open_trade_count,
            closed_trade_count=closed_trade_count,
        )
    return StrategyPortfolioEquityPoint(
        ts=ts,
        equity=equity,
        cash=cash,
        drawdown_pct=drawdown_pct,
        open_trade_count=open_trade_count,
        closed_trade_count=closed_trade_count,
    )


def _per_unit_pnl(*, direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "long":
        return exit_price - entry_price
    return entry_price - exit_price


__all__ = [
    "PortfolioLedgerEvent",
    "PortfolioSkipReason",
    "StrategyPortfolioLedgerResult",
    "StrategyPortfolioLedgerRow",
    "StrategyPortfolioEquityPoint",
    "StrategyPortfolioTargetSubset",
    "build_strategy_portfolio_ledger",
    "select_portfolio_trade_subset",
]
