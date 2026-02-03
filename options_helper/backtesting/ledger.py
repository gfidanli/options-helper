from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class TradeLogRow:
    symbol: str
    contract_symbol: str | None
    expiry: date
    strike: float
    option_type: str
    quantity: int
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    holding_days: int
    pnl: float
    pnl_pct: float | None
    max_favorable: float | None
    max_adverse: float | None


@dataclass
class PositionState:
    symbol: str
    contract_symbol: str | None
    expiry: date
    strike: float
    option_type: str
    quantity: int
    entry_date: date
    entry_price: float
    max_favorable: float | None = None
    max_adverse: float | None = None

    def update_mark(self, mark: float) -> None:
        pnl = (mark - self.entry_price) * self.quantity * CONTRACT_MULTIPLIER
        if self.max_favorable is None or pnl > self.max_favorable:
            self.max_favorable = pnl
        if self.max_adverse is None or pnl < self.max_adverse:
            self.max_adverse = pnl


@dataclass
class BacktestLedger:
    cash: float
    positions: dict[str, PositionState] = field(default_factory=dict)
    trades: list[TradeLogRow] = field(default_factory=list)

    def open_long(
        self,
        position_id: str,
        *,
        symbol: str,
        contract_symbol: str | None,
        expiry: date,
        strike: float,
        option_type: str,
        quantity: int,
        entry_date: date,
        entry_price: float,
    ) -> None:
        if position_id in self.positions:
            raise ValueError(f"position already open: {position_id}")
        if entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        cost = entry_price * quantity * CONTRACT_MULTIPLIER
        self.cash -= cost
        self.positions[position_id] = PositionState(
            symbol=symbol,
            contract_symbol=contract_symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            quantity=quantity,
            entry_date=entry_date,
            entry_price=entry_price,
        )

    def update_mark(self, position_id: str, *, mark: float | None) -> None:
        if mark is None:
            return
        if position_id not in self.positions:
            raise KeyError(f"position not open: {position_id}")
        self.positions[position_id].update_mark(mark)

    def close(
        self,
        position_id: str,
        *,
        exit_date: date,
        exit_price: float,
    ) -> TradeLogRow:
        if exit_price <= 0:
            raise ValueError("exit_price must be > 0")
        if position_id not in self.positions:
            raise KeyError(f"position not open: {position_id}")

        pos = self.positions.pop(position_id)
        proceeds = exit_price * pos.quantity * CONTRACT_MULTIPLIER
        self.cash += proceeds

        holding_days = (exit_date - pos.entry_date).days
        pnl = (exit_price - pos.entry_price) * pos.quantity * CONTRACT_MULTIPLIER
        pnl_pct = None
        if pos.entry_price > 0:
            pnl_pct = (exit_price / pos.entry_price) - 1.0

        trade = TradeLogRow(
            symbol=pos.symbol,
            contract_symbol=pos.contract_symbol,
            expiry=pos.expiry,
            strike=pos.strike,
            option_type=pos.option_type,
            quantity=pos.quantity,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            holding_days=holding_days,
            pnl=pnl,
            pnl_pct=pnl_pct,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
        )
        self.trades.append(trade)
        return trade

