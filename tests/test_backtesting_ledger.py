from __future__ import annotations

from datetime import date

import pytest

from options_helper.backtesting.ledger import BacktestLedger


def test_ledger_open_update_close_long() -> None:
    ledger = BacktestLedger(cash=1000.0)
    position_id = "pos-1"

    ledger.open_long(
        position_id,
        symbol="AAA",
        contract_symbol="AAA260220C00100000",
        expiry=date(2026, 2, 20),
        strike=100.0,
        option_type="call",
        quantity=1,
        entry_date=date(2026, 1, 2),
        entry_price=1.0,
    )

    assert ledger.cash == 900.0
    ledger.update_mark(position_id, mark=1.5)
    ledger.update_mark(position_id, mark=0.5)

    trade = ledger.close(position_id, exit_date=date(2026, 1, 5), exit_price=1.2)
    assert trade.pnl == pytest.approx(20.0)
    assert trade.holding_days == 3
    assert trade.max_favorable == 50.0
    assert trade.max_adverse == -50.0
    assert ledger.cash == 1020.0
