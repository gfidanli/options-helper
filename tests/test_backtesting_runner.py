from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from options_helper.backtesting.data_source import BacktestDataSource
from options_helper.backtesting.runner import run_backtest
from options_helper.backtesting.strategy import BaselineLongCallStrategy, DayContext
from options_helper.data.candles import CandleStore
from options_helper.data.options_snapshots import OptionsSnapshotStore


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "backtest"


def test_run_backtest_single_trade_mark_slippage(tmp_path: Path) -> None:
    snapshot_store = OptionsSnapshotStore(_fixture_root())
    candle_store = CandleStore(tmp_path / "candles")
    data_source = BacktestDataSource(candle_store=candle_store, snapshot_store=snapshot_store)

    def _ctx_builder(_candles: pd.DataFrame, as_of: date) -> DayContext:
        return DayContext(as_of=as_of, spot=100.0, weekly_trend_up=True, extension_percentile=5.0)

    strategy = BaselineLongCallStrategy(
        extension_low_pct=20.0,
        take_profit_pct=10.0,
        stop_loss_pct=10.0,
        max_holding_days=1,
    )

    result = run_backtest(
        data_source,
        symbol="AAA",
        contract_symbol="AAA260220C00100000",
        strategy=strategy,
        fill_mode="mark_slippage",
        slippage_factor=0.0,
        day_context_builder=_ctx_builder,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == date(2026, 1, 2)
    assert trade.exit_date == date(2026, 1, 3)
    assert trade.entry_price == pytest.approx(1.1)
    assert trade.exit_price == pytest.approx(1.5)
    assert trade.pnl == pytest.approx(40.0)
