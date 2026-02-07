from __future__ import annotations

from datetime import datetime, timezone

import pytest

from options_helper.analysis.strategy_portfolio import build_strategy_portfolio_ledger
from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _trade(
    *,
    trade_id: str,
    symbol: str = "SPY",
    direction: str = "long",
    entry_ts: str,
    exit_ts: str,
    entry_price: float,
    exit_price: float,
    initial_risk: float,
    status: str = "closed",
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "event_id": f"evt-{trade_id}",
        "strategy": "sfp",
        "symbol": symbol,
        "direction": direction,
        "signal_ts": _ts("2026-01-01T21:00:00Z"),
        "signal_confirmed_ts": _ts("2026-01-01T21:00:00Z"),
        "entry_ts": _ts(entry_ts),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "entry_price": entry_price,
        "stop_price": entry_price - initial_risk if direction == "long" else entry_price + initial_risk,
        "target_price": entry_price + initial_risk if direction == "long" else entry_price - initial_risk,
        "exit_ts": _ts(exit_ts),
        "exit_price": exit_price,
        "status": status,
        "exit_reason": "time_stop" if status == "closed" else None,
        "reject_code": None,
        "initial_risk": initial_risk,
        "realized_r": ((exit_price - entry_price) / initial_risk)
        if direction == "long"
        else ((entry_price - exit_price) / initial_risk),
        "mae_r": 0.0,
        "mfe_r": 0.0,
        "holding_bars": 1,
        "gap_fill_applied": False,
    }


def test_portfolio_ledger_tracks_cash_and_equity_transitions() -> None:
    trade = _trade(
        trade_id="tr-1",
        entry_ts="2026-01-05T14:30:00Z",
        exit_ts="2026-01-05T15:00:00Z",
        entry_price=100.0,
        exit_price=105.0,
        initial_risk=5.0,
    )

    result = build_strategy_portfolio_ledger([trade], starting_capital=10_000.0)

    assert result.accepted_trade_ids == ("tr-1",)
    assert result.skipped_trade_ids == ()
    assert result.ending_cash == pytest.approx(10_100.0)
    assert result.ending_equity == pytest.approx(10_100.0)

    assert len(result.ledger) == 2
    entry_row, exit_row = result.ledger

    assert entry_row.event == "entry"
    assert entry_row.quantity == 20
    assert entry_row.risk_budget == pytest.approx(100.0)
    assert entry_row.risk_amount == pytest.approx(100.0)
    assert entry_row.cash_after == pytest.approx(8_000.0)
    assert entry_row.equity_after == pytest.approx(10_000.0)

    assert exit_row.event == "exit"
    assert exit_row.quantity == 20
    assert exit_row.realized_pnl == pytest.approx(100.0)
    assert exit_row.cash_after == pytest.approx(10_100.0)
    assert exit_row.equity_after == pytest.approx(10_100.0)

    assert len(result.equity_curve) == 2
    assert result.equity_curve[0].open_trade_count == 1
    assert result.equity_curve[1].open_trade_count == 0
    assert result.equity_curve[1].closed_trade_count == 1


def test_portfolio_ledger_enforces_one_open_per_symbol_and_allows_other_symbols() -> None:
    trade_spy_1 = _trade(
        trade_id="tr-spy-1",
        symbol="SPY",
        entry_ts="2026-01-05T14:30:00Z",
        exit_ts="2026-01-05T16:00:00Z",
        entry_price=10.0,
        exit_price=11.0,
        initial_risk=1.0,
    )
    trade_spy_2 = _trade(
        trade_id="tr-spy-2",
        symbol="SPY",
        entry_ts="2026-01-05T15:00:00Z",
        exit_ts="2026-01-05T16:30:00Z",
        entry_price=10.0,
        exit_price=9.0,
        initial_risk=1.0,
    )
    trade_qqq = _trade(
        trade_id="tr-qqq",
        symbol="QQQ",
        entry_ts="2026-01-05T15:00:00Z",
        exit_ts="2026-01-05T16:30:00Z",
        entry_price=10.0,
        exit_price=12.0,
        initial_risk=1.0,
    )

    default_policy_result = build_strategy_portfolio_ledger(
        [trade_spy_1, trade_spy_2, trade_qqq],
        starting_capital=20_000.0,
    )

    assert set(default_policy_result.accepted_trade_ids) == {"tr-spy-1", "tr-qqq"}
    assert set(default_policy_result.skipped_trade_ids) == {"tr-spy-2"}
    skip_rows = [row for row in default_policy_result.ledger if row.event == "skip"]
    assert len(skip_rows) == 1
    assert skip_rows[0].trade_id == "tr-spy-2"
    assert skip_rows[0].skip_reason == "one_open_per_symbol"

    allow_overlap_policy = StrategyModelingPolicyConfig(one_open_per_symbol=False)
    overlap_result = build_strategy_portfolio_ledger(
        [trade_spy_1, trade_spy_2, trade_qqq],
        starting_capital=30_000.0,
        policy=allow_overlap_policy,
    )

    assert set(overlap_result.accepted_trade_ids) == {"tr-spy-1", "tr-spy-2", "tr-qqq"}
    assert overlap_result.skipped_trade_ids == ()


def test_portfolio_ledger_skips_invalid_or_unaffordable_trades_and_keeps_invariants() -> None:
    valid = _trade(
        trade_id="tr-valid",
        symbol="SPY",
        entry_ts="2026-01-05T14:30:00Z",
        exit_ts="2026-01-05T15:00:00Z",
        entry_price=10.0,
        exit_price=11.0,
        initial_risk=1.0,
    )
    invalid_exit_before_entry = _trade(
        trade_id="tr-bad-fill",
        symbol="QQQ",
        entry_ts="2026-01-05T15:30:00Z",
        exit_ts="2026-01-05T15:00:00Z",
        entry_price=10.0,
        exit_price=11.0,
        initial_risk=1.0,
    )
    non_closed = _trade(
        trade_id="tr-non-closed",
        symbol="IWM",
        entry_ts="2026-01-05T15:30:00Z",
        exit_ts="2026-01-05T16:00:00Z",
        entry_price=10.0,
        exit_price=10.5,
        initial_risk=1.0,
        status="rejected",
    )
    risk_too_small = _trade(
        trade_id="tr-risk-small",
        symbol="DIA",
        entry_ts="2026-01-05T16:30:00Z",
        exit_ts="2026-01-05T17:00:00Z",
        entry_price=100.0,
        exit_price=101.0,
        initial_risk=10_000.0,
    )
    insufficient_cash = _trade(
        trade_id="tr-no-cash",
        symbol="XLF",
        entry_ts="2026-01-05T16:30:00Z",
        exit_ts="2026-01-05T17:00:00Z",
        entry_price=5_000.0,
        exit_price=5_100.0,
        initial_risk=1.0,
    )

    result = build_strategy_portfolio_ledger(
        [valid, invalid_exit_before_entry, non_closed, risk_too_small, insufficient_cash],
        starting_capital=1_000.0,
    )

    assert set(result.accepted_trade_ids) == {"tr-valid"}
    assert set(result.skipped_trade_ids) == {
        "tr-bad-fill",
        "tr-non-closed",
        "tr-risk-small",
        "tr-no-cash",
    }

    skip_reason_by_trade = {row.trade_id: row.skip_reason for row in result.ledger if row.event == "skip"}
    assert skip_reason_by_trade == {
        "tr-bad-fill": "invalid_trade_fill",
        "tr-non-closed": "non_closed_trade_status",
        "tr-risk-small": "risk_budget_too_small",
        "tr-no-cash": "insufficient_cash",
    }

    for row in result.ledger:
        assert row.quantity >= 0
        assert row.cash_after >= -1e-9
        if row.event in {"entry", "exit"}:
            assert row.quantity > 0
            assert row.price is not None and row.price > 0.0
        if row.event == "skip":
            assert row.quantity == 0
            assert row.skip_reason is not None

    assert all(point.equity is not None and point.equity >= 0.0 for point in result.equity_curve)
    assert all(point.drawdown_pct is not None and point.drawdown_pct <= 0.0 for point in result.equity_curve)
