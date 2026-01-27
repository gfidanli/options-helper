from __future__ import annotations

import pytest

from options_helper.analysis.performance import compute_daily_performance_quote


def test_daily_quote_derived_percent_change() -> None:
    q = compute_daily_performance_quote(last_price=0.92, change=0.31, percent_change_raw=None, contracts=1)
    assert q.daily_pnl == pytest.approx(31.0)
    assert q.prev_close_price == pytest.approx(0.61)
    assert q.percent_change == pytest.approx(50.81967, abs=1e-4)


def test_daily_quote_pass_through_percent_change() -> None:
    q = compute_daily_performance_quote(last_price=None, change=None, percent_change_raw=5.4, contracts=2)
    assert q.daily_pnl is None
    assert q.prev_close_price is None
    assert q.percent_change == 5.4

