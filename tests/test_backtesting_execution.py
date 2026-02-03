from __future__ import annotations

import pytest

from options_helper.backtesting.execution import fill_price


def test_fill_price_worst_case_bid_ask() -> None:
    buy = fill_price(side="buy", fill_mode="worst_case", bid=1.0, ask=1.2)
    sell = fill_price(side="sell", fill_mode="worst_case", bid=1.0, ask=1.2)

    assert buy.price == pytest.approx(1.2)
    assert buy.used == "ask"
    assert sell.price == pytest.approx(1.0)
    assert sell.used == "bid"


def test_fill_price_worst_case_missing_bid_ask() -> None:
    res = fill_price(side="buy", fill_mode="worst_case", bid=None, ask=None, mark=1.1)
    assert res.price is None
    assert res.reason == "missing_bid_ask"


def test_fill_price_worst_case_fallback_mark() -> None:
    res = fill_price(
        side="buy",
        fill_mode="worst_case",
        bid=None,
        ask=None,
        mark=1.1,
        allow_worst_case_mark_fallback=True,
    )
    assert res.price == pytest.approx(1.1)
    assert res.reason == "fallback_mark"


def test_fill_price_mark_slippage_with_spread_pct() -> None:
    buy = fill_price(
        side="buy",
        fill_mode="mark_slippage",
        mark=1.0,
        spread_pct=0.2,
        slippage_factor=0.5,
    )
    sell = fill_price(
        side="sell",
        fill_mode="mark_slippage",
        mark=1.0,
        spread_pct=0.2,
        slippage_factor=0.5,
    )

    assert buy.price == pytest.approx(1.1)
    assert sell.price == pytest.approx(0.9)


def test_fill_price_mark_slippage_computes_spread_pct() -> None:
    res = fill_price(
        side="buy",
        fill_mode="mark_slippage",
        bid=1.0,
        ask=1.2,
        mark=1.1,
        slippage_factor=1.0,
    )
    # spread_pct = (1.2 - 1.0) / 1.1 = 0.181818...
    assert res.price == pytest.approx(1.1 * (1.0 + (0.2 / 1.1)))


def test_fill_price_mark_slippage_uses_mid_when_mark_missing() -> None:
    res = fill_price(
        side="buy",
        fill_mode="mark_slippage",
        bid=1.0,
        ask=1.2,
        mark=None,
        slippage_factor=0.0,
    )
    assert res.price == pytest.approx(1.1)


def test_fill_price_mark_slippage_missing_mark() -> None:
    res = fill_price(side="buy", fill_mode="mark_slippage", bid=None, ask=None, mark=None)
    assert res.price is None
    assert res.reason == "missing_mark"

