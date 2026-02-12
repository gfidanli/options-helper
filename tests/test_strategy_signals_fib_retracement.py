from __future__ import annotations

import pandas as pd
import pytest

from options_helper.analysis.strategy_signals import build_strategy_signal_events
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent


def _fib_bullish_setup_ohlc(*, final_touch_only: bool) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=8, freq="B")
    lows = [9.0, 8.0, 9.0, 9.4, 11.0, 10.1, 10.2, 10.1]
    highs = [10.0, 11.0, 12.0, 11.5, 14.0, 13.0, 10.4, 10.8]
    closes = [9.5, 10.5, 11.5, 12.5, 13.2, 10.9, 10.3, 10.2]

    if final_touch_only:
        highs[6] = 10.5
        lows[6] = 10.35
        closes[6] = 10.4

    return pd.DataFrame(
        {
            "Open": [9.5, 9.8, 10.8, 11.4, 12.8, 12.9, 10.8, 10.5],
            "High": highs,
            "Low": lows,
            "Close": closes,
        },
        index=idx,
    )


def test_fib_retracement_adapter_normalizes_direction_stop_and_entry_anchor() -> None:
    ohlc = _fib_bullish_setup_ohlc(final_touch_only=False)
    events = build_strategy_signal_events(
        "fib_retracement",
        ohlc,
        symbol="spy",
        timeframe="1d",
        fib_retracement_pct=61.8,
    )

    assert len(events) == 1
    event = events[0]

    assert isinstance(event, StrategySignalEvent)
    assert event.strategy == "fib_retracement"
    assert event.direction == "long"
    assert event.stop_price == pytest.approx(8.0, abs=1e-9)
    assert event.signal_ts == ohlc.index[6].to_pydatetime()
    assert event.entry_ts == ohlc.index[7].to_pydatetime()
    assert event.entry_ts > event.signal_confirmed_ts


def test_fib_retracement_adapter_skips_final_touch_without_i_plus_one_bar() -> None:
    ohlc = _fib_bullish_setup_ohlc(final_touch_only=True)
    events = build_strategy_signal_events(
        "fib_retracement",
        ohlc,
        symbol="SPY",
        timeframe="1d",
        fib_retracement_pct=61.8,
    )

    assert events == []
