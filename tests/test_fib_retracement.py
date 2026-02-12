from __future__ import annotations

import pandas as pd
import pytest

from options_helper.analysis.fib_retracement import compute_fib_retracement_signals


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


def test_compute_fib_retracement_signals_bullish_flow_and_confirmation_lag() -> None:
    ohlc = _fib_bullish_setup_ohlc(final_touch_only=False)
    signals = compute_fib_retracement_signals(ohlc, fib_retracement_pct=61.8)

    expected_entry = 14.0 - ((14.0 - 8.0) * 0.618)
    pre_lag_touch_idx = ohlc.index[5]
    signal_idx = ohlc.index[6]

    assert bool(signals.loc[ohlc.index[3], "bullish_msb"]) is True
    assert bool(signals.loc[ohlc.index[4], "swing_high"]) is True
    assert ohlc.loc[pre_lag_touch_idx, "Low"] <= expected_entry <= ohlc.loc[pre_lag_touch_idx, "High"]
    assert bool(signals.loc[pre_lag_touch_idx, "fib_retracement_long"]) is False

    assert int(signals["fib_retracement_long"].sum()) == 1
    assert int(signals["fib_retracement_short"].sum()) == 0
    assert bool(signals.loc[signal_idx, "fib_retracement_long"]) is True
    assert float(signals.loc[signal_idx, "fib_entry_level"]) == pytest.approx(expected_entry, abs=1e-9)
    assert float(signals.loc[signal_idx, "fib_range_low_level"]) == pytest.approx(8.0, abs=1e-9)
    assert float(signals.loc[signal_idx, "fib_range_high_level"]) == pytest.approx(14.0, abs=1e-9)
    assert signals.loc[signal_idx, "fib_msb_timestamp"] == ohlc.index[3].isoformat()
    assert signals.loc[signal_idx, "fib_range_high_timestamp"] == ohlc.index[4].isoformat()
    assert signals.loc[signal_idx, "fib_range_low_timestamp"] == ohlc.index[1].isoformat()


def test_compute_fib_retracement_signals_skips_final_bar_touch_without_next_entry_bar() -> None:
    ohlc = _fib_bullish_setup_ohlc(final_touch_only=True)
    signals = compute_fib_retracement_signals(ohlc, fib_retracement_pct=61.8)

    expected_entry = 14.0 - ((14.0 - 8.0) * 0.618)
    final_idx = ohlc.index[-1]

    assert ohlc.loc[final_idx, "Low"] <= expected_entry <= ohlc.loc[final_idx, "High"]
    assert int(signals["fib_retracement_long"].sum()) == 0
    assert int(signals["fib_retracement_short"].sum()) == 0
