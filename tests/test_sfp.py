from __future__ import annotations

import pandas as pd

from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events


def _sample_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=15, freq="B")
    high = [10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0]
    low = [9.0, 10.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 8.0, 7.0, 8.0, 9.0, 6.8, 8.0, 9.0]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 11.8, 11.5, 10.5, 8.5, 8.2, 8.5, 9.5, 7.2, 8.8, 9.2]
    open_ = [9.2, 10.2, 11.2, 10.2, 9.2, 12.2, 11.7, 10.7, 8.8, 8.6, 8.7, 9.1, 7.9, 8.6, 9.0]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_sfp_flags_bearish_and_bullish_patterns() -> None:
    df = _sample_ohlc()
    signals = compute_sfp_signals(df, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=1)

    # Bearish SFP: wick above prior swing high (12.0) then close back below.
    assert bool(signals.iloc[5]["bearish_sfp"]) is True
    assert signals.iloc[5]["swept_swing_high_level"] == 12.0
    assert signals.iloc[5]["swept_swing_high_timestamp"] == df.index[2].isoformat()
    assert int(signals.iloc[5]["bars_since_swing_high"]) == 3

    # Bullish SFP: wick below prior swing low (7.0) then close back above.
    assert bool(signals.iloc[12]["bullish_sfp"]) is True
    assert signals.iloc[12]["swept_swing_low_level"] == 7.0
    assert signals.iloc[12]["swept_swing_low_timestamp"] == df.index[9].isoformat()
    assert int(signals.iloc[12]["bars_since_swing_low"]) == 3


def test_sfp_min_swing_distance_filters_too_recent_sweeps() -> None:
    df = _sample_ohlc()
    signals = compute_sfp_signals(df, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=4)

    # Both synthetic events are 3 bars away from their swept swing points.
    assert bool(signals.iloc[5]["bearish_sfp"]) is False
    assert bool(signals.iloc[12]["bullish_sfp"]) is False


def test_sfp_extract_events_returns_directional_rows() -> None:
    df = _sample_ohlc()
    signals = compute_sfp_signals(df, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=1)
    events = extract_sfp_events(signals)

    directions = {ev.direction for ev in events}
    assert "bearish" in directions
    assert "bullish" in directions


def test_sfp_supports_timeframe_resampling() -> None:
    df = _sample_ohlc()
    signals = compute_sfp_signals(
        df,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
        timeframe="W-FRI",
    )

    assert not signals.empty
    assert len(signals) < len(df)


def test_sfp_weekly_resample_uses_monday_labels() -> None:
    df = _sample_ohlc()
    signals = compute_sfp_signals(df, timeframe="W-FRI")

    assert not signals.empty
    assert isinstance(signals.index, pd.DatetimeIndex)
    assert all(ts.weekday() == 0 for ts in signals.index)
