from __future__ import annotations

import pandas as pd

from options_helper.analysis.msb import compute_msb_signals, extract_msb_events


def _sample_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=15, freq="B")
    high = [10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0]
    low = [9.0, 10.0, 11.0, 10.0, 9.0, 11.5, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 6.5, 6.0, 5.0]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 12.3, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 6.8, 6.2, 5.5]
    open_ = [9.2, 10.2, 11.2, 10.2, 9.2, 11.8, 11.7, 10.7, 9.7, 8.7, 9.3, 10.2, 7.2, 6.4, 5.8]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_msb_flags_bullish_and_bearish_breaks() -> None:
    df = _sample_ohlc()
    signals = compute_msb_signals(df, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=1)

    # Bullish MSB: close above prior swing high (12.0).
    assert bool(signals.iloc[5]["bullish_msb"]) is True
    assert signals.iloc[5]["broken_swing_high_level"] == 12.0
    assert signals.iloc[5]["broken_swing_high_timestamp"] == df.index[2].isoformat()
    assert int(signals.iloc[5]["bars_since_swing_high"]) == 3

    # Bearish MSB: close below prior swing low (8.0).
    assert bool(signals.iloc[12]["bearish_msb"]) is True
    assert signals.iloc[12]["broken_swing_low_level"] == 8.0
    assert signals.iloc[12]["broken_swing_low_timestamp"] == df.index[9].isoformat()
    assert int(signals.iloc[12]["bars_since_swing_low"]) == 3


def test_msb_min_swing_distance_filters_too_recent_breaks() -> None:
    df = _sample_ohlc()
    signals = compute_msb_signals(df, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=4)

    assert bool(signals.iloc[5]["bullish_msb"]) is False
    assert bool(signals.iloc[12]["bearish_msb"]) is False


def test_msb_extract_events_returns_directional_rows() -> None:
    df = _sample_ohlc()
    signals = compute_msb_signals(df, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=1)
    events = extract_msb_events(signals)

    directions = {ev.direction for ev in events}
    assert "bullish" in directions
    assert "bearish" in directions


def test_msb_supports_timeframe_resampling() -> None:
    df = _sample_ohlc()
    signals = compute_msb_signals(
        df,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
        timeframe="W-FRI",
    )

    assert not signals.empty
    assert len(signals) < len(df)


def test_msb_weekly_resample_uses_monday_labels() -> None:
    df = _sample_ohlc()
    signals = compute_msb_signals(df, timeframe="W-FRI")

    assert not signals.empty
    assert isinstance(signals.index, pd.DatetimeIndex)
    assert all(ts.weekday() == 0 for ts in signals.index)


def test_msb_uses_confirmed_swings_only_when_right_bars_gt_one() -> None:
    idx = pd.date_range("2026-01-05", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.5, 2.0, 1.2, 1.5, 1.1],
            "High": [1.0, 3.0, 2.0, 1.0, 2.0, 1.0],
            "Low": [0.8, 1.8, 1.5, 0.7, 1.2, 0.9],
            "Close": [0.9, 2.4, 1.9, 0.9, 1.4, 1.0],
        },
        index=idx,
    )

    signals = compute_msb_signals(df, swing_left_bars=1, swing_right_bars=2, min_swing_distance_bars=1)

    # Swing high at index 1 is confirmed only after index 3.
    assert pd.isna(signals.iloc[2]["last_swing_high_level"])
    assert signals.iloc[3]["last_swing_high_level"] == 3.0
