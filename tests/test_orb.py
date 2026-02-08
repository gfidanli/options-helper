from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from options_helper.analysis.orb import compute_orb_signals, extract_orb_signal_candidates

_MARKET_TZ = ZoneInfo("America/New_York")


def _utc_ts(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz=_MARKET_TZ).tz_convert("UTC")


def _intraday_frame(rows: list[tuple[str, str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [_utc_ts(day, hhmm) for day, hhmm, _, _, _, _ in rows],
            "open": [open_ for _, _, open_, _, _, _ in rows],
            "high": [high for _, _, _, high, _, _ in rows],
            "low": [low for _, _, _, _, low, _ in rows],
            "close": [close for _, _, _, _, _, close in rows],
        }
    )


def test_orb_opening_range_breakout_and_anti_lookahead_timestamps() -> None:
    intraday = _intraday_frame(
        [
            ("2026-01-05", "09:25", 98.0, 101.0, 97.5, 100.0),  # premarket, ignored
            ("2026-01-05", "09:30", 100.0, 102.0, 99.0, 101.0),
            ("2026-01-05", "09:35", 101.0, 103.0, 100.0, 102.0),
            ("2026-01-05", "09:40", 102.0, 104.0, 101.0, 103.0),
            ("2026-01-05", "09:45", 103.0, 106.0, 102.0, 105.0),
            ("2026-01-05", "09:50", 106.0, 107.0, 105.0, 106.5),
            ("2026-01-05", "09:55", 106.5, 108.0, 106.0, 107.0),
        ]
    )

    signals = compute_orb_signals(intraday, range_minutes=15, cutoff_et="10:30")
    candidates = extract_orb_signal_candidates(signals)

    assert len(candidates) == 1
    event = candidates[0]
    assert event["direction"] == "long"
    assert event["opening_range_high"] == 104.0
    assert event["opening_range_low"] == 99.0
    assert event["stop_price"] == 99.0
    assert pd.Timestamp(event["signal_ts"]) == _utc_ts("2026-01-05", "09:45")
    assert pd.Timestamp(event["signal_confirmed_ts"]) == _utc_ts("2026-01-05", "09:50") - pd.Timedelta(
        microseconds=1
    )
    assert pd.Timestamp(event["entry_ts"]) == _utc_ts("2026-01-05", "09:50")
    assert pd.Timestamp(event["entry_ts"]) > pd.Timestamp(event["signal_confirmed_ts"])


def test_orb_one_event_per_session_earliest_confirmed_breakout_wins() -> None:
    intraday = _intraday_frame(
        [
            ("2026-01-06", "09:30", 100.0, 103.0, 99.0, 101.0),
            ("2026-01-06", "09:35", 101.0, 104.0, 100.0, 102.0),
            ("2026-01-06", "09:40", 102.0, 105.0, 101.0, 103.0),
            ("2026-01-06", "09:45", 103.0, 103.5, 97.0, 98.0),  # bearish breakout first
            ("2026-01-06", "09:50", 98.5, 106.5, 98.0, 106.0),  # later bullish breakout ignored
            ("2026-01-06", "09:55", 106.0, 107.0, 105.0, 106.5),
        ]
    )

    signals = compute_orb_signals(intraday, range_minutes=15, cutoff_et="10:30")
    candidates = extract_orb_signal_candidates(signals)

    assert len(candidates) == 1
    event = candidates[0]
    assert event["direction"] == "short"
    assert event["stop_price"] == 105.0
    assert pd.Timestamp(event["signal_ts"]) == _utc_ts("2026-01-06", "09:45")
    assert pd.Timestamp(event["entry_ts"]) == _utc_ts("2026-01-06", "09:50")


def test_orb_cutoff_is_applied_on_breakout_confirmation_time() -> None:
    intraday = _intraday_frame(
        [
            ("2026-01-07", "09:30", 100.0, 102.0, 99.0, 101.0),
            ("2026-01-07", "09:35", 101.0, 103.0, 100.0, 102.0),
            ("2026-01-07", "09:40", 102.0, 104.0, 101.0, 103.0),
            ("2026-01-07", "09:45", 103.0, 104.0, 102.0, 103.5),
            ("2026-01-07", "09:50", 103.5, 104.0, 103.0, 103.4),
            ("2026-01-07", "09:55", 103.4, 104.0, 103.0, 103.2),
            ("2026-01-07", "10:00", 103.2, 106.0, 103.1, 105.5),  # confirms after 10:00
            ("2026-01-07", "10:05", 105.6, 106.0, 105.0, 105.4),
        ]
    )

    signals = compute_orb_signals(intraday, range_minutes=15, cutoff_et="10:00")
    candidates = extract_orb_signal_candidates(signals)

    assert candidates == []


def test_orb_returns_no_event_when_opening_range_bars_missing() -> None:
    intraday = _intraday_frame(
        [
            ("2026-01-08", "09:45", 100.0, 103.0, 99.0, 102.0),
            ("2026-01-08", "09:50", 102.0, 104.0, 101.0, 103.0),
            ("2026-01-08", "09:55", 103.0, 106.0, 102.0, 105.0),
        ]
    )

    signals = compute_orb_signals(intraday, range_minutes=15, cutoff_et="10:30")
    candidates = extract_orb_signal_candidates(signals)

    assert candidates == []


def test_orb_returns_no_event_when_breakout_has_no_next_bar_for_entry() -> None:
    intraday = _intraday_frame(
        [
            ("2026-01-09", "09:30", 100.0, 102.0, 99.0, 101.0),
            ("2026-01-09", "09:35", 101.0, 103.0, 100.0, 102.0),
            ("2026-01-09", "09:40", 102.0, 104.0, 101.0, 103.0),
            ("2026-01-09", "09:45", 103.0, 106.0, 102.0, 105.0),  # breakout, but no next bar
        ]
    )

    signals = compute_orb_signals(intraday, range_minutes=15, cutoff_et="10:30")
    candidates = extract_orb_signal_candidates(signals)

    assert candidates == []
