from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from options_helper.analysis.strategy_modeling_contracts import serialize_strategy_signal_events
from options_helper.analysis.strategy_signals import (
    build_strategy_signal_events,
    get_strategy_signal_adapter,
    list_registered_strategy_signal_adapters,
    normalize_msb_signal_events,
)
from options_helper.schemas.strategy_modeling_contracts import STRATEGY_SIGNAL_EVENT_FIELDS, StrategySignalEvent

_MARKET_TZ = ZoneInfo("America/New_York")


def _notes_as_map(notes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for note in notes:
        if "=" not in note:
            continue
        key, value = note.split("=", 1)
        out[key] = value
    return out


def _sample_msb_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=15, freq="B")
    high = [10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0]
    low = [9.0, 10.0, 11.0, 10.0, 9.0, 11.5, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 6.5, 6.0, 5.0]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 12.3, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 6.8, 6.2, 5.5]
    open_ = [9.2, 10.2, 11.2, 10.2, 9.2, 11.8, 11.7, 10.7, 9.7, 8.7, 9.3, 10.2, 7.2, 6.4, 5.8]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _sample_ma_ohlc_long() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=7, freq="B")
    close = [10.0, 9.0, 8.0, 7.0, 9.0, 11.0, 12.0]
    open_ = [10.2, 9.2, 8.2, 7.2, 8.8, 10.8, 11.8]
    high = [10.6, 9.5, 8.5, 7.4, 9.4, 11.4, 12.4]
    low = [9.6, 8.6, 7.6, 6.6, 8.4, 10.4, 11.4]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _sample_ma_ohlc_short() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=7, freq="B")
    close = [8.0, 9.0, 10.0, 11.0, 9.0, 7.0, 6.0]
    open_ = [8.2, 9.2, 10.2, 11.2, 9.2, 7.2, 6.2]
    high = [8.6, 9.6, 10.6, 11.6, 9.6, 7.6, 6.6]
    low = [7.6, 8.6, 9.6, 10.6, 8.6, 6.6, 5.6]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _utc_ts(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz=_MARKET_TZ).tz_convert("UTC")


def _sample_orb_intraday_bars() -> pd.DataFrame:
    rows = [
        ("2026-01-05", "09:30", 100.0, 102.0, 99.0, 101.0),
        ("2026-01-05", "09:35", 101.0, 103.0, 100.0, 102.0),
        ("2026-01-05", "09:40", 102.0, 104.0, 101.0, 103.0),
        ("2026-01-05", "09:45", 103.0, 106.0, 102.0, 105.0),
        ("2026-01-05", "09:50", 106.0, 107.0, 105.0, 106.0),
    ]
    return pd.DataFrame(
        {
            "timestamp": [_utc_ts(day, hhmm) for day, hhmm, _, _, _, _ in rows],
            "open": [open_ for _, _, open_, _, _, _ in rows],
            "high": [high for _, _, _, high, _, _ in rows],
            "low": [low for _, _, _, _, low, _ in rows],
            "close": [close for _, _, _, _, _, close in rows],
        }
    )


def test_strategy_signal_registry_registers_supported_adapters() -> None:
    assert "sfp" in list_registered_strategy_signal_adapters()
    assert "msb" in list_registered_strategy_signal_adapters()
    assert "orb" in list_registered_strategy_signal_adapters()
    assert "ma_crossover" in list_registered_strategy_signal_adapters()
    assert "trend_following" in list_registered_strategy_signal_adapters()
    adapter = get_strategy_signal_adapter("sfp")
    assert callable(adapter)
    msb_adapter = get_strategy_signal_adapter("msb")
    assert callable(msb_adapter)
    orb_adapter = get_strategy_signal_adapter("orb")
    assert callable(orb_adapter)
    ma_crossover_adapter = get_strategy_signal_adapter("ma_crossover")
    assert callable(ma_crossover_adapter)
    trend_following_adapter = get_strategy_signal_adapter("trend_following")
    assert callable(trend_following_adapter)


def test_strategy_signal_registry_rejects_unknown_strategy() -> None:
    with pytest.raises(KeyError):
        get_strategy_signal_adapter("unknown-strategy")


def test_sfp_adapter_normalizes_events_with_confirmation_lag_and_next_bar_anchor() -> None:
    idx = pd.date_range("2026-01-05", periods=7, freq="B")
    ohlc = pd.DataFrame(
        {
            "Open": [1.0, 5.2, 4.4, 2.9, 4.6, 4.2, 4.0],
            "High": [1.2, 5.5, 4.0, 3.0, 6.0, 4.5, 4.3],
            "Low": [0.8, 4.2, 3.8, 2.5, 4.1, 3.8, 3.7],
            "Close": [1.0, 5.1, 3.9, 2.8, 4.7, 4.1, 4.0],
        },
        index=idx,
    )

    events = build_strategy_signal_events(
        "sfp",
        ohlc,
        symbol="spy",
        timeframe="1d",
        swing_left_bars=1,
        swing_right_bars=2,
        min_swing_distance_bars=1,
    )

    assert len(events) == 1
    event = events[0]

    assert isinstance(event, StrategySignalEvent)
    assert event.strategy == "sfp"
    assert event.symbol == "SPY"
    assert event.direction == "short"
    assert event.signal_ts == datetime(2026, 1, 9)
    assert event.signal_confirmed_ts == event.signal_ts
    assert event.entry_ts == datetime(2026, 1, 12)
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.entry_price_source == "first_tradable_bar_open_after_signal_confirmed_ts"

    notes = _notes_as_map(event.notes)
    assert notes["entry_ts_policy"] == "next_bar_open_after_signal_confirmed_ts"
    assert notes["swept_swing_timestamp"] == "2026-01-06T00:00:00"
    assert int(notes["bars_since_swing"]) >= int(notes["swing_right_bars"]) == 2

    payload = serialize_strategy_signal_events(events)[0]
    for field in ("signal_ts", "signal_confirmed_ts", "entry_ts", "entry_price_source"):
        assert field in payload


def test_sfp_adapter_skips_final_bar_signal_without_next_bar_entry_anchor() -> None:
    idx = pd.date_range("2026-01-05", periods=5, freq="B")
    ohlc = pd.DataFrame(
        {
            "Open": [1.0, 2.8, 2.1, 1.9, 2.7],
            "High": [1.1, 3.0, 2.2, 2.1, 4.0],
            "Low": [0.9, 2.4, 1.7, 1.5, 2.2],
            "Close": [1.0, 2.7, 2.0, 1.8, 2.5],
        },
        index=idx,
    )

    events = build_strategy_signal_events(
        "sfp",
        ohlc,
        symbol="SPY",
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
    )

    assert events == []


def test_msb_adapter_normalizes_events_with_confirmation_lag_and_next_bar_anchor() -> None:
    events = build_strategy_signal_events(
        "msb",
        _sample_msb_ohlc(),
        symbol="spy",
        timeframe="1d",
        swing_left_bars=1,
        swing_right_bars=2,
        min_swing_distance_bars=1,
    )

    assert len(events) >= 1
    event = events[0]

    assert isinstance(event, StrategySignalEvent)
    assert event.strategy == "msb"
    assert event.symbol == "SPY"
    assert event.direction == "long"
    assert event.signal_ts == datetime(2026, 1, 12)
    assert event.signal_confirmed_ts == event.signal_ts
    assert event.entry_ts == datetime(2026, 1, 13)
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.entry_price_source == "first_tradable_bar_open_after_signal_confirmed_ts"

    notes = _notes_as_map(event.notes)
    assert notes["entry_ts_policy"] == "next_bar_open_after_signal_confirmed_ts"
    assert notes["broken_swing_timestamp"] == "2026-01-07T00:00:00"
    assert int(notes["bars_since_swing"]) >= int(notes["swing_right_bars"]) == 2

    payload = serialize_strategy_signal_events([event])[0]
    assert set(payload) == set(STRATEGY_SIGNAL_EVENT_FIELDS)


def test_msb_adapter_payload_fields_match_sfp_contract_parity() -> None:
    sfp_idx = pd.date_range("2026-01-05", periods=7, freq="B")
    sfp_ohlc = pd.DataFrame(
        {
            "Open": [1.0, 5.2, 4.4, 2.9, 4.6, 4.2, 4.0],
            "High": [1.2, 5.5, 4.0, 3.0, 6.0, 4.5, 4.3],
            "Low": [0.8, 4.2, 3.8, 2.5, 4.1, 3.8, 3.7],
            "Close": [1.0, 5.1, 3.9, 2.8, 4.7, 4.1, 4.0],
        },
        index=sfp_idx,
    )
    sfp_events = build_strategy_signal_events(
        "sfp",
        sfp_ohlc,
        symbol="SPY",
        swing_left_bars=1,
        swing_right_bars=2,
        min_swing_distance_bars=1,
    )
    msb_events = build_strategy_signal_events(
        "msb",
        _sample_msb_ohlc(),
        symbol="SPY",
        swing_left_bars=1,
        swing_right_bars=2,
        min_swing_distance_bars=1,
    )

    assert sfp_events
    assert msb_events

    sfp_payload = serialize_strategy_signal_events([sfp_events[0]])[0]
    msb_payload = serialize_strategy_signal_events([msb_events[0]])[0]
    assert set(sfp_payload) == set(msb_payload) == set(STRATEGY_SIGNAL_EVENT_FIELDS)


def test_orb_adapter_emits_contract_fields_and_anti_lookahead_timestamps() -> None:
    daily_idx = pd.date_range("2026-01-05", periods=3, freq="B")
    daily_ohlc = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
        },
        index=daily_idx,
    )

    intraday = _sample_orb_intraday_bars()
    events = build_strategy_signal_events(
        "orb",
        daily_ohlc,
        symbol="spy",
        timeframe="1m",
        intraday_bars=intraday,
        orb_range_minutes=15,
        orb_confirmation_cutoff_et="10:30",
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, StrategySignalEvent)
    assert event.strategy == "orb"
    assert event.symbol == "SPY"
    assert event.direction == "long"
    assert pd.Timestamp(event.signal_ts) == _utc_ts("2026-01-05", "09:45")
    assert pd.Timestamp(event.entry_ts) == _utc_ts("2026-01-05", "09:50")
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.stop_price == 99.0

    notes = _notes_as_map(event.notes)
    assert notes["entry_ts_policy"] == "next_bar_open_after_signal_confirmed_ts"
    assert notes["opening_range_high"] == "104.0"
    assert notes["opening_range_low"] == "99.0"

    payload = serialize_strategy_signal_events([event])[0]
    assert set(payload) == set(STRATEGY_SIGNAL_EVENT_FIELDS)


def test_orb_adapter_accepts_intraday_bars_by_symbol_mapping() -> None:
    daily_idx = pd.date_range("2026-01-05", periods=2, freq="B")
    daily_ohlc = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
        },
        index=daily_idx,
    )
    events = build_strategy_signal_events(
        "orb",
        daily_ohlc,
        symbol="spy",
        intraday_bars_by_symbol={"SPY": _sample_orb_intraday_bars()},
    )

    assert len(events) == 1
    assert events[0].strategy == "orb"


@pytest.mark.parametrize(
    ("fast_type", "slow_type"),
    [
        ("sma", "sma"),
        ("ema", "ema"),
    ],
)
def test_ma_crossover_adapter_emits_events_with_next_bar_anchor_and_atr_stop(
    fast_type: str,
    slow_type: str,
) -> None:
    events = build_strategy_signal_events(
        "ma_crossover",
        _sample_ma_ohlc_long(),
        symbol="spy",
        timeframe="1d",
        fast_window=2,
        slow_window=3,
        fast_type=fast_type,
        slow_type=slow_type,
        atr_window=2,
        atr_stop_multiple=1.5,
    )

    assert events
    event = events[0]
    assert isinstance(event, StrategySignalEvent)
    assert event.strategy == "ma_crossover"
    assert event.symbol == "SPY"
    assert event.direction == "long"
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.stop_price is not None
    assert event.signal_close is not None
    assert float(event.stop_price) < float(event.signal_close)

    notes = _notes_as_map(event.notes)
    assert notes["entry_ts_policy"] == "next_bar_open_after_signal_confirmed_ts"
    assert notes["fast_type"] == fast_type
    assert notes["slow_type"] == slow_type

    payload = serialize_strategy_signal_events([event])[0]
    assert set(payload) == set(STRATEGY_SIGNAL_EVENT_FIELDS)


def test_ma_crossover_adapter_emits_short_events() -> None:
    events = build_strategy_signal_events(
        "ma_crossover",
        _sample_ma_ohlc_short(),
        symbol="spy",
        timeframe="1d",
        fast_window=2,
        slow_window=3,
        atr_window=2,
        atr_stop_multiple=1.5,
    )

    assert events
    event = events[0]
    assert event.strategy == "ma_crossover"
    assert event.direction == "short"
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.stop_price is not None
    assert event.signal_close is not None
    assert float(event.stop_price) > float(event.signal_close)


def test_ma_crossover_adapter_skips_final_bar_signal_without_next_bar_entry_anchor() -> None:
    idx = pd.date_range("2026-01-05", periods=6, freq="B")
    ohlc = pd.DataFrame(
        {
            "Open": [10.2, 9.2, 8.2, 7.2, 8.8, 10.8],
            "High": [10.6, 9.5, 8.5, 7.4, 9.4, 11.4],
            "Low": [9.6, 8.6, 7.6, 6.6, 8.4, 10.4],
            "Close": [10.0, 9.0, 8.0, 7.0, 9.0, 11.0],
        },
        index=idx,
    )
    events = build_strategy_signal_events(
        "ma_crossover",
        ohlc,
        symbol="SPY",
        fast_window=2,
        slow_window=3,
        atr_window=2,
        atr_stop_multiple=1.5,
    )
    assert events == []


@pytest.mark.parametrize(
    ("trend_type", "fast_type"),
    [
        ("sma", "sma"),
        ("ema", "ema"),
    ],
)
def test_trend_following_adapter_emits_events_with_next_bar_anchor_and_atr_stop(
    trend_type: str,
    fast_type: str,
) -> None:
    events = build_strategy_signal_events(
        "trend_following",
        _sample_ma_ohlc_long(),
        symbol="spy",
        timeframe="1d",
        trend_window=3,
        trend_type=trend_type,
        fast_window=2,
        fast_type=fast_type,
        slope_lookback_bars=1,
        atr_window=2,
        atr_stop_multiple=1.5,
    )

    assert events
    event = events[0]
    assert isinstance(event, StrategySignalEvent)
    assert event.strategy == "trend_following"
    assert event.symbol == "SPY"
    assert event.direction == "long"
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.stop_price is not None
    assert event.signal_close is not None
    assert float(event.stop_price) < float(event.signal_close)

    notes = _notes_as_map(event.notes)
    assert notes["entry_ts_policy"] == "next_bar_open_after_signal_confirmed_ts"
    assert notes["trend_type"] == trend_type
    assert notes["fast_type"] == fast_type

    payload = serialize_strategy_signal_events([event])[0]
    assert set(payload) == set(STRATEGY_SIGNAL_EVENT_FIELDS)


def test_trend_following_adapter_emits_short_events() -> None:
    events = build_strategy_signal_events(
        "trend_following",
        _sample_ma_ohlc_short(),
        symbol="spy",
        timeframe="1d",
        trend_window=3,
        trend_type="sma",
        fast_window=2,
        fast_type="sma",
        slope_lookback_bars=1,
        atr_window=2,
        atr_stop_multiple=1.5,
    )

    assert events
    event = events[0]
    assert event.strategy == "trend_following"
    assert event.direction == "short"
    assert event.entry_ts > event.signal_confirmed_ts
    assert event.stop_price is not None
    assert event.signal_close is not None
    assert float(event.stop_price) > float(event.signal_close)


def test_trend_following_adapter_skips_final_bar_signal_without_next_bar_entry_anchor() -> None:
    idx = pd.date_range("2026-01-05", periods=5, freq="B")
    ohlc = pd.DataFrame(
        {
            "Open": [10.2, 9.2, 8.2, 7.2, 8.8],
            "High": [10.6, 9.5, 8.5, 7.4, 9.4],
            "Low": [9.6, 8.6, 7.6, 6.6, 8.4],
            "Close": [10.0, 9.0, 8.0, 7.0, 9.0],
        },
        index=idx,
    )
    events = build_strategy_signal_events(
        "trend_following",
        ohlc,
        symbol="SPY",
        trend_window=3,
        fast_window=2,
        slope_lookback_bars=1,
        atr_window=2,
        atr_stop_multiple=1.5,
    )
    assert events == []


def test_msb_adapter_skips_final_bar_signal_without_next_bar_entry_anchor() -> None:
    idx = pd.date_range("2026-01-05", periods=5, freq="B")
    signals = pd.DataFrame(
        {
            "Open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "High": [1.1, 1.2, 1.3, 1.4, 1.5],
            "Low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "Close": [1.0, 1.1, 1.2, 1.3, 1.4],
            "bullish_msb": [False, False, False, False, True],
            "bearish_msb": [False, False, False, False, False],
            "broken_swing_high_level": [None, None, None, None, 1.2],
            "broken_swing_low_level": [None, None, None, None, None],
            "broken_swing_high_timestamp": [None, None, None, None, "2026-01-07T00:00:00"],
            "broken_swing_low_timestamp": [None, None, None, None, None],
            "bars_since_swing_high": [None, None, None, None, 2],
            "bars_since_swing_low": [None, None, None, None, None],
        },
        index=idx,
    )

    events = normalize_msb_signal_events(
        signals,
        symbol="SPY",
        swing_right_bars=1,
    )

    assert events == []
