from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from options_helper.analysis.strategy_modeling_contracts import serialize_strategy_signal_events
from options_helper.analysis.strategy_signals import (
    build_strategy_signal_events,
    get_strategy_signal_adapter,
    list_registered_strategy_signal_adapters,
)
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent


def _notes_as_map(notes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for note in notes:
        if "=" not in note:
            continue
        key, value = note.split("=", 1)
        out[key] = value
    return out


def test_strategy_signal_registry_registers_sfp_adapter() -> None:
    assert "sfp" in list_registered_strategy_signal_adapters()
    adapter = get_strategy_signal_adapter("sfp")
    assert callable(adapter)


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
