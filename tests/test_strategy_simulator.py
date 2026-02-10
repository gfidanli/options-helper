from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from options_helper.analysis.strategy_simulator import (
    build_r_target_ladder,
    simulate_strategy_trade_paths,
)
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent
from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig


def _ts(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _bars_with_session(
    rows: list[tuple[str, str, float, float, float, float]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "session_date", "open", "high", "low", "close"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
    return frame


def _event(
    *,
    event_id: str,
    direction: str = "long",
    symbol: str = "SPY",
    signal_ts: str = "2026-01-02 21:00:00+00:00",
    entry_ts: str = "2026-01-05 00:00:00+00:00",
    stop_price: float | None = 99.0,
    signal_high: float | None = None,
    signal_low: float | None = None,
) -> StrategySignalEvent:
    return StrategySignalEvent(
        event_id=event_id,
        strategy="sfp",
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        signal_ts=_ts(signal_ts),
        signal_confirmed_ts=_ts(signal_ts),
        entry_ts=_ts(entry_ts),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
        signal_high=signal_high,
        signal_low=signal_low,
        stop_price=stop_price,
    )


def _single_target_ladder() -> tuple:
    return build_r_target_ladder(min_target_tenths=10, max_target_tenths=10)


def test_build_r_target_ladder_is_stable_in_integer_tenths() -> None:
    ladder = build_r_target_ladder()
    assert [target.target_tenths for target in ladder] == list(range(10, 21))
    assert [target.label for target in ladder] == [f"{tenths // 10}.{tenths % 10}R" for tenths in range(10, 21)]
    assert [target.target_r for target in ladder] == [tenths / 10.0 for tenths in range(10, 21)]


def test_simulator_long_path_uses_intraday_first_touch_chronology() -> None:
    event = _event(event_id="evt-long-chronology", stop_price=99.0)
    bars = _bars(
        [
            ("2026-01-05 14:30:00+00:00", 100.0, 100.6, 99.7, 100.4),
            ("2026-01-05 14:31:00+00:00", 100.4, 101.3, 100.2, 101.0),
            ("2026-01-05 14:32:00+00:00", 101.0, 101.1, 98.6, 98.8),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=3,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "target_hit"
    assert trade.entry_ts == datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    assert trade.exit_ts == datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)
    assert trade.holding_bars == 2
    assert trade.gap_fill_applied is False
    assert trade.realized_r == pytest.approx(1.0)


def test_simulator_supports_short_direction_paths() -> None:
    event = _event(event_id="evt-short", direction="short", stop_price=101.0)
    bars = _bars(
        [
            ("2026-01-05 14:30:00+00:00", 100.0, 100.4, 99.6, 99.8),
            ("2026-01-05 14:31:00+00:00", 99.8, 100.0, 98.9, 99.1),
            ("2026-01-05 14:32:00+00:00", 99.1, 101.3, 99.0, 101.0),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=3,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "target_hit"
    assert trade.holding_bars == 2
    assert trade.realized_r == pytest.approx(1.0)


def test_simulator_uses_conservative_stop_first_tie_break() -> None:
    event = _event(event_id="evt-stop-first", stop_price=99.0)
    bars = _bars([("2026-01-05 14:30:00+00:00", 100.0, 101.2, 98.8, 100.5)])

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=1,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "stop_hit"
    assert trade.realized_r == pytest.approx(-1.0)
    assert trade.mae_r == pytest.approx(-1.0)
    assert trade.mfe_r == pytest.approx(0.0)
    assert trade.gap_fill_applied is False


def test_simulator_applies_gap_fills_and_can_realize_less_than_negative_one_r() -> None:
    event = _event(event_id="evt-gap-stop", stop_price=99.0)
    bars = _bars(
        [
            ("2026-01-05 14:30:00+00:00", 100.0, 100.4, 99.7, 100.1),
            ("2026-01-05 14:31:00+00:00", 98.4, 98.9, 98.1, 98.7),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=3,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "stop_hit"
    assert trade.gap_fill_applied is True
    assert trade.realized_r is not None and trade.realized_r < -1.0
    assert trade.realized_r == pytest.approx(-1.6)


def test_simulator_exits_at_time_stop_when_no_stop_or_target() -> None:
    event = _event(event_id="evt-time-stop", stop_price=99.0)
    bars = _bars(
        [
            ("2026-01-05 14:30:00+00:00", 100.0, 100.5, 99.6, 100.2),
            ("2026-01-05 14:31:00+00:00", 100.2, 100.7, 99.8, 100.4),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=2,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "time_stop"
    assert trade.exit_ts == datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc)
    assert trade.exit_price == pytest.approx(100.4)
    assert trade.holding_bars == 2


def test_simulator_max_hold_timeframe_can_use_non_entry_bars() -> None:
    event = _event(event_id="evt-max-hold-10min-bars", stop_price=99.0)
    timestamps = pd.date_range("2026-01-05 14:30:00+00:00", periods=80, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.6,
            "low": 99.6,
            "close": 100.2,
        }
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=6,
        max_hold_timeframe="10Min",
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "time_stop"
    assert trade.exit_ts == datetime(2026, 1, 5, 15, 29, tzinfo=timezone.utc)
    assert trade.holding_bars == 60


def test_simulator_max_hold_timeframe_daily_uses_sessions() -> None:
    event = _event(event_id="evt-max-hold-daily-bars", stop_price=99.0)
    bars = _bars_with_session(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.6, 99.5, 100.2),
            ("2026-01-05 14:31:00+00:00", "2026-01-05", 100.2, 100.6, 99.7, 100.3),
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 100.3, 100.7, 99.8, 100.4),
            ("2026-01-06 14:31:00+00:00", "2026-01-06", 100.4, 100.8, 99.9, 100.5),
            ("2026-01-07 14:30:00+00:00", "2026-01-07", 100.5, 101.6, 100.0, 101.5),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=2,
        max_hold_timeframe="1D",
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "time_stop"
    assert trade.exit_ts == datetime(2026, 1, 6, 14, 31, tzinfo=timezone.utc)
    assert trade.holding_bars == 4


def test_simulator_rejects_invalid_signal() -> None:
    event = _event(event_id="evt-invalid", stop_price=None)
    bars = _bars([("2026-01-05 14:30:00+00:00", 100.0, 100.2, 99.8, 100.1)])

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "invalid_signal"


def test_simulator_rejects_missing_intraday_coverage() -> None:
    event = _event(event_id="evt-missing-coverage", stop_price=99.0)

    trade = simulate_strategy_trade_paths(
        [event],
        {},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "missing_intraday_coverage"


def test_simulator_rejects_missing_entry_bar_when_no_bar_after_anchor() -> None:
    event = _event(event_id="evt-missing-entry", stop_price=99.0)
    bars = _bars(
        [
            ("2026-01-02 14:30:00+00:00", 100.0, 100.2, 99.8, 100.1),
            ("2026-01-02 14:31:00+00:00", 100.1, 100.3, 99.9, 100.2),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "missing_entry_bar"


def test_simulator_rejects_non_positive_risk() -> None:
    event = _event(event_id="evt-non-positive-risk", stop_price=101.0)
    bars = _bars([("2026-01-05 14:30:00+00:00", 100.0, 100.2, 99.8, 100.1)])

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "non_positive_risk"


def test_simulator_rejects_long_when_entry_open_below_signal_low() -> None:
    event = _event(
        event_id="evt-entry-below-signal-low",
        direction="long",
        stop_price=98.0,
        signal_low=100.0,
    )
    bars = _bars([("2026-01-05 14:30:00+00:00", 99.5, 100.0, 98.9, 99.8)])

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "entry_open_outside_signal_range"


def test_simulator_rejects_short_when_entry_open_above_signal_high() -> None:
    event = _event(
        event_id="evt-entry-above-signal-high",
        direction="short",
        stop_price=103.0,
        signal_high=100.0,
    )
    bars = _bars([("2026-01-05 14:30:00+00:00", 100.5, 101.0, 99.9, 100.2)])

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "entry_open_outside_signal_range"


def test_simulator_rejects_when_future_bars_are_insufficient() -> None:
    event = _event(event_id="evt-insufficient-bars", stop_price=99.0)
    bars = _bars([("2026-01-05 14:30:00+00:00", 100.0, 100.2, 99.7, 100.0)])

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=3,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "insufficient_future_bars"


def test_simulator_rejects_when_daily_session_horizon_is_incomplete() -> None:
    event = _event(event_id="evt-incomplete-session-horizon", stop_price=99.0)
    bars = _bars_with_session(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.2, 99.7, 100.0),
            ("2026-01-05 14:31:00+00:00", "2026-01-05", 100.0, 100.3, 99.8, 100.1),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        policy=StrategyModelingPolicyConfig(max_hold_timeframe="1D"),
        max_hold_bars=2,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "rejected"
    assert trade.reject_code == "insufficient_future_bars"


def test_simulator_without_max_hold_exits_at_end_of_entry_session() -> None:
    event = _event(event_id="evt-unbounded-hold", stop_price=99.0)
    bars = _bars_with_session(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.5, 99.5, 100.2),
            ("2026-01-05 14:31:00+00:00", "2026-01-05", 100.2, 100.4, 99.7, 100.3),
            ("2026-01-05 14:32:00+00:00", "2026-01-05", 100.3, 100.6, 99.8, 100.4),
            # Next-session rows should not be included when max_hold_bars is unlimited.
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 100.4, 101.8, 100.1, 101.6),
            ("2026-01-06 14:31:00+00:00", "2026-01-06", 101.6, 102.0, 101.2, 101.8),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.exit_reason == "time_stop"
    assert trade.exit_ts == datetime(2026, 1, 5, 14, 32, tzinfo=timezone.utc)
    assert trade.holding_bars == 3


def test_simulator_uses_first_tradable_session_after_entry_anchor_gap() -> None:
    event = _event(
        event_id="evt-session-gap",
        signal_ts="2026-01-02 21:00:00+00:00",
        entry_ts="2026-01-05 00:00:00+00:00",
        stop_price=99.0,
    )
    bars = _bars(
        [
            ("2026-01-07 14:30:00+00:00", 100.9, 101.0, 100.4, 100.8),
            ("2026-01-06 14:31:00+00:00", 100.2, 101.1, 100.0, 101.0),
            ("2026-01-06 14:30:00+00:00", 100.0, 100.3, 99.8, 100.2),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=3,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.entry_ts == datetime(2026, 1, 6, 14, 30, tzinfo=timezone.utc)
    assert trade.exit_reason == "target_hit"


def test_simulator_enters_at_regular_open_and_ignores_cross_session_after_hours_rows() -> None:
    event = _event(
        event_id="evt-regular-open",
        signal_ts="2026-01-14 00:00:00+00:00",
        entry_ts="2026-01-15 00:00:00+00:00",
        stop_price=693.0,
    )
    bars = _bars_with_session(
        [
            ("2026-01-15 00:00:00+00:00", "2026-01-14", 689.55, 689.60, 689.40, 689.45),
            ("2026-01-15 09:00:00+00:00", "2026-01-15", 691.10, 691.63, 690.65, 691.63),
            ("2026-01-15 14:30:00+00:00", "2026-01-15", 694.57, 694.69, 693.97, 694.07),
            ("2026-01-15 14:31:00+00:00", "2026-01-15", 694.07, 694.20, 693.90, 694.00),
        ]
    )

    trade = simulate_strategy_trade_paths(
        [event],
        {"SPY": bars},
        max_hold_bars=2,
        target_ladder=_single_target_ladder(),
    )[0]

    assert trade.status == "closed"
    assert trade.entry_ts == datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc)
    assert trade.entry_price == pytest.approx(694.57)
