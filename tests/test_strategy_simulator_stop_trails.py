from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from options_helper.analysis.strategy_simulator import StrategyRTarget, _EntryDecision, _prepare_intraday_frame
from options_helper.analysis.strategy_simulator_trade_paths import simulate_one_target
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent
from options_helper.schemas.strategy_modeling_policy import StopTrailRule


def _ts(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def _event(*, event_id: str = "evt-stop-trail") -> StrategySignalEvent:
    return StrategySignalEvent(
        event_id=event_id,
        strategy="sfp",
        symbol="SPY",
        direction="long",
        signal_ts=_ts("2026-01-02 21:00:00+00:00"),
        signal_confirmed_ts=_ts("2026-01-02 21:00:00+00:00"),
        entry_ts=_ts("2026-01-05 00:00:00+00:00"),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
        stop_price=99.0,
    )


def _prepared_intraday() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.9, 99.8, 100.8),
            ("2026-01-05 14:31:00+00:00", "2026-01-05", 100.8, 100.9, 100.6, 100.7),
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 101.2, 101.9, 101.1, 101.7),
            ("2026-01-06 14:31:00+00:00", "2026-01-06", 101.7, 101.8, 101.4, 101.6),
            ("2026-01-07 14:30:00+00:00", "2026-01-07", 103.5, 103.6, 102.8, 103.0),
        ],
        columns=["timestamp", "session_date", "open", "high", "low", "close"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
    return frame


def _entry_decision(*, prepared) -> _EntryDecision:
    entry_ts = pd.Timestamp(int(prepared.timestamp_ns[0]), unit="ns", tz="UTC")
    return _EntryDecision(
        reject_code=None,
        direction="long",
        stop_price=99.0,
        entry_row_index=0,
        entry_ts=entry_ts,
        entry_price=float(prepared.open_values[0]),
        initial_risk=1.0,
    )


def _target() -> StrategyRTarget:
    return StrategyRTarget(label="4.0R", target_r=4.0, target_tenths=40)


def _daily_ohlc_stage_fixture() -> pd.DataFrame:
    dates = pd.date_range("2025-11-29", "2026-01-07", freq="D")
    ramp = np.clip(np.arange(dates.size, dtype="float64") - 22.0, 0.0, None)
    close = 99.0 + (0.2 * ramp)
    return pd.DataFrame(
        {
            "date": dates.date,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
        }
    )


def test_simulator_stop_trails_apply_at_next_session_open_and_switch_stages() -> None:
    prepared = _prepare_intraday_frame(_prepared_intraday())
    daily = _daily_ohlc_stage_fixture()
    trade = simulate_one_target(
        event=_event(),
        prepared=prepared,
        entry_decision=_entry_decision(prepared=prepared),
        target=_target(),
        max_hold_bars=5,
        max_hold_timeframe=("entry", 1),
        gap_fill_policy="fill_at_open",
        stop_move_rules=(),
        stop_trail_rules=(
            StopTrailRule(start_r=0.5, ema_span=21),
            StopTrailRule(start_r=1.5, ema_span=9),
        ),
        daily_ohlc=daily,
    )

    daily_copy = daily.copy()
    daily_copy["date"] = pd.to_datetime(daily_copy["date"]).dt.date
    ema21 = daily_copy["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    ema9 = daily_copy["close"].ewm(span=9, adjust=False, min_periods=9).mean()
    expected_stage1_stop = float(ema21[daily_copy["date"] == date(2026, 1, 5)].iloc[-1])
    expected_stage2_stop = float(ema9[daily_copy["date"] == date(2026, 1, 6)].iloc[-1])

    assert [update.reason for update in trade.stop_updates] == [
        "stop_trail_tightened",
        "stop_trail_tightened",
    ]
    assert [update.stage for update in trade.stop_updates] == [
        "start_0.5R_ema21",
        "start_1.5R_ema9",
    ]
    assert [update.ts for update in trade.stop_updates] == [
        datetime(2026, 1, 6, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 7, 14, 30, tzinfo=timezone.utc),
    ]
    assert trade.stop_updates[0].stop_price == pytest.approx(expected_stage1_stop)
    assert trade.stop_updates[1].stop_price == pytest.approx(expected_stage2_stop)
    assert trade.stop_price_final == pytest.approx(expected_stage2_stop)


def test_simulator_stop_trails_stage_switches_on_exact_r_threshold_without_lookahead() -> None:
    intraday = pd.DataFrame(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.7, 99.9, 100.49),
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 100.6, 100.8, 100.3, 100.5),
            ("2026-01-07 14:30:00+00:00", "2026-01-07", 100.7, 101.8, 100.6, 101.5),
            ("2026-01-08 14:30:00+00:00", "2026-01-08", 101.6, 101.9, 101.4, 101.7),
        ],
        columns=["timestamp", "session_date", "open", "high", "low", "close"],
    )
    intraday["timestamp"] = pd.to_datetime(intraday["timestamp"], utc=True)
    intraday["session_date"] = pd.to_datetime(intraday["session_date"], errors="coerce").dt.date
    prepared = _prepare_intraday_frame(intraday)

    daily = _daily_ohlc_stage_fixture().copy()
    daily.loc[daily["date"] == date(2026, 1, 6), "close"] = 100.0
    daily.loc[daily["date"] == date(2026, 1, 7), "close"] = 102.5
    daily["high"] = daily["close"] + 0.3
    daily["low"] = daily["close"] - 0.3

    trade = simulate_one_target(
        event=_event(event_id="evt-stop-trail-threshold"),
        prepared=prepared,
        entry_decision=_entry_decision(prepared=prepared),
        target=_target(),
        max_hold_bars=4,
        max_hold_timeframe=("entry", 1),
        gap_fill_policy="fill_at_open",
        stop_move_rules=(),
        stop_trail_rules=(
            StopTrailRule(start_r=0.5, ema_span=21),
            StopTrailRule(start_r=1.5, ema_span=9),
        ),
        daily_ohlc=daily,
    )

    daily_copy = daily.copy()
    daily_copy["date"] = pd.to_datetime(daily_copy["date"]).dt.date
    ema21 = daily_copy["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    ema9 = daily_copy["close"].ewm(span=9, adjust=False, min_periods=9).mean()
    expected_stage1_prior_session = float(ema21[daily_copy["date"] == date(2026, 1, 6)].iloc[-1])
    lookahead_stage1_current_session = float(ema21[daily_copy["date"] == date(2026, 1, 7)].iloc[-1])
    expected_stage2_prior_session = float(ema9[daily_copy["date"] == date(2026, 1, 7)].iloc[-1])

    assert [update.reason for update in trade.stop_updates] == [
        "stop_trail_tightened",
        "stop_trail_tightened",
    ]
    assert [update.stage for update in trade.stop_updates] == [
        "start_0.5R_ema21",
        "start_1.5R_ema9",
    ]
    assert [update.ts for update in trade.stop_updates] == [
        datetime(2026, 1, 7, 14, 30, tzinfo=timezone.utc),
        datetime(2026, 1, 8, 14, 30, tzinfo=timezone.utc),
    ]
    assert trade.stop_updates[0].stop_price == pytest.approx(expected_stage1_prior_session)
    assert trade.stop_updates[0].stop_price != pytest.approx(lookahead_stage1_current_session)
    assert trade.stop_updates[1].stop_price == pytest.approx(expected_stage2_prior_session)


def test_simulator_stop_trails_emit_missing_indicator_trace_and_keep_stop() -> None:
    intraday = pd.DataFrame(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.6, 99.8, 100.4),
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 100.5, 100.8, 100.3, 100.6),
        ],
        columns=["timestamp", "session_date", "open", "high", "low", "close"],
    )
    intraday["timestamp"] = pd.to_datetime(intraday["timestamp"], utc=True)
    intraday["session_date"] = pd.to_datetime(intraday["session_date"], errors="coerce").dt.date
    prepared = _prepare_intraday_frame(intraday)

    daily = pd.DataFrame(
        {
            "date": pd.date_range("2025-12-27", "2026-01-05", freq="D").date,
            "high": np.linspace(100.0, 101.2, 10),
            "low": np.linspace(99.0, 100.2, 10),
            "close": np.linspace(99.5, 100.7, 10),
        }
    )

    trade = simulate_one_target(
        event=_event(event_id="evt-missing-indicator"),
        prepared=prepared,
        entry_decision=_entry_decision(prepared=prepared),
        target=_target(),
        max_hold_bars=2,
        max_hold_timeframe=("entry", 1),
        gap_fill_policy="fill_at_open",
        stop_move_rules=(),
        stop_trail_rules=(StopTrailRule(start_r=0.2, ema_span=9, buffer_atr_multiple=0.5),),
        daily_ohlc=daily,
    )

    assert len(trade.stop_updates) == 1
    assert trade.stop_updates[0].reason == "stop_trail_missing_prior_session_indicator"
    assert trade.stop_updates[0].stage == "start_0.2R_ema9"
    assert trade.stop_updates[0].ts == datetime(2026, 1, 6, 14, 30, tzinfo=timezone.utc)
    assert trade.stop_updates[0].stop_price == pytest.approx(99.0)
    assert trade.stop_price == pytest.approx(99.0)
    assert trade.stop_price_final == pytest.approx(99.0)


def test_simulator_stop_trails_emit_missing_indicator_trace_for_insufficient_ema() -> None:
    intraday = pd.DataFrame(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.6, 99.8, 100.4),
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 100.5, 100.8, 100.3, 100.6),
        ],
        columns=["timestamp", "session_date", "open", "high", "low", "close"],
    )
    intraday["timestamp"] = pd.to_datetime(intraday["timestamp"], utc=True)
    intraday["session_date"] = pd.to_datetime(intraday["session_date"], errors="coerce").dt.date
    prepared = _prepare_intraday_frame(intraday)

    daily = pd.DataFrame(
        {
            "date": pd.date_range("2025-12-27", "2026-01-05", freq="D").date,
            "high": np.linspace(100.0, 101.2, 10),
            "low": np.linspace(99.0, 100.2, 10),
            "close": np.linspace(99.5, 100.7, 10),
        }
    )

    trade = simulate_one_target(
        event=_event(event_id="evt-missing-ema"),
        prepared=prepared,
        entry_decision=_entry_decision(prepared=prepared),
        target=_target(),
        max_hold_bars=2,
        max_hold_timeframe=("entry", 1),
        gap_fill_policy="fill_at_open",
        stop_move_rules=(),
        stop_trail_rules=(StopTrailRule(start_r=0.2, ema_span=21),),
        daily_ohlc=daily,
    )

    assert len(trade.stop_updates) == 1
    assert trade.stop_updates[0].reason == "stop_trail_missing_prior_session_indicator"
    assert trade.stop_updates[0].stage == "start_0.2R_ema21"
    assert trade.stop_updates[0].ts == datetime(2026, 1, 6, 14, 30, tzinfo=timezone.utc)
    assert trade.stop_updates[0].stop_price == pytest.approx(99.0)
    assert trade.stop_price == pytest.approx(99.0)
    assert trade.stop_price_final == pytest.approx(99.0)
