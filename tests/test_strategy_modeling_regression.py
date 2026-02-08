from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from typer.testing import CliRunner

from options_helper.analysis.msb import compute_msb_signals, extract_msb_events
from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events
from options_helper.analysis.strategy_features import StrategyFeatureConfig
from options_helper.analysis.strategy_modeling_filters import FILTER_REJECT_REASONS, apply_entry_filters
from options_helper.analysis.strategy_portfolio import build_strategy_portfolio_ledger
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.zero_dte_dataset import ZeroDTEIntradayDatasetLoader
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent
from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig
from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig
from options_helper.cli import app


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _bars_frame(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [500.0 + idx for idx in range(len(timestamps))],
            "high": [500.1 + idx for idx in range(len(timestamps))],
            "low": [499.9 + idx for idx in range(len(timestamps))],
            "close": [500.05 + idx for idx in range(len(timestamps))],
            "volume": [1000 + idx for idx in range(len(timestamps))],
        }
    )


def _signal_fixture_with_confirmation_lag() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=6, freq="B")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.5, 2.0, 1.2, 1.5, 1.1],
            "High": [1.0, 3.0, 2.0, 1.0, 2.0, 1.0],
            "Low": [0.8, 1.8, 1.5, 0.7, 1.2, 0.9],
            "Close": [0.9, 2.4, 1.9, 0.9, 1.4, 1.0],
        },
        index=idx,
    )


def _sfp_scan_fixture() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=15, freq="B")
    high = [10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0]
    low = [9.0, 10.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 8.0, 7.0, 8.0, 9.0, 6.8, 8.0, 9.0]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 11.8, 11.5, 10.5, 8.5, 8.2, 8.5, 9.5, 7.2, 8.8, 9.2]
    open_ = [9.2, 10.2, 11.2, 10.2, 9.2, 12.2, 11.7, 10.7, 8.8, 8.6, 8.7, 9.1, 7.9, 8.6, 9.0]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _msb_scan_fixture() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=15, freq="B")
    high = [10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0]
    low = [9.0, 10.0, 11.0, 10.0, 9.0, 11.5, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 6.5, 6.0, 5.0]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 12.3, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 6.8, 6.2, 5.5]
    open_ = [9.2, 10.2, 11.2, 10.2, 9.2, 11.8, 11.7, 10.7, 9.7, 8.7, 9.3, 10.2, 7.2, 6.4, 5.8]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _trade(
    *,
    trade_id: str,
    symbol: str = "SPY",
    direction: str = "long",
    entry_ts: str,
    exit_ts: str,
    entry_price: float,
    exit_price: float,
    initial_risk: float,
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "event_id": f"evt-{trade_id}",
        "strategy": "sfp",
        "symbol": symbol,
        "direction": direction,
        "signal_ts": _ts("2026-01-05T14:00:00Z"),
        "signal_confirmed_ts": _ts("2026-01-05T14:00:00Z"),
        "entry_ts": _ts(entry_ts),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "entry_price": entry_price,
        "stop_price": entry_price - initial_risk if direction == "long" else entry_price + initial_risk,
        "target_price": entry_price + initial_risk if direction == "long" else entry_price - initial_risk,
        "exit_ts": _ts(exit_ts),
        "exit_price": exit_price,
        "status": "closed",
        "exit_reason": "time_stop",
        "reject_code": None,
        "initial_risk": initial_risk,
        "realized_r": ((exit_price - entry_price) / initial_risk)
        if direction == "long"
        else ((entry_price - exit_price) / initial_risk),
        "mae_r": 0.0,
        "mfe_r": 0.0,
        "holding_bars": 1,
        "gap_fill_applied": False,
    }


_MARKET_TZ = ZoneInfo("America/New_York")


def _market_utc(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz=_MARKET_TZ).tz_convert("UTC")


def _strategy_event(
    *,
    event_id: str,
    direction: str = "long",
    strategy: str = "sfp",
    symbol: str = "SPY",
    signal_day: str = "2026-01-05",
    entry_day: str = "2026-01-06",
    signal_close: float = 100.0,
    stop_price: float = 99.0,
) -> StrategySignalEvent:
    signal_ts = pd.Timestamp(f"{signal_day}T00:00:00Z")
    entry_ts = pd.Timestamp(f"{entry_day}T00:00:00Z")
    return StrategySignalEvent(
        event_id=event_id,
        strategy=strategy,  # type: ignore[arg-type]
        symbol=symbol,
        timeframe="1d",
        direction=direction,  # type: ignore[arg-type]
        signal_ts=signal_ts.to_pydatetime(),
        signal_confirmed_ts=signal_ts.to_pydatetime(),
        entry_ts=entry_ts.to_pydatetime(),
        entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
        signal_open=signal_close - 0.5,
        signal_high=signal_close + 0.5,
        signal_low=signal_close - 1.0,
        signal_close=signal_close,
        stop_price=stop_price,
        notes=[],
    )


def _daily_features_frame(
    rows: list[tuple[str, float, float, float, float, str]],
) -> pd.DataFrame:
    index = [pd.Timestamp(f"{day}T00:00:00Z") for day, *_ in rows]
    return pd.DataFrame(
        {
            "rsi": [rsi for _, rsi, _, _, _, _ in rows],
            "atr": [atr for _, _, atr, _, _, _ in rows],
            "ema9": [ema9 for _, _, _, ema9, _, _ in rows],
            "ema9_slope": [ema9_slope for _, _, _, _, ema9_slope, _ in rows],
            "volatility_regime": [regime for _, _, _, _, _, regime in rows],
        },
        index=index,
    )


def _daily_ohlc_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    index = [pd.Timestamp(f"{day}T00:00:00Z") for day, _ in rows]
    close = [value for _, value in rows]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
        },
        index=index,
    )


def _intraday_frame(day: str, rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [_market_utc(day, hhmm) for hhmm, *_ in rows],
            "open": [open_ for _, open_, _, _, _ in rows],
            "high": [high for _, _, high, _, _ in rows],
            "low": [low for _, _, _, low, _ in rows],
            "close": [close for _, _, _, _, close in rows],
        }
    )


def _assert_single_reject(summary: dict[str, object], reason: str) -> None:
    assert summary["base_event_count"] == 1
    assert summary["kept_event_count"] == 0
    assert summary["rejected_event_count"] == 1
    reject_counts = summary["reject_counts"]
    assert isinstance(reject_counts, dict)
    assert reject_counts[reason] == 1
    for key in FILTER_REJECT_REASONS:
        if key != reason:
            assert reject_counts[key] == 0


@pytest.mark.parametrize(
    ("signals_fn", "kwargs"),
    (
        (compute_sfp_signals, {"swing_left_bars": 1, "swing_right_bars": 2, "min_swing_distance_bars": 1}),
        (compute_msb_signals, {"swing_left_bars": 1, "swing_right_bars": 2, "min_swing_distance_bars": 1}),
    ),
)
def test_strategy_modeling_confirmation_lag_blocks_preconfirmation_swing_usage(signals_fn, kwargs) -> None:  # type: ignore[no-untyped-def]
    frame = _signal_fixture_with_confirmation_lag()
    signals = signals_fn(frame, **kwargs)

    # Swing high at index 1 can only be consumed after two right bars confirm it.
    assert pd.isna(signals.iloc[2]["last_swing_high_level"])
    assert signals.iloc[3]["last_swing_high_level"] == 3.0


@pytest.mark.parametrize(
    ("command_name", "frame_builder", "signals_fn", "events_fn"),
    (
        ("sfp-scan", _sfp_scan_fixture, compute_sfp_signals, extract_sfp_events),
        ("msb-scan", _msb_scan_fixture, compute_msb_signals, extract_msb_events),
    ),
)
def test_strategy_modeling_cli_parity_preserves_next_bar_open_anchor(  # type: ignore[no-untyped-def]
    tmp_path: Path,
    command_name: str,
    frame_builder,
    signals_fn,
    events_fn,
) -> None:
    frame = frame_builder()
    ohlc_path = tmp_path / "ohlc.csv"
    out_dir = tmp_path / "reports"
    frame.to_csv(ohlc_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            command_name,
            "--ohlc-path",
            str(ohlc_path),
            "--out",
            str(out_dir),
            "--swing-left-bars",
            "1",
            "--swing-right-bars",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    asof = frame.index.max().date().isoformat()
    payload_path = out_dir / "UNKNOWN" / f"{asof}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    signals = signals_fn(frame, swing_left_bars=1, swing_right_bars=1, min_swing_distance_bars=1)
    direct_events = events_fn(signals)
    assert payload["events"]
    assert len(payload["events"]) == len(direct_events)
    assert [event["timestamp"] for event in payload["events"]] == [
        pd.Timestamp(event.timestamp).date().isoformat() for event in direct_events
    ]

    index_positions = {ts: pos for pos, ts in enumerate(signals.index)}
    open_series = signals["Open"].astype(float)
    close_series = signals["Close"].astype(float)

    for event in payload["events"]:
        event_ts = pd.Timestamp(event["timestamp"])
        event_pos = index_positions[event_ts]
        entry_pos = event_pos + 1

        assert entry_pos < len(signals)
        expected_entry_ts = signals.index[entry_pos]
        expected_entry_price = float(open_series.iloc[entry_pos])
        expected_next_close = float(close_series.iloc[entry_pos])
        wrong_same_bar_close = float(close_series.iloc[event_pos])

        expected_1d = round((expected_next_close / expected_entry_price - 1.0) * 100.0, 2)
        wrong_lookahead = round((expected_next_close / wrong_same_bar_close - 1.0) * 100.0, 2)

        assert event["entry_anchor_timestamp"] == expected_entry_ts.date().isoformat()
        assert event["entry_anchor_price"] == round(expected_entry_price, 2)
        assert event["forward_returns_pct"]["1d"] == expected_1d
        assert event["forward_returns_pct"]["1d"] != wrong_lookahead


def test_strategy_modeling_intraday_required_gate_blocks_when_bars_missing(tmp_path: Path) -> None:
    policy = StrategyModelingPolicyConfig()
    assert policy.require_intraday_bars is True

    store = IntradayStore(tmp_path / "intraday")
    loader = ZeroDTEIntradayDatasetLoader(intraday_store=store)
    target_day = date(2026, 2, 3)

    missing = loader.load_day(target_day, decision_times=["09:30"])
    assert list(missing.state_rows["status"].astype(str)) == ["no_underlying_data"]

    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "SPY",
        target_day,
        _bars_frame(["2026-02-03T14:30:00Z"]),
    )
    available = loader.load_day(target_day, decision_times=["09:30"])
    assert list(available.state_rows["status"].astype(str)) == ["ok"]


def test_strategy_modeling_gap_and_overlap_policy_regressions() -> None:
    trades = [
        _trade(
            trade_id="gap-loss",
            symbol="SPY",
            entry_ts="2026-01-05T14:30:00Z",
            exit_ts="2026-01-05T15:00:00Z",
            entry_price=100.0,
            exit_price=90.0,  # Gap-through stop proxy: realized loss is worse than -1R.
            initial_risk=5.0,
        ),
        _trade(
            trade_id="overlap-same-symbol",
            symbol="SPY",
            entry_ts="2026-01-05T14:45:00Z",
            exit_ts="2026-01-05T15:30:00Z",
            entry_price=50.0,
            exit_price=55.0,
            initial_risk=5.0,
        ),
        _trade(
            trade_id="overlap-other-symbol",
            symbol="QQQ",
            entry_ts="2026-01-05T14:45:00Z",
            exit_ts="2026-01-05T15:30:00Z",
            entry_price=40.0,
            exit_price=44.0,
            initial_risk=4.0,
        ),
    ]

    default_result = build_strategy_portfolio_ledger(trades, starting_capital=50_000.0)
    assert set(default_result.accepted_trade_ids) == {"gap-loss", "overlap-other-symbol"}
    assert set(default_result.skipped_trade_ids) == {"overlap-same-symbol"}

    skip_reason_by_trade = {row.trade_id: row.skip_reason for row in default_result.ledger if row.event == "skip"}
    assert skip_reason_by_trade["overlap-same-symbol"] == "one_open_per_symbol"

    entry_row = next(row for row in default_result.ledger if row.event == "entry" and row.trade_id == "gap-loss")
    exit_row = next(row for row in default_result.ledger if row.event == "exit" and row.trade_id == "gap-loss")
    assert entry_row.risk_amount == pytest.approx(500.0)
    assert exit_row.realized_pnl == pytest.approx(-1000.0)
    assert exit_row.realized_pnl < -entry_row.risk_amount

    overlap_policy = StrategyModelingPolicyConfig(one_open_per_symbol=False)
    overlap_result = build_strategy_portfolio_ledger(
        trades,
        starting_capital=50_000.0,
        policy=overlap_policy,
    )
    assert set(overlap_result.accepted_trade_ids) == {
        "gap-loss",
        "overlap-same-symbol",
        "overlap-other-symbol",
    }
    assert overlap_result.skipped_trade_ids == ()


def test_apply_entry_filters_orb_confirmation_gate_shifts_entry_and_confirmation_timestamps() -> None:
    event = _strategy_event(
        event_id="evt-orb-pass",
        direction="long",
        signal_day="2026-01-05",
        entry_day="2026-01-06",
        signal_close=102.0,
        stop_price=95.0,
    )
    intraday = _intraday_frame(
        "2026-01-06",
        [
            ("09:30", 100.0, 102.0, 99.0, 101.0),
            ("09:35", 101.0, 103.0, 100.0, 102.0),
            ("09:40", 102.0, 104.0, 101.0, 103.0),
            ("09:45", 103.0, 106.0, 102.0, 105.0),
            ("09:50", 106.0, 107.0, 105.0, 106.0),
        ],
    )

    filtered, summary, metadata = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(
            enable_orb_confirmation=True,
            orb_range_minutes=15,
            orb_confirmation_cutoff_et="10:30",
            orb_stop_policy="orb_range",
        ),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={},
        daily_ohlc_by_symbol={},
        intraday_bars_by_symbol={"SPY": intraday},
    )

    assert len(filtered) == 1
    updated = filtered[0]
    expected_confirmed = _market_utc("2026-01-06", "09:50") - pd.Timedelta(microseconds=1)
    expected_entry = _market_utc("2026-01-06", "09:50")
    assert pd.Timestamp(updated.signal_confirmed_ts) == expected_confirmed
    assert pd.Timestamp(updated.entry_ts) == expected_entry
    assert pd.Timestamp(updated.entry_ts) > pd.Timestamp(updated.signal_confirmed_ts)
    assert updated.stop_price == 99.0
    assert summary["base_event_count"] == 1
    assert summary["kept_event_count"] == 1
    assert summary["rejected_event_count"] == 0
    assert summary["reject_counts"]["orb_opening_range_missing"] == 0
    assert summary["reject_counts"]["orb_breakout_missing"] == 0
    assert metadata["parsed_orb_range_minutes"] == 15
    assert metadata["parsed_orb_confirmation_cutoff_et"]["hour"] == 10
    assert metadata["parsed_orb_confirmation_cutoff_et"]["minute"] == 30


def test_apply_entry_filters_reject_reason_shorts_disabled() -> None:
    event = _strategy_event(event_id="evt-short", direction="short", stop_price=101.0)
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(allow_shorts=False),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={},
        daily_ohlc_by_symbol={},
        intraday_bars_by_symbol={},
    )

    assert filtered == ()
    _assert_single_reject(summary, "shorts_disabled")


def test_apply_entry_filters_reject_reason_missing_daily_context() -> None:
    event = _strategy_event(event_id="evt-missing-daily")
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(enable_rsi_extremes=True),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={},
        daily_ohlc_by_symbol={},
        intraday_bars_by_symbol={},
    )

    assert filtered == ()
    _assert_single_reject(summary, "missing_daily_context")


def test_apply_entry_filters_reject_reason_rsi_not_extreme() -> None:
    event = _strategy_event(event_id="evt-rsi-fail")
    features = _daily_features_frame([("2026-01-05", 50.0, 2.0, 99.0, 1.0, "normal")])
    ohlc = _daily_ohlc_frame([("2026-01-05", 101.0)])
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(enable_rsi_extremes=True),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={"SPY": features},
        daily_ohlc_by_symbol={"SPY": ohlc},
        intraday_bars_by_symbol={},
    )

    assert filtered == ()
    _assert_single_reject(summary, "rsi_not_extreme")


def test_apply_entry_filters_reject_reason_ema9_regime_mismatch() -> None:
    event = _strategy_event(event_id="evt-ema-fail")
    features = _daily_features_frame(
        [
            ("2026-01-04", 20.0, 2.0, 106.0, 0.0, "normal"),
            ("2026-01-05", 20.0, 2.0, 105.0, 0.0, "normal"),
        ]
    )
    ohlc = _daily_ohlc_frame(
        [
            ("2026-01-04", 107.0),
            ("2026-01-05", 101.0),
        ]
    )
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(enable_ema9_regime=True, ema9_slope_lookback_bars=1),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={"SPY": features},
        daily_ohlc_by_symbol={"SPY": ohlc},
        intraday_bars_by_symbol={},
    )

    assert filtered == ()
    _assert_single_reject(summary, "ema9_regime_mismatch")


def test_apply_entry_filters_reject_reason_volatility_regime_disallowed() -> None:
    event = _strategy_event(event_id="evt-vol-fail")
    features = _daily_features_frame([("2026-01-05", 20.0, 2.0, 99.0, 1.0, "normal")])
    ohlc = _daily_ohlc_frame([("2026-01-05", 101.0)])
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(
            enable_volatility_regime=True,
            allowed_volatility_regimes=("high",),
        ),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={"SPY": features},
        daily_ohlc_by_symbol={"SPY": ohlc},
        intraday_bars_by_symbol={},
    )

    assert filtered == ()
    _assert_single_reject(summary, "volatility_regime_disallowed")


def test_apply_entry_filters_reject_reason_orb_opening_range_missing() -> None:
    event = _strategy_event(
        event_id="evt-orb-range-missing",
        direction="long",
        signal_day="2026-01-05",
        entry_day="2026-01-06",
    )
    intraday = _intraday_frame(
        "2026-01-06",
        [
            ("09:45", 100.0, 101.0, 99.0, 100.0),
            ("09:50", 100.0, 101.0, 99.0, 100.0),
        ],
    )
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(enable_orb_confirmation=True),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={},
        daily_ohlc_by_symbol={},
        intraday_bars_by_symbol={"SPY": intraday},
    )

    assert filtered == ()
    _assert_single_reject(summary, "orb_opening_range_missing")


def test_apply_entry_filters_reject_reason_orb_breakout_missing() -> None:
    event = _strategy_event(
        event_id="evt-orb-breakout-missing",
        direction="long",
        signal_day="2026-01-05",
        entry_day="2026-01-06",
    )
    intraday = _intraday_frame(
        "2026-01-06",
        [
            ("09:30", 100.0, 102.0, 99.0, 101.0),
            ("09:35", 101.0, 103.0, 100.0, 102.0),
            ("09:40", 102.0, 104.0, 101.0, 103.0),
            ("09:45", 103.0, 103.8, 102.0, 103.2),
            ("09:50", 103.2, 103.5, 102.8, 103.1),
        ],
    )
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(enable_orb_confirmation=True),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={},
        daily_ohlc_by_symbol={},
        intraday_bars_by_symbol={"SPY": intraday},
    )

    assert filtered == ()
    _assert_single_reject(summary, "orb_breakout_missing")


def test_apply_entry_filters_reject_reason_atr_floor_failed() -> None:
    event = _strategy_event(event_id="evt-atr-floor-fail", signal_close=100.0, stop_price=99.0)
    features = _daily_features_frame([("2026-01-05", 20.0, 2.0, 99.0, 1.0, "normal")])
    ohlc = _daily_ohlc_frame([("2026-01-05", 101.0)])
    filtered, summary, _ = apply_entry_filters(
        [event],
        filter_config=StrategyEntryFilterConfig(
            enable_atr_stop_floor=True,
            atr_stop_floor_multiple=2.0,
        ),
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={"SPY": features},
        daily_ohlc_by_symbol={"SPY": ohlc},
        intraday_bars_by_symbol={},
    )

    assert filtered == ()
    _assert_single_reject(summary, "atr_floor_failed")


def test_apply_entry_filters_deterministic_reject_ordering_and_kept_order() -> None:
    short_event = _strategy_event(event_id="evt-short-reject", direction="short", stop_price=101.0)
    rsi_event = _strategy_event(event_id="evt-rsi-reject", direction="long", signal_day="2026-01-05")
    keep_a = _strategy_event(event_id="evt-keep-a", direction="long", signal_day="2026-01-06")
    keep_b = _strategy_event(event_id="evt-keep-b", direction="long", signal_day="2026-01-07")

    features = _daily_features_frame(
        [
            ("2026-01-05", 50.0, 2.0, 99.0, 1.0, "normal"),
            ("2026-01-06", 20.0, 2.0, 99.0, 1.0, "normal"),
            ("2026-01-07", 25.0, 2.0, 99.0, 1.0, "normal"),
        ]
    )
    ohlc = _daily_ohlc_frame(
        [
            ("2026-01-05", 101.0),
            ("2026-01-06", 102.0),
            ("2026-01-07", 103.0),
        ]
    )
    config = StrategyEntryFilterConfig(
        allow_shorts=False,
        enable_rsi_extremes=True,
    )

    filtered_a, summary_a, _ = apply_entry_filters(
        [rsi_event, keep_b, short_event, keep_a],
        filter_config=config,
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={"SPY": features},
        daily_ohlc_by_symbol={"SPY": ohlc},
        intraday_bars_by_symbol={},
    )
    filtered_b, summary_b, _ = apply_entry_filters(
        [keep_a, short_event, keep_b, rsi_event],
        filter_config=config,
        feature_config=StrategyFeatureConfig(),
        daily_features_by_symbol={"SPY": features},
        daily_ohlc_by_symbol={"SPY": ohlc},
        intraday_bars_by_symbol={},
    )

    assert summary_a == summary_b
    assert [event.event_id for event in filtered_a] == ["evt-keep-a", "evt-keep-b"]
    assert [event.event_id for event in filtered_b] == ["evt-keep-a", "evt-keep-b"]
    assert summary_a["reject_counts"]["shorts_disabled"] == 1
    assert summary_a["reject_counts"]["rsi_not_extreme"] == 1
    assert list(summary_a["reject_counts"]) == list(FILTER_REJECT_REASONS)
