from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from options_helper.analysis.msb import compute_msb_signals, extract_msb_events
from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events
from options_helper.analysis.strategy_portfolio import build_strategy_portfolio_ledger
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.zero_dte_dataset import ZeroDTEIntradayDatasetLoader
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
