from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from typer.testing import CliRunner

from options_helper.analysis.msb import compute_msb_signals, extract_msb_events
from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_events
from options_helper.cli import app


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


def _run_scan(
    *,
    command_name: str,
    frame: pd.DataFrame,
    tmp_path: Path,
) -> dict[str, Any]:
    ohlc_path = tmp_path / f"{command_name}.csv"
    out_dir = tmp_path / f"{command_name}_reports"
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
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _normalize_scan_event(event: dict[str, Any], *, strategy: str) -> dict[str, Any]:
    level_key = "sweep_level" if strategy == "sfp" else "break_level"
    swing_ts_key = "swept_swing_timestamp" if strategy == "sfp" else "broken_swing_timestamp"
    return {
        "timestamp": event.get("timestamp"),
        "direction": event.get("direction"),
        "candle_open": event.get("candle_open"),
        "candle_high": event.get("candle_high"),
        "candle_low": event.get("candle_low"),
        "candle_close": event.get("candle_close"),
        "signal_level": event.get(level_key),
        "swing_timestamp": event.get(swing_ts_key),
        "bars_since_swing": event.get("bars_since_swing"),
        "entry_anchor_timestamp": event.get("entry_anchor_timestamp"),
        "entry_anchor_price": event.get("entry_anchor_price"),
        "forward_returns_pct": event.get("forward_returns_pct"),
    }


def _build_stub_run_result(request: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=request.strategy,
        requested_symbols=tuple(request.symbols),
        modeled_symbols=tuple(request.symbols),
        signal_events=("evt-1", "evt-2"),
        accepted_trade_ids=("tr-1",),
        skipped_trade_ids=("tr-2",),
        intraday_preflight=SimpleNamespace(
            require_intraday_bars=True,
            blocked_symbols=[],
            coverage_by_symbol={},
            notes=[],
        ),
        portfolio_metrics=SimpleNamespace(
            starting_capital=float(request.starting_capital),
            ending_capital=float(request.starting_capital) + 50.0,
            total_return_pct=0.05,
            trade_count=2,
            avg_realized_r=0.1,
        ),
        target_hit_rates=(
            SimpleNamespace(
                target_label="1.0R",
                target_r=1.0,
                trade_count=2,
                hit_count=1,
                hit_rate=50.0,
                avg_bars_to_hit=2.0,
                median_bars_to_hit=2.0,
                expectancy_r=0.1,
            ),
        ),
        segment_records=(
            SimpleNamespace(
                segment_dimension="symbol",
                segment_value="SPY",
                trade_count=2,
                win_rate=50.0,
                avg_realized_r=0.1,
                expectancy_r=0.1,
                profit_factor=1.2,
                sharpe_ratio=0.4,
                max_drawdown_pct=-0.5,
            ),
            SimpleNamespace(
                segment_dimension="volatility_regime",
                segment_value="unknown",
                trade_count=2,
                win_rate=50.0,
                avg_realized_r=0.1,
                expectancy_r=0.1,
                profit_factor=1.2,
                sharpe_ratio=0.4,
                max_drawdown_pct=-0.5,
            ),
        ),
        trade_simulations=(
            {
                "trade_id": "tr-1",
                "event_id": "evt-1",
                "symbol": "SPY",
                "direction": "long",
                "status": "closed",
                "signal_confirmed_ts": "2026-01-06T20:55:00+00:00",
                "entry_ts": "2026-01-06T21:00:00+00:00",
                "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
                "entry_price": 100.0,
                "stop_price": 99.0,
                "target_price": 101.0,
                "exit_ts": "2026-01-06T21:05:00+00:00",
                "exit_price": 100.5,
                "exit_reason": "time_stop",
                "initial_risk": 1.0,
                "realized_r": 0.5,
                "gap_fill_applied": False,
                "reject_code": None,
            },
            {
                "trade_id": "tr-2",
                "event_id": "evt-2",
                "symbol": "SPY",
                "direction": "long",
                "status": "rejected",
                "reject_code": "one_open_per_symbol",
            },
        ),
    )


class _StubStrategyModelingService:
    def __init__(self) -> None:
        self.requests: list[SimpleNamespace] = []

    def list_universe_loader(self, *, database_path=None):  # noqa: ANN001
        return SimpleNamespace(symbols=["SPY", "QQQ"], notes=[])

    def run(self, request):  # noqa: ANN001
        self.requests.append(request)
        return _build_stub_run_result(request)


def _run_strategy_model_command(*, strategy: str, out_dir: Path) -> str:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--strategy",
            strategy,
            "--symbols",
            "SPY",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-31",
            "--segment-dimensions",
            "symbol,volatility_regime",
            "--segment-values",
            "spy,unknown",
            "--segment-min-trades",
            "1",
            "--segment-limit",
            "10",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


def _request_snapshot(request: SimpleNamespace) -> dict[str, Any]:
    return {
        "strategy": request.strategy,
        "symbols": tuple(request.symbols),
        "start_date": request.start_date.isoformat() if request.start_date else None,
        "end_date": request.end_date.isoformat() if request.end_date else None,
        "intraday_timeframe": request.intraday_timeframe,
        "intraday_source": request.intraday_source,
        "target_ladder": tuple(round(float(target.target_r), 2) for target in request.target_ladder),
        "starting_capital": float(request.starting_capital),
        "policy": dict(request.policy),
        "gap_fill_policy": request.gap_fill_policy,
        "intra_bar_tie_break_rule": request.intra_bar_tie_break_rule,
        "segment_dimensions": tuple(request.segment_dimensions),
        "segment_values": tuple(request.segment_values),
        "segment_min_trades": int(request.segment_min_trades),
        "segment_limit": int(request.segment_limit),
    }


def test_msb_scan_registry_contract_parity_with_sfp_shared_fields(tmp_path: Path) -> None:
    sfp_frame = _sfp_scan_fixture()
    msb_frame = _msb_scan_fixture()

    sfp_payload = _run_scan(command_name="sfp-scan", frame=sfp_frame, tmp_path=tmp_path)
    msb_payload = _run_scan(command_name="msb-scan", frame=msb_frame, tmp_path=tmp_path)

    assert sfp_payload["config"]["forward_returns_entry_anchor"] == "next_bar_open"
    assert msb_payload["config"]["forward_returns_entry_anchor"] == "next_bar_open"

    sfp_signals = compute_sfp_signals(
        sfp_frame,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
    )
    sfp_events = extract_sfp_events(sfp_signals)
    assert [event["timestamp"] for event in sfp_payload["events"]] == [
        pd.Timestamp(event.timestamp).date().isoformat() for event in sfp_events
    ]

    msb_signals = compute_msb_signals(
        msb_frame,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
    )
    msb_events = extract_msb_events(msb_signals)
    assert [event["timestamp"] for event in msb_payload["events"]] == [
        pd.Timestamp(event.timestamp).date().isoformat() for event in msb_events
    ]

    normalized_sfp_events = [_normalize_scan_event(event, strategy="sfp") for event in sfp_payload["events"]]
    normalized_msb_events = [_normalize_scan_event(event, strategy="msb") for event in msb_payload["events"]]

    assert normalized_sfp_events
    assert normalized_msb_events
    assert set(normalized_sfp_events[0].keys()) == set(normalized_msb_events[0].keys())
    for event in normalized_msb_events:
        assert event["entry_anchor_timestamp"] is not None
        assert set((event["forward_returns_pct"] or {}).keys()) == {"1d", "5d", "10d"}


def test_msb_strategy_model_registry_service_and_segments_parity(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    service = _StubStrategyModelingService()
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: service,
    )

    out_dir = tmp_path / "strategy_model_reports"
    sfp_output = _run_strategy_model_command(strategy="sfp", out_dir=out_dir)
    msb_output = _run_strategy_model_command(strategy="msb", out_dir=out_dir)

    assert "strategy=sfp" in sfp_output
    assert "strategy=msb" in msb_output
    assert len(service.requests) == 2

    sfp_request, msb_request = service.requests
    sfp_snapshot = _request_snapshot(sfp_request)
    msb_snapshot = _request_snapshot(msb_request)

    assert sfp_snapshot["strategy"] == "sfp"
    assert msb_snapshot["strategy"] == "msb"
    sfp_snapshot.pop("strategy")
    msb_snapshot.pop("strategy")
    assert msb_snapshot == sfp_snapshot

    sfp_summary_path = out_dir / "sfp" / "2026-01-31" / "summary.json"
    msb_summary_path = out_dir / "msb" / "2026-01-31" / "summary.json"
    sfp_summary = json.loads(sfp_summary_path.read_text(encoding="utf-8"))
    msb_summary = json.loads(msb_summary_path.read_text(encoding="utf-8"))

    assert sfp_summary["strategy"] == "sfp"
    assert msb_summary["strategy"] == "msb"
    assert msb_summary["policy_metadata"]["entry_anchor"] == "next_bar_open"
    assert msb_summary["policy_metadata"]["entry_anchor"] == sfp_summary["policy_metadata"]["entry_anchor"]
    assert msb_summary["segments"] == sfp_summary["segments"]
