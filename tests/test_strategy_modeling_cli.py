from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.strategy_modeling_artifacts import DISCLAIMER_TEXT


class _StubStrategyModelingService:
    def __init__(self, *, blocked: bool = False) -> None:
        self.blocked = blocked
        self.last_request = None
        self.universe_calls: list[object] = []

    def list_universe_loader(self, *, database_path=None):  # noqa: ANN001
        self.universe_calls.append(database_path)
        return SimpleNamespace(symbols=["SPY", "QQQ", "IWM"], notes=[])

    def run(self, request):  # noqa: ANN001
        self.last_request = request
        preflight = _build_preflight(blocked=self.blocked)
        return _build_run_result(request=request, preflight=preflight)


def _build_preflight(*, blocked: bool):  # noqa: ANN202
    required_day = date(2026, 1, 5)
    coverage = {
        "SPY": SimpleNamespace(
            symbol="SPY",
            required_days=(required_day,),
            covered_days=() if blocked else (required_day,),
            missing_days=(required_day,) if blocked else (),
        )
    }
    return SimpleNamespace(
        require_intraday_bars=True,
        coverage_by_symbol=coverage,
        blocked_symbols=["SPY"] if blocked else [],
        notes=["SPY missing intraday coverage"] if blocked else [],
    )


def _build_run_result(*, request, preflight):  # noqa: ANN001
    portfolio_metrics = SimpleNamespace(
        starting_capital=float(request.starting_capital),
        ending_capital=float(request.starting_capital) + 250.0,
        total_return_pct=2.5,
        cagr_pct=None,
        max_drawdown_pct=-1.2,
        sharpe_ratio=1.4,
        sortino_ratio=None,
        calmar_ratio=None,
        profit_factor=1.6,
        expectancy_r=0.22,
        avg_realized_r=0.18,
        trade_count=3,
        win_rate=66.67,
        loss_rate=33.33,
        avg_hold_bars=2.0,
        exposure_pct=14.0,
    )
    target_hit_rates = [
        SimpleNamespace(
            target_label="1.2R",
            target_r=1.2,
            trade_count=3,
            hit_count=2,
            hit_rate=66.67,
            avg_bars_to_hit=2.0,
            median_bars_to_hit=2.0,
            expectancy_r=0.22,
        )
    ]
    segment_records = [
        SimpleNamespace(
            segment_dimension="symbol",
            segment_value="SPY",
            trade_count=3,
            win_rate=66.67,
            avg_realized_r=0.18,
            expectancy_r=0.22,
            profit_factor=1.6,
            sharpe_ratio=1.4,
            max_drawdown_pct=-1.2,
        ),
        SimpleNamespace(
            segment_dimension="direction",
            segment_value="long",
            trade_count=2,
            win_rate=100.0,
            avg_realized_r=0.5,
            expectancy_r=0.5,
            profit_factor=None,
            sharpe_ratio=None,
            max_drawdown_pct=0.0,
        ),
        SimpleNamespace(
            segment_dimension="direction",
            segment_value="short",
            trade_count=1,
            win_rate=0.0,
            avg_realized_r=-0.4,
            expectancy_r=-0.4,
            profit_factor=0.0,
            sharpe_ratio=None,
            max_drawdown_pct=-0.4,
        ),
    ]
    return SimpleNamespace(
        strategy=request.strategy,
        as_of=request.end_date,
        requested_symbols=tuple(request.symbols or ()),
        modeled_symbols=tuple(request.symbols or ()),
        signal_events=("evt-1", "evt-2"),
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
                "status": "closed",
                "signal_confirmed_ts": "2026-01-07T20:55:00+00:00",
                "entry_ts": "2026-01-07T21:00:00+00:00",
                "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
                "entry_price": 102.0,
                "stop_price": 100.0,
                "target_price": 104.0,
                "exit_ts": "2026-01-07T21:05:00+00:00",
                "exit_price": 99.5,
                "exit_reason": "stop_gap",
                "initial_risk": 2.0,
                "realized_r": -1.25,
                "gap_fill_applied": True,
                "reject_code": None,
            },
            {
                "trade_id": "tr-3",
                "event_id": "evt-3",
                "symbol": "SPY",
                "direction": "long",
                "status": "rejected",
                "reject_code": "one_open_per_symbol",
            },
        ),
        accepted_trade_ids=("tr-1", "tr-2"),
        skipped_trade_ids=("tr-3",),
        intraday_preflight=preflight,
        portfolio_metrics=portfolio_metrics,
        target_hit_rates=target_hit_rates,
        segment_records=segment_records,
        filter_summary={
            "base_event_count": 5,
            "kept_event_count": 2,
            "rejected_event_count": 3,
            "reject_counts": {
                "shorts_disabled": 1,
                "rsi_not_extreme": 2,
                "orb_breakout_missing": 0,
            },
        },
        directional_metrics={
            "combined": {
                "trade_count": 2,
                "portfolio_metrics": {"total_return_pct": 2.5},
            },
            "long_only": {
                "trade_count": 1,
                "portfolio_metrics": {"total_return_pct": 5.0},
            },
            "short_only": {
                "trade_count": 1,
                "portfolio_metrics": {"total_return_pct": -2.0},
            },
        },
    )


def test_strategy_model_command_is_registered_under_technicals() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["technicals", "--help"])
    assert res.exit_code == 0, res.output
    assert "strategy-model" in res.output


def test_strategy_model_rejects_invalid_strategy_value() -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--strategy",
            "invalid",
            "--symbols",
            "SPY",
        ],
    )
    assert res.exit_code != 0
    assert "--strategy must be one of:" in res.output


def test_strategy_model_rejects_invalid_orb_confirmation_cutoff() -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--symbols",
            "SPY",
            "--orb-confirmation-cutoff-et",
            "10-30",
        ],
    )
    assert res.exit_code != 0
    assert "--orb-confirmation-cutoff-et must be HH:MM" in res.output
    assert "24-hour ET" in res.output


def test_strategy_model_rejects_invalid_allowed_volatility_regimes() -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--symbols",
            "SPY",
            "--allowed-volatility-regimes",
            "low,extreme",
        ],
    )
    assert res.exit_code != 0
    assert "--allowed-volatility-regimes contains invalid regime(s):" in res.output
    assert "extreme" in res.output


def test_strategy_model_accepts_orb_strategy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=False)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--strategy",
            "orb",
            "--symbols",
            "SPY",
        ],
    )

    assert res.exit_code == 0, res.output
    assert stub.last_request is not None
    assert stub.last_request.strategy == "orb"


def test_strategy_model_success_parses_options_and_runs_service(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=False)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--strategy",
            "msb",
            "--no-allow-shorts",
            "--enable-orb-confirmation",
            "--orb-range-minutes",
            "20",
            "--orb-confirmation-cutoff-et",
            "10:15",
            "--orb-stop-policy",
            "tighten",
            "--enable-atr-stop-floor",
            "--atr-stop-floor-multiple",
            "0.8",
            "--enable-rsi-extremes",
            "--enable-ema9-regime",
            "--ema9-slope-lookback-bars",
            "5",
            "--enable-volatility-regime",
            "--allowed-volatility-regimes",
            "low,normal",
            "--symbols",
            "spy,qqq",
            "--exclude-symbols",
            "qqq",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-31",
            "--intraday-timeframe",
            "5Min",
            "--r-ladder-min-tenths",
            "12",
            "--r-ladder-max-tenths",
            "16",
            "--r-ladder-step-tenths",
            "2",
            "--starting-capital",
            "25000",
            "--risk-per-trade-pct",
            "2.5",
            "--signal-confirmation-lag-bars",
            "2",
            "--segment-dimensions",
            "symbol,direction",
            "--segment-values",
            "spy,long",
            "--segment-min-trades",
            "2",
            "--segment-limit",
            "5",
        ],
    )

    assert res.exit_code == 0, res.output
    req = stub.last_request
    assert req is not None
    assert req.strategy == "msb"
    assert tuple(req.symbols) == ("SPY",)
    assert req.start_date.isoformat() == "2026-01-01"
    assert req.end_date.isoformat() == "2026-01-31"
    assert req.intraday_timeframe == "5Min"
    assert [target.label for target in req.target_ladder] == ["1.2R", "1.4R", "1.6R"]
    assert [round(target.target_r, 1) for target in req.target_ladder] == [1.2, 1.4, 1.6]
    assert req.starting_capital == 25000.0
    assert req.policy["risk_per_trade_pct"] == 2.5
    assert req.policy["gap_fill_policy"] == "fill_at_open"
    assert req.policy["sizing_rule"] == "risk_pct_of_equity"
    assert req.policy["entry_ts_anchor_policy"] == "first_tradable_bar_open_after_signal_confirmed_ts"
    assert req.signal_confirmation_lag_bars == 2
    assert req.output_timezone == "America/Chicago"
    assert req.segment_dimensions == ("symbol", "direction")
    assert req.segment_values == ("spy", "long")
    assert req.segment_min_trades == 2
    assert req.segment_limit == 5
    assert req.filter_config.allow_shorts is False
    assert req.filter_config.enable_orb_confirmation is True
    assert req.filter_config.orb_range_minutes == 20
    assert req.filter_config.orb_confirmation_cutoff_et == "10:15"
    assert req.filter_config.orb_stop_policy == "tighten"
    assert req.filter_config.enable_atr_stop_floor is True
    assert req.filter_config.atr_stop_floor_multiple == 0.8
    assert req.filter_config.enable_rsi_extremes is True
    assert req.filter_config.enable_ema9_regime is True
    assert req.filter_config.ema9_slope_lookback_bars == 5
    assert req.filter_config.enable_volatility_regime is True
    assert req.filter_config.allowed_volatility_regimes == ("low", "normal")
    assert "strategy=msb" in res.output
    assert "segments_shown=3" in res.output
    assert "filters base=5 kept=2 rejected=3" in res.output
    assert "filter_rejects shorts_disabled=1, rsi_not_extreme=2" in res.output
    assert "directional combined trades=2 return=2.50%" in res.output
    assert "long_only trades=1 return=5.00%" in res.output
    assert "short_only trades=1 return=-2.00%" in res.output


def test_strategy_model_success_resolves_universe_filters(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=False)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--exclude-symbols",
            "qqq",
            "--universe-limit",
            "1",
        ],
    )

    assert res.exit_code == 0, res.output
    assert stub.universe_calls == [None]
    assert tuple(stub.last_request.symbols) == ("SPY",)
    assert stub.last_request.filter_config.allow_shorts is True
    assert stub.last_request.filter_config.enable_orb_confirmation is False
    assert stub.last_request.filter_config.enable_atr_stop_floor is False
    assert stub.last_request.filter_config.enable_rsi_extremes is False
    assert stub.last_request.filter_config.enable_ema9_regime is False
    assert stub.last_request.filter_config.enable_volatility_regime is False


def test_strategy_model_filters_fail_when_no_symbols_remain(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=False)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--exclude-symbols",
            "spy,qqq,iwm",
        ],
    )

    assert res.exit_code != 0
    assert "No symbols remain after applying include/exclude/universe" in res.output
    assert "filters." in res.output
    assert stub.last_request is None


def test_strategy_model_writes_artifacts_and_summary_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=False)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--strategy",
            "msb",
            "--symbols",
            "SPY",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-31",
            "--starting-capital",
            "50000",
            "--risk-per-trade-pct",
            "3.0",
            "--signal-confirmation-lag-bars",
            "3",
            "--segment-limit",
            "1",
            "--out",
            str(tmp_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "strategy=msb symbols=1 trades=3 segments_shown=1" in res.output
    assert "Wrote summary JSON:" in res.output
    assert "Wrote trades CSV:" in res.output
    assert "Wrote R ladder CSV:" in res.output
    assert "Wrote segments CSV:" in res.output
    assert "Wrote summary Markdown:" in res.output

    run_dir = tmp_path / "msb" / "2026-01-31"
    summary_path = run_dir / "summary.json"
    trades_path = run_dir / "trades.csv"
    r_ladder_path = run_dir / "r_ladder.csv"
    segments_path = run_dir / "segments.csv"
    summary_md_path = run_dir / "summary.md"

    assert summary_path.exists()
    assert trades_path.exists()
    assert r_ladder_path.exists()
    assert segments_path.exists()
    assert summary_md_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "schema_version",
        "generated_at",
        "strategy",
        "requested_symbols",
        "modeled_symbols",
        "disclaimer",
        "summary",
        "policy_metadata",
        "metrics",
        "r_ladder",
        "segments",
        "trade_log",
    }
    assert payload["schema_version"] == 1
    assert payload["strategy"] == "msb"
    assert payload["disclaimer"] == DISCLAIMER_TEXT
    assert payload["requested_symbols"] == ["SPY"]
    assert payload["modeled_symbols"] == ["SPY"]
    assert payload["summary"] == {
        "signal_event_count": 2,
        "trade_count": 3,
        "accepted_trade_count": 2,
        "skipped_trade_count": 1,
        "losses_below_minus_one_r": 1,
    }
    assert payload["policy_metadata"]["entry_anchor"] == "next_bar_open"
    assert payload["policy_metadata"]["signal_confirmation_lag_bars"] == 3
    assert payload["policy_metadata"]["require_intraday_bars"] is True
    assert payload["policy_metadata"]["risk_per_trade_pct"] == 3.0
    assert payload["policy_metadata"]["sizing_rule"] == "risk_pct_of_equity"
    assert payload["policy_metadata"]["gap_fill_policy"] == "fill_at_open"
    assert payload["policy_metadata"]["intra_bar_tie_break_rule"] == "stop_first"
    assert payload["policy_metadata"]["output_timezone"] == "America/Chicago"
    assert payload["policy_metadata"]["entry_ts_anchor_policy"] == (
        "first_tradable_bar_open_after_signal_confirmed_ts"
    )
    assert payload["metrics"]["starting_capital"] == 50000.0
    assert payload["metrics"]["trade_count"] == 3
    assert len(payload["r_ladder"]) == 1
    assert len(payload["segments"]) == 3
    assert len(payload["trade_log"]) == 3
    assert any(row.get("loss_below_1r") is True for row in payload["trade_log"])

    with trades_path.open("r", encoding="utf-8", newline="") as handle:
        trade_rows = list(csv.DictReader(handle))
    with r_ladder_path.open("r", encoding="utf-8", newline="") as handle:
        r_ladder_rows = list(csv.DictReader(handle))
    with segments_path.open("r", encoding="utf-8", newline="") as handle:
        segment_rows = list(csv.DictReader(handle))

    assert len(trade_rows) == 3
    assert len(r_ladder_rows) == 1
    assert len(segment_rows) == 3

    summary_md = summary_md_path.read_text(encoding="utf-8")
    assert DISCLAIMER_TEXT in summary_md
    assert "Not financial advice." in summary_md


def test_strategy_model_artifact_write_flags_are_respected(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=False)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--strategy",
            "sfp",
            "--symbols",
            "SPY",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-31",
            "--out",
            str(tmp_path),
            "--no-write-csv",
            "--no-write-md",
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Wrote summary JSON:" in res.output
    assert "Wrote trades CSV:" not in res.output
    assert "Wrote summary Markdown:" not in res.output

    run_dir = tmp_path / "sfp" / "2026-01-31"
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "trades.csv").exists()
    assert not (run_dir / "r_ladder.csv").exists()
    assert not (run_dir / "segments.csv").exists()
    assert not (run_dir / "summary.md").exists()


def test_strategy_model_fails_fast_when_intraday_coverage_is_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    stub = _StubStrategyModelingService(blocked=True)
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: stub,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--symbols",
            "SPY",
        ],
    )

    assert res.exit_code != 0
    assert "Missing required intraday coverage for requested scope" in res.output
    assert "SPY(1/1 missing)" in res.output
