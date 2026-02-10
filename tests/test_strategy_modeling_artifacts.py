from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from types import SimpleNamespace

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.strategy_modeling_artifacts import (
    DISCLAIMER_TEXT,
    write_strategy_modeling_artifacts,
)


def _sample_request() -> SimpleNamespace:
    return SimpleNamespace(
        strategy="sfp",
        symbols=("SPY",),
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
        end_date=datetime(2026, 1, 31, tzinfo=timezone.utc).date(),
        intraday_timeframe="5Min",
        intraday_source="alpaca",
        signal_confirmation_lag_bars=2,
        output_timezone="America/Chicago",
        policy={
            "require_intraday_bars": True,
            "risk_per_trade_pct": 1.0,
            "gap_fill_policy": "fill_at_open",
            "sizing_rule": "risk_pct_of_equity",
            "entry_ts_anchor_policy": "first_tradable_bar_open_after_signal_confirmed_ts",
        },
        intra_bar_tie_break_rule="stop_first",
    )


def _sample_run_result() -> SimpleNamespace:
    return SimpleNamespace(
        strategy="sfp",
        requested_symbols=("SPY",),
        modeled_symbols=("SPY",),
        signal_events=("evt-1",),
        accepted_trade_ids=("tr-1",),
        skipped_trade_ids=(),
        intraday_preflight=SimpleNamespace(
            require_intraday_bars=True,
            blocked_symbols=[],
            coverage_by_symbol={},
            notes=[],
        ),
        portfolio_metrics=SimpleNamespace(
            starting_capital=10_000.0,
            ending_capital=9_750.0,
            total_return_pct=-2.5,
            trade_count=1,
            avg_realized_r=-1.25,
        ),
        target_hit_rates=(
            SimpleNamespace(
                target_label="1.0R",
                target_r=1.0,
                trade_count=1,
                hit_count=0,
                hit_rate=0.0,
                avg_bars_to_hit=None,
                median_bars_to_hit=None,
                expectancy_r=-1.25,
            ),
        ),
        segment_records=(
            SimpleNamespace(
                segment_dimension="symbol",
                segment_value="SPY",
                trade_count=1,
                win_rate=0.0,
                avg_realized_r=-1.25,
                expectancy_r=-1.25,
                profit_factor=0.0,
                sharpe_ratio=None,
                max_drawdown_pct=-2.5,
            ),
        ),
        filter_metadata={
            "allow_shorts": True,
            "active_filters": ["rsi_extremes"],
            "reject_reason_order": [
                "shorts_disabled",
                "missing_daily_context",
                "rsi_not_extreme",
                "ema9_regime_mismatch",
                "volatility_regime_disallowed",
                "orb_opening_range_missing",
                "orb_breakout_missing",
                "atr_floor_failed",
            ],
        },
        filter_summary={
            "base_event_count": 2,
            "kept_event_count": 1,
            "rejected_event_count": 1,
            "reject_counts": {
                "shorts_disabled": 0,
                "missing_daily_context": 0,
                "rsi_not_extreme": 1,
                "ema9_regime_mismatch": 0,
                "volatility_regime_disallowed": 0,
                "orb_opening_range_missing": 0,
                "orb_breakout_missing": 0,
                "atr_floor_failed": 0,
            },
        },
        directional_metrics={
            "counts": {
                "all_simulated_trade_count": 1,
                "portfolio_subset_trade_count": 1,
                "portfolio_subset_closed_trade_count": 1,
                "portfolio_subset_long_trade_count": 1,
                "portfolio_subset_short_trade_count": 0,
            },
            "portfolio_target": {
                "target_label": "1.0R",
                "target_r": 1.0,
                "selection_source": "preferred_target_label",
            },
            "combined": {
                "simulated_trade_count": 1,
                "closed_trade_count": 1,
                "accepted_trade_count": 1,
                "skipped_trade_count": 0,
                "portfolio_metrics": {
                    "total_return_pct": -0.025,
                    "avg_realized_r": -1.25,
                },
            },
            "long_only": {
                "simulated_trade_count": 1,
                "closed_trade_count": 1,
                "accepted_trade_count": 1,
                "skipped_trade_count": 0,
                "portfolio_metrics": {
                    "total_return_pct": -0.025,
                    "avg_realized_r": -1.25,
                },
            },
            "short_only": {
                "simulated_trade_count": 0,
                "closed_trade_count": 0,
                "accepted_trade_count": 0,
                "skipped_trade_count": 0,
                "portfolio_metrics": {
                    "total_return_pct": 0.0,
                    "avg_realized_r": None,
                },
            },
        },
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
                "stop_price": 98.0,
                "target_price": 102.0,
                "exit_ts": "2026-01-06T21:05:00+00:00",
                "exit_price": 97.5,
                "exit_reason": "stop_gap",
                "initial_risk": 2.0,
                "realized_r": -1.25,
                "gap_fill_applied": True,
                "reject_code": None,
            },
        ),
    )


def _assert_required_summary_keys(payload: dict[str, object]) -> None:
    required_top_level_keys = {
        "schema_version",
        "generated_at",
        "strategy",
        "requested_symbols",
        "modeled_symbols",
        "disclaimer",
        "summary",
        "policy_metadata",
        "filter_metadata",
        "filter_summary",
        "directional_metrics",
        "metrics",
        "r_ladder",
        "segments",
        "trade_log",
        "trade_review",
    }
    assert required_top_level_keys.issubset(payload.keys())

    summary = payload.get("summary")
    assert isinstance(summary, dict)
    required_summary_keys = {
        "signal_event_count",
        "trade_count",
        "accepted_trade_count",
        "skipped_trade_count",
        "losses_below_minus_one_r",
    }
    assert required_summary_keys.issubset(summary.keys())

    trade_review = payload.get("trade_review")
    assert isinstance(trade_review, dict)
    required_trade_review_keys = {
        "metric",
        "scope",
        "candidate_trade_count",
        "top_best_count",
        "top_worst_count",
        "top_best",
        "top_worst",
    }
    assert required_trade_review_keys.issubset(trade_review.keys())


def test_strategy_modeling_artifacts_include_required_metadata_and_disclaimer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = write_strategy_modeling_artifacts(
        out_dir=tmp_path,
        strategy="sfp",
        request=_sample_request(),
        run_result=_sample_run_result(),
        generated_at=datetime(2026, 2, 8, 12, 30, tzinfo=timezone.utc),
    )

    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    _assert_required_summary_keys(payload)
    assert payload["schema_version"] == 1
    assert payload["disclaimer"] == DISCLAIMER_TEXT
    assert payload["policy_metadata"]["entry_anchor"] == "next_bar_open"
    assert payload["policy_metadata"]["signal_confirmation_lag_bars"] == 2
    assert payload["policy_metadata"]["require_intraday_bars"] is True
    assert payload["policy_metadata"]["gap_fill_policy"] == "fill_at_open"
    assert payload["policy_metadata"]["intra_bar_tie_break_rule"] == "stop_first"
    assert payload["policy_metadata"]["output_timezone"] == "America/Chicago"
    assert payload["generated_at"] == "2026-02-08T06:30:00-06:00"
    assert payload["summary"]["losses_below_minus_one_r"] == 1
    assert payload["filter_metadata"]["active_filters"] == ["rsi_extremes"]
    assert payload["filter_summary"]["base_event_count"] == 2
    assert payload["filter_summary"]["reject_counts"]["rsi_not_extreme"] == 1
    assert payload["directional_metrics"]["portfolio_target"]["target_label"] == "1.0R"
    assert payload["directional_metrics"]["combined"]["portfolio_metrics"]["total_return_pct"] == -0.025
    assert payload["trade_log"][0]["signal_confirmed_ts"] == "2026-01-06T14:55:00-06:00"
    assert payload["trade_log"][0]["entry_ts"] == "2026-01-06T15:00:00-06:00"
    assert payload["trade_log"][0]["exit_ts"] == "2026-01-06T15:05:00-06:00"
    assert payload["trade_review"]["metric"] == "realized_r"
    assert payload["trade_review"]["scope"] == "accepted_closed_trades"
    assert payload["trade_review"]["candidate_trade_count"] == 1
    assert payload["trade_review"]["top_best_count"] == 1
    assert payload["trade_review"]["top_worst_count"] == 1
    assert payload["trade_review"]["top_best"][0]["trade_id"] == "tr-1"
    assert payload["trade_review"]["top_worst"][0]["trade_id"] == "tr-1"

    markdown = paths.summary_md.read_text(encoding="utf-8")
    assert DISCLAIMER_TEXT in markdown
    assert "## Filters" in markdown
    assert "## Directional Results" in markdown
    assert "## Trade Review" in markdown
    assert "- Scope: `accepted_closed_trades`" in markdown
    assert "- Candidate trades / best / worst: `1` / `1` / `1`" in markdown
    assert "Combined:" in markdown
    assert "Long-only:" in markdown
    assert "Short-only:" in markdown
    assert "Reject reasons: `rsi_not_extreme=1`" in markdown
    assert "Not financial advice." in markdown
    assert "Gap-through stop fills can produce realized losses below `-1.0R`" in markdown

    llm_prompt = paths.llm_analysis_prompt_md.read_text(encoding="utf-8")
    assert "Strategy Modeling LLM Analysis Prompt" in llm_prompt
    assert "Files To Read" in llm_prompt
    assert "`summary.json`" in llm_prompt
    assert "`top_20_best_trades.csv`" in llm_prompt
    assert "`top_20_worst_trades.csv`" in llm_prompt
    assert "Not financial advice." in llm_prompt


def test_strategy_modeling_trade_csv_surfaces_gap_and_stop_slippage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = write_strategy_modeling_artifacts(
        out_dir=tmp_path,
        strategy="sfp",
        request=_sample_request(),
        run_result=_sample_run_result(),
        generated_at=datetime(2026, 2, 8, 12, 30, tzinfo=timezone.utc),
    )

    with paths.trade_log_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["signal_confirmed_ts"] == "2026-01-06T14:55:00-06:00"
    assert row["entry_ts"] == "2026-01-06T15:00:00-06:00"
    assert row["exit_ts"] == "2026-01-06T15:05:00-06:00"
    assert row["realized_r"] == "-1.25"
    assert row["gap_fill_applied"] == "True"
    assert row["gap_exit"] == "True"
    assert row["stop_slippage_r"] == "-0.25"
    assert row["loss_below_1r"] == "True"

    with paths.top_best_trades_csv.open("r", encoding="utf-8", newline="") as handle:
        best_rows = list(csv.DictReader(handle))
    with paths.top_worst_trades_csv.open("r", encoding="utf-8", newline="") as handle:
        worst_rows = list(csv.DictReader(handle))
    assert [row["rank"] for row in best_rows] == ["1"]
    assert [row["rank"] for row in worst_rows] == ["1"]
    assert [row["trade_id"] for row in best_rows] == ["tr-1"]
    assert [row["trade_id"] for row in worst_rows] == ["tr-1"]


def test_strategy_modeling_trade_review_scope_uses_fallback_only_when_accepted_ids_missing(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    empty_scope_result = _sample_run_result()
    empty_scope_result.accepted_trade_ids = ()
    empty_scope_paths = write_strategy_modeling_artifacts(
        out_dir=tmp_path / "empty_scope",
        strategy="sfp",
        request=_sample_request(),
        run_result=empty_scope_result,
        generated_at=datetime(2026, 2, 8, 12, 30, tzinfo=timezone.utc),
    )
    empty_scope_payload = json.loads(empty_scope_paths.summary_json.read_text(encoding="utf-8"))
    empty_scope_review = empty_scope_payload["trade_review"]
    assert empty_scope_review["scope"] == "accepted_closed_trades"
    assert empty_scope_review["candidate_trade_count"] == 0
    assert empty_scope_review["top_best_count"] == 0
    assert empty_scope_review["top_worst_count"] == 0
    assert empty_scope_review["top_best"] == []
    assert empty_scope_review["top_worst"] == []

    with empty_scope_paths.top_best_trades_csv.open("r", encoding="utf-8", newline="") as handle:
        empty_best_rows = list(csv.DictReader(handle))
    with empty_scope_paths.top_worst_trades_csv.open("r", encoding="utf-8", newline="") as handle:
        empty_worst_rows = list(csv.DictReader(handle))
    assert empty_best_rows == []
    assert empty_worst_rows == []

    fallback_scope_result = _sample_run_result()
    delattr(fallback_scope_result, "accepted_trade_ids")
    fallback_scope_paths = write_strategy_modeling_artifacts(
        out_dir=tmp_path / "fallback_scope",
        strategy="sfp",
        request=_sample_request(),
        run_result=fallback_scope_result,
        generated_at=datetime(2026, 2, 8, 12, 30, tzinfo=timezone.utc),
    )
    fallback_scope_payload = json.loads(fallback_scope_paths.summary_json.read_text(encoding="utf-8"))
    fallback_scope_review = fallback_scope_payload["trade_review"]
    assert fallback_scope_review["scope"] == "closed_nonrejected_trades"
    assert fallback_scope_review["candidate_trade_count"] == 1
    assert fallback_scope_review["top_best_count"] == 1
    assert fallback_scope_review["top_worst_count"] == 1
    assert fallback_scope_review["top_best"][0]["trade_id"] == "tr-1"
    assert fallback_scope_review["top_worst"][0]["trade_id"] == "tr-1"


def test_strategy_model_cli_writes_strategy_modeling_artifacts(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_result = _sample_run_result()

    class _StubService:
        def list_universe_loader(self, *, database_path=None):  # noqa: ANN001
            return SimpleNamespace(symbols=["SPY", "QQQ"], notes=[])

        def run(self, request):  # noqa: ANN001
            return run_result

    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: _StubService(),
    )

    runner = CliRunner()
    result = runner.invoke(
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert "strategy=sfp" in result.output

    summary_path = tmp_path / "sfp" / "2026-01-31" / "summary.json"
    top_best_path = tmp_path / "sfp" / "2026-01-31" / "top_20_best_trades.csv"
    top_worst_path = tmp_path / "sfp" / "2026-01-31" / "top_20_worst_trades.csv"
    llm_prompt_path = tmp_path / "sfp" / "2026-01-31" / "llm_analysis_prompt.md"
    assert summary_path.exists()
    assert top_best_path.exists()
    assert top_worst_path.exists()
    assert llm_prompt_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    _assert_required_summary_keys(payload)
    assert payload["disclaimer"] == DISCLAIMER_TEXT
    assert payload["summary"]["losses_below_minus_one_r"] == 1
