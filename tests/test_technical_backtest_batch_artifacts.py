from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path

from options_helper.data.technical_backtest_batch_artifacts import (
    write_technical_backtest_batch_artifacts,
)
from options_helper.schemas.technical_backtest_batch import (
    TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION,
    validate_technical_backtest_batch_summary_payload,
)


def _summary_payload(*, run_id: str = "mean-reversion-ibs-001") -> dict[str, object]:
    return {
        "schema_version": TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION,
        "generated_at": datetime(2026, 3, 3, 14, 0, tzinfo=timezone.utc),
        "as_of": date(2026, 3, 2),
        "run_id": run_id,
        "strategy": "MeanReversionIBS",
        "benchmark_symbol": "SPY",
        "requested_symbols": ["SPY", "QQQ", "IWM"],
        "modeled_symbols": ["SPY", "QQQ"],
        "failed_symbols": [
            {
                "symbol": "IWM",
                "stage": "backtest",
                "error": "missing cached candles for requested date range",
            }
        ],
        "period_start": date(2025, 1, 2),
        "period_end": date(2026, 3, 2),
        "aggregate_metrics": {
            "starting_equity": 100_000.0,
            "ending_equity": 114_350.0,
            "total_return_pct": 0.1435,
            "annualized_return_pct": 0.1154,
            "max_drawdown_pct": -0.0821,
            "trade_count": 124,
            "win_rate": 0.7016,
            "profit_factor": 1.79,
            "invested_pct": 0.214,
        },
        "benchmark_metrics": {
            "starting_equity": 100_000.0,
            "ending_equity": 109_120.0,
            "total_return_pct": 0.0912,
            "annualized_return_pct": 0.0731,
            "max_drawdown_pct": -0.109,
            "trade_count": 0,
            "win_rate": None,
            "profit_factor": None,
            "invested_pct": 1.0,
        },
        "per_symbol_metrics": [
            {
                "symbol": "SPY",
                "period_start": date(2025, 1, 2),
                "period_end": date(2026, 3, 2),
                "metrics": {
                    "starting_equity": 50_000.0,
                    "ending_equity": 57_200.0,
                    "total_return_pct": 0.144,
                    "annualized_return_pct": 0.1162,
                    "max_drawdown_pct": -0.079,
                    "trade_count": 67,
                    "win_rate": 0.7164,
                    "profit_factor": 1.85,
                    "invested_pct": 0.227,
                },
            },
            {
                "symbol": "QQQ",
                "period_start": date(2025, 1, 2),
                "period_end": date(2026, 3, 2),
                "metrics": {
                    "starting_equity": 50_000.0,
                    "ending_equity": 57_150.0,
                    "total_return_pct": 0.143,
                    "annualized_return_pct": 0.1147,
                    "max_drawdown_pct": -0.0853,
                    "trade_count": 57,
                    "win_rate": 0.6842,
                    "profit_factor": 1.72,
                    "invested_pct": 0.201,
                },
            },
        ],
        "equity_curve": [
            {
                "session_date": date(2026, 2, 27),
                "aggregate_equity": 113_900.0,
                "benchmark_equity": 108_770.0,
                "aggregate_drawdown_pct": -0.0345,
                "benchmark_drawdown_pct": -0.0514,
            },
            {
                "session_date": date(2026, 3, 2),
                "aggregate_equity": 114_350.0,
                "benchmark_equity": 109_120.0,
                "aggregate_drawdown_pct": -0.032,
                "benchmark_drawdown_pct": -0.0488,
            },
        ],
        "monthly_returns": [
            {"year": 2026, "month": 1, "aggregate_return_pct": 0.021, "benchmark_return_pct": 0.014},
            {"year": 2026, "month": 2, "aggregate_return_pct": 0.018, "benchmark_return_pct": 0.011},
        ],
        "yearly_returns": [
            {"year": 2025, "aggregate_return_pct": 0.104, "benchmark_return_pct": 0.063},
            {"year": 2026, "aggregate_return_pct": 0.036, "benchmark_return_pct": 0.027},
        ],
        "warnings": ["1 symbol failed and was excluded from aggregate analytics."],
        "disclaimer": "Not financial advice. For informational/educational use only.",
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_write_technical_backtest_batch_artifacts_persists_required_files(tmp_path: Path) -> None:
    paths = write_technical_backtest_batch_artifacts(
        reports_dir=tmp_path,
        summary_payload=_summary_payload(),
    )

    assert paths.run_dir == tmp_path / "mean-reversion-ibs-001"
    assert paths.summary_json.exists()
    assert paths.report_html.exists()
    assert paths.per_symbol_metrics_csv.exists()
    assert paths.equity_curve_csv.exists()
    assert paths.monthly_returns_csv.exists()
    assert paths.yearly_returns_csv.exists()
    assert paths.failed_symbols_csv.exists()

    summary_payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    artifact = validate_technical_backtest_batch_summary_payload(summary_payload)
    assert artifact.schema_version == TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION
    assert artifact.run_id == "mean-reversion-ibs-001"
    assert [item.symbol for item in artifact.per_symbol_metrics] == ["SPY", "QQQ"]

    assert len(_read_rows(paths.per_symbol_metrics_csv)) == 2
    assert len(_read_rows(paths.equity_curve_csv)) == 2
    assert len(_read_rows(paths.monthly_returns_csv)) == 2
    assert len(_read_rows(paths.yearly_returns_csv)) == 2
    failed_rows = _read_rows(paths.failed_symbols_csv)
    assert failed_rows == [
        {
            "symbol": "IWM",
            "stage": "backtest",
            "error": "missing cached candles for requested date range",
        }
    ]

    report_html = paths.report_html.read_text(encoding="utf-8")
    assert "<h1>Technical Backtest Batch Report</h1>" in report_html
    assert 'id="headline-metrics"' in report_html
    assert 'id="equity-drawdown"' in report_html
    assert 'id="monthly-returns"' in report_html
    assert 'id="yearly-returns"' in report_html
    assert "<h2>Headline Metrics</h2>" in report_html
    assert "<h2>Equity + Drawdown (Strategy vs SPY)</h2>" in report_html
    assert "<h2>Monthly Returns</h2>" in report_html
    assert "<h2>Yearly Returns</h2>" in report_html
    assert "Total Return" in report_html
    assert "Win Rate" in report_html
    assert "Profit Factor" in report_html
    assert "Max Drawdown" in report_html
    assert "Run ID: <strong>mean-reversion-ibs-001</strong>" in report_html
    assert "Not financial advice. For informational/educational use only." in report_html


def test_write_technical_backtest_batch_artifacts_handles_empty_partial_datasets(tmp_path: Path) -> None:
    payload = _summary_payload(run_id="mean-reversion-ibs-empty")
    payload["modeled_symbols"] = []
    payload["per_symbol_metrics"] = []
    payload["equity_curve"] = []
    payload["monthly_returns"] = []
    payload["yearly_returns"] = []
    payload["aggregate_metrics"] = {
        "starting_equity": 100_000.0,
        "ending_equity": 100_000.0,
        "total_return_pct": 0.0,
        "annualized_return_pct": None,
        "max_drawdown_pct": None,
        "trade_count": 0,
        "win_rate": None,
        "profit_factor": None,
        "invested_pct": 0.0,
    }

    paths = write_technical_backtest_batch_artifacts(
        reports_dir=tmp_path,
        summary_payload=payload,
    )

    rows = _read_rows(paths.per_symbol_metrics_csv)
    assert rows == []
    assert len(_read_rows(paths.failed_symbols_csv)) == 1

    report_html = paths.report_html.read_text(encoding="utf-8")
    assert "<h2>Headline Metrics</h2>" in report_html
    assert "<h2>Equity + Drawdown (Strategy vs SPY)</h2>" in report_html
    assert "<h2>Monthly Returns</h2>" in report_html
    assert "<h2>Yearly Returns</h2>" in report_html
    assert "No equity history available." in report_html
    assert report_html.count("No rows available.") == 2
