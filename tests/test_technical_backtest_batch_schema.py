from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from options_helper.schemas.technical_backtest_batch import (
    TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION,
    TechnicalBacktestBatchSummaryArtifact,
    validate_technical_backtest_batch_summary_payload,
)


def _writer_payload() -> dict[str, object]:
    return {
        "schema_version": TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION,
        "generated_at": datetime(2026, 3, 3, 14, 0, tzinfo=timezone.utc),
        "as_of": date(2026, 3, 2),
        "run_id": "batch-run-001",
        "strategy": "MeanReversionIBS",
        "benchmark_symbol": "SPY",
        "requested_symbols": ("SPY", "QQQ"),
        "modeled_symbols": ("SPY", "QQQ"),
        "failed_symbols": [],
        "period_start": date(2025, 1, 2),
        "period_end": date(2026, 3, 2),
        "aggregate_metrics": {
            "starting_equity": 100_000.0,
            "ending_equity": 110_000.0,
            "total_return_pct": 0.10,
            "annualized_return_pct": 0.08,
            "max_drawdown_pct": -0.07,
            "trade_count": 120,
            "win_rate": 0.70,
            "profit_factor": 1.6,
            "invested_pct": 0.21,
        },
        "benchmark_metrics": {
            "starting_equity": 100_000.0,
            "ending_equity": 106_000.0,
            "total_return_pct": 0.06,
            "annualized_return_pct": 0.05,
            "max_drawdown_pct": -0.09,
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
                    "ending_equity": 55_300.0,
                    "total_return_pct": 0.106,
                    "annualized_return_pct": 0.084,
                    "max_drawdown_pct": -0.069,
                    "trade_count": 63,
                    "win_rate": 0.714,
                    "profit_factor": 1.69,
                    "invested_pct": 0.23,
                },
            },
            {
                "symbol": "QQQ",
                "period_start": date(2025, 1, 2),
                "period_end": date(2026, 3, 2),
                "metrics": {
                    "starting_equity": 50_000.0,
                    "ending_equity": 54_700.0,
                    "total_return_pct": 0.094,
                    "annualized_return_pct": 0.076,
                    "max_drawdown_pct": -0.072,
                    "trade_count": 57,
                    "win_rate": 0.684,
                    "profit_factor": 1.53,
                    "invested_pct": 0.19,
                },
            },
        ],
        "equity_curve": [
            {
                "session_date": date(2026, 2, 28),
                "aggregate_equity": 109_800.0,
                "benchmark_equity": 105_700.0,
                "aggregate_drawdown_pct": -0.035,
                "benchmark_drawdown_pct": -0.051,
            },
            {
                "session_date": date(2026, 3, 2),
                "aggregate_equity": 110_000.0,
                "benchmark_equity": 106_000.0,
                "aggregate_drawdown_pct": -0.033,
                "benchmark_drawdown_pct": -0.049,
            },
        ],
        "monthly_returns": [
            {"year": 2026, "month": 1, "aggregate_return_pct": 0.02, "benchmark_return_pct": 0.015},
            {"year": 2026, "month": 2, "aggregate_return_pct": 0.017, "benchmark_return_pct": 0.01},
        ],
        "yearly_returns": [
            {"year": 2025, "aggregate_return_pct": 0.071, "benchmark_return_pct": 0.046},
            {"year": 2026, "aggregate_return_pct": 0.027, "benchmark_return_pct": 0.014},
        ],
        "warnings": [],
    }


@pytest.mark.parametrize(
    "missing_key",
    (
        "run_id",
        "strategy",
        "requested_symbols",
        "modeled_symbols",
        "period_start",
        "period_end",
        "aggregate_metrics",
        "benchmark_metrics",
        "per_symbol_metrics",
        "equity_curve",
        "monthly_returns",
        "yearly_returns",
    ),
)
def test_technical_backtest_batch_summary_requires_contract_sections(missing_key: str) -> None:
    payload = _writer_payload()
    payload.pop(missing_key)

    with pytest.raises(ValidationError):
        TechnicalBacktestBatchSummaryArtifact.model_validate(payload)


def test_technical_backtest_batch_summary_rejects_unknown_field() -> None:
    payload = _writer_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        TechnicalBacktestBatchSummaryArtifact.model_validate(payload)


def test_technical_backtest_batch_summary_rejects_unknown_nested_field() -> None:
    payload = _writer_payload()
    aggregate_metrics = payload["aggregate_metrics"]
    assert isinstance(aggregate_metrics, dict)
    aggregate_metrics["unexpected"] = "x"

    with pytest.raises(ValidationError):
        TechnicalBacktestBatchSummaryArtifact.model_validate(payload)


def test_validate_technical_backtest_batch_summary_payload_accepts_writer_payload() -> None:
    artifact = validate_technical_backtest_batch_summary_payload(_writer_payload())
    serialized = artifact.to_dict()

    assert artifact.schema_version == TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION
    assert serialized["schema_version"] == TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION
    assert serialized["requested_symbols"] == ["SPY", "QQQ"]
    assert serialized["benchmark_symbol"] == "SPY"


def test_validate_technical_backtest_batch_summary_supports_payload_without_optional_version_fields() -> None:
    payload = _writer_payload()
    payload.pop("schema_version")
    payload.pop("generated_at")
    payload.pop("disclaimer", None)

    artifact = validate_technical_backtest_batch_summary_payload(payload)
    serialized = artifact.to_dict()

    assert artifact.schema_version == TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION
    assert artifact.generated_at.tzinfo == timezone.utc
    assert serialized["schema_version"] == TECHNICAL_BACKTEST_BATCH_SCHEMA_VERSION
    assert serialized["disclaimer"] == "Not financial advice. For informational/educational use only."
