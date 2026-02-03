from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from options_helper.analysis.chain_metrics import ChainTotals, GammaSummary, Walls
from options_helper.schemas.briefing import (
    BriefingArtifact,
    BriefingPortfolio,
    BriefingSection,
    BriefingTechnicals,
)
from options_helper.schemas.backtest import BacktestSummaryArtifact, BacktestSummaryStats
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.common import clean_nan
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.flow import FlowArtifact, FlowNetRow


def test_clean_nan_recurses() -> None:
    payload = {
        "a": float("nan"),
        "b": [1, np.nan, {"c": np.inf}],
        "d": pd.NA,
    }

    cleaned = clean_nan(payload)
    assert cleaned["a"] is None
    assert cleaned["b"][1] is None
    assert cleaned["b"][2]["c"] is None
    assert cleaned["d"] is None


def test_flow_artifact_to_dict_cleans_nan() -> None:
    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    artifact = FlowArtifact(
        schema_version=1,
        generated_at=ts,
        as_of="2026-01-02",
        symbol="AAA",
        from_date="2026-01-01",
        to_date="2026-01-02",
        window=1,
        group_by="contract",
        snapshot_dates=["2026-01-01", "2026-01-02"],
        net=[FlowNetRow(option_type="call", strike=100.0, delta_oi=float("nan"))],
    )

    payload = artifact.to_dict()
    assert payload["net"][0]["delta_oi"] is None


def test_compare_artifact_alias_and_clean() -> None:
    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    artifact = CompareArtifact(
        schema_version=1,
        generated_at=ts,
        as_of="2026-01-02",
        symbol="AAA",
        from_report={"symbol": "AAA"},
        to_report={"symbol": "AAA"},
        diff={"spot_change": np.nan},
    )

    payload = artifact.to_dict()
    assert "from" in payload
    assert payload["diff"]["spot_change"] is None


def test_chain_report_artifact_minimal() -> None:
    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    artifact = ChainReportArtifact(
        generated_at=ts,
        symbol="AAA",
        as_of="2026-01-02",
        spot=100.0,
        totals=ChainTotals(),
        walls_overall=Walls(),
        gamma=GammaSummary(),
    )

    payload = artifact.to_dict()
    assert payload["symbol"] == "AAA"
    assert payload["generated_at"].startswith("2026-01-02")


def test_briefing_artifact_minimal() -> None:
    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    artifact = BriefingArtifact(
        schema_version=1,
        generated_at=ts,
        as_of="2026-01-02",
        disclaimer="Not financial advice.",
        report_date="2026-01-02",
        portfolio_path="data/portfolio.json",
        symbols=["AAA"],
        top=3,
        technicals=BriefingTechnicals(source="technicals_backtesting", config_path=None),
        portfolio=BriefingPortfolio(exposure=None, stress=[]),
        sections=[BriefingSection(symbol="AAA", as_of="2026-01-02")],
    )

    payload = artifact.to_dict()
    assert payload["sections"][0]["symbol"] == "AAA"


def test_backtest_summary_artifact_minimal() -> None:
    artifact = BacktestSummaryArtifact(
        run_id="run-1",
        symbol="AAA",
        contract_symbol=None,
        start=None,
        end=None,
        strategy_name="baseline",
        fill_mode="worst_case",
        slippage_factor=0.0,
        quantity=1,
        initial_cash=10000.0,
        final_cash=10000.0,
        stats=BacktestSummaryStats(
            total_pnl=0.0,
            total_pnl_pct=0.0,
            trade_count=0,
            win_rate=None,
            avg_pnl=None,
            avg_pnl_pct=None,
        ),
    )

    payload = artifact.to_dict()
    assert payload["schema_version"] == 1
    assert payload["symbol"] == "AAA"
