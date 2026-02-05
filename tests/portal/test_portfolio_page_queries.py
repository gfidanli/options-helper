from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apps.streamlit.components.portfolio_page import (
    build_portfolio_risk_summary,
    build_positions_dataframe,
    load_portfolio_safe,
)


def _write_portfolio(path: Path) -> None:
    payload = {
        "cash": 10000.0,
        "positions": [
            {
                "id": "aapl-call",
                "symbol": "AAPL",
                "option_type": "call",
                "expiry": "2026-06-19",
                "strike": 200.0,
                "contracts": 1,
                "cost_basis": 2.5,
                "opened_at": "2026-01-02",
            },
            {
                "id": "msft-spread",
                "symbol": "MSFT",
                "net_debit": 1.25,
                "opened_at": "2026-01-03",
                "legs": [
                    {
                        "side": "short",
                        "option_type": "put",
                        "expiry": "2026-07-17",
                        "strike": 300.0,
                        "contracts": 1,
                    },
                    {
                        "side": "long",
                        "option_type": "put",
                        "expiry": "2026-07-17",
                        "strike": 280.0,
                        "contracts": 1,
                    },
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_briefing(reports_dir: Path) -> None:
    daily_dir = reports_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": "2026-02-05T00:00:00+00:00",
        "as_of": "2026-02-05",
        "disclaimer": "Not financial advice.",
        "report_date": "2026-02-05",
        "portfolio_path": "portfolio.json",
        "symbols": ["AAPL", "MSFT"],
        "top": 5,
        "technicals": {"source": "technical_backtesting", "config_path": None},
        "portfolio": {
            "exposure": {
                "positions": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
                "total_delta_shares": 120.5,
                "total_theta_dollars_per_day": -25.4,
                "total_vega_dollars_per_iv": 90.1,
                "missing_greeks": 1,
                "warnings": ["best-effort Greeks"],
            },
            "stress": [
                {"label": "spot -5%", "pnl": -150.0},
            ],
        },
        "sections": [{"symbol": "AAPL", "as_of": "2026-02-05"}],
    }
    (daily_dir / "2026-02-05.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_portfolio_safe_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    portfolio, resolved, error = load_portfolio_safe(missing)

    assert portfolio is None
    assert resolved == missing.resolve()
    assert error is not None
    assert "not found" in error.lower()


def test_build_positions_dataframe_handles_single_and_multileg(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    _write_portfolio(portfolio_path)

    portfolio, _resolved, error = load_portfolio_safe(portfolio_path)
    assert error is None
    assert portfolio is not None

    df = build_positions_dataframe(portfolio)
    assert isinstance(df, pd.DataFrame)
    assert list(df["id"]) == ["aapl-call", "msft-spread"]

    aapl = df[df["id"] == "aapl-call"].iloc[0].to_dict()
    assert aapl["structure"] == "single"
    assert aapl["option_type"] == "call"
    assert float(aapl["premium_at_risk"]) == 250.0

    spread = df[df["id"] == "msft-spread"].iloc[0].to_dict()
    assert spread["structure"] == "multi-leg"
    assert spread["option_type"] == "multi"
    assert float(spread["premium_at_risk"]) == 1.25
    assert "SP 2026-07-17 300" in str(spread["notes"]) or "SP 2026-07-17 300.0" in str(spread["notes"])


def test_build_portfolio_risk_summary_uses_briefing_when_available(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    reports_dir = tmp_path / "reports"
    _write_portfolio(portfolio_path)
    _write_briefing(reports_dir)

    portfolio, _resolved, error = load_portfolio_safe(portfolio_path)
    assert error is None
    assert portfolio is not None

    summary, note = build_portfolio_risk_summary(portfolio, reports_path=reports_dir)
    assert note is None
    assert summary["source"] == "briefing"
    assert summary["total_delta_shares"] == 120.5
    assert summary["total_theta_dollars_per_day"] == -25.4
    assert summary["total_vega_dollars_per_iv"] == 90.1
    assert summary["missing_greeks"] == 1
    # Keep computed fallbacks for values briefing does not provide.
    assert summary["cash"] == 10000.0
    assert summary["premium_at_risk"] is not None
    assert summary["stress"] == [{"label": "spot -5%", "pnl": -150.0}]
    assert summary["warnings"] == ["best-effort Greeks"]


def test_build_portfolio_risk_summary_falls_back_when_briefing_missing(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    _write_portfolio(portfolio_path)

    portfolio, _resolved, error = load_portfolio_safe(portfolio_path)
    assert error is None
    assert portfolio is not None

    summary, note = build_portfolio_risk_summary(portfolio, reports_path=tmp_path / "no-reports")
    assert summary["source"] == "computed"
    assert summary["position_count"] == 2
    assert summary["symbol_count"] == 2
    assert summary["cash"] == 10000.0
    assert note is not None
