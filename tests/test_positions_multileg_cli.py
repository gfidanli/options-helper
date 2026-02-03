from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.models import MultiLegPosition
from options_helper.storage import load_portfolio


def test_add_spread_cli_adds_multileg(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        """
        {
          "cash": 0,
          "positions": []
        }
        """,
        encoding="utf-8",
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "add-spread",
            str(portfolio_path),
            "--symbol",
            "AAPL",
            "--leg",
            "long,call,2026-04-17,100,1",
            "--leg",
            "short,call,2026-04-17,105,1",
            "--net-debit",
            "120",
            "--id",
            "spread-1",
        ],
    )

    assert res.exit_code == 0, res.output

    portfolio = load_portfolio(portfolio_path)
    assert len(portfolio.positions) == 1
    assert isinstance(portfolio.positions[0], MultiLegPosition)
    assert portfolio.positions[0].id == "spread-1"


def test_remove_position_cli_removes_multileg(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        """
        {
          "cash": 0,
          "positions": [
            {
              "id": "spread-2",
              "symbol": "AAPL",
              "net_debit": 80,
              "legs": [
                {"side": "long", "option_type": "call", "expiry": "2026-04-17", "strike": 100, "contracts": 1},
                {"side": "short", "option_type": "call", "expiry": "2026-04-17", "strike": 105, "contracts": 1}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    runner = CliRunner()
    res = runner.invoke(app, ["remove-position", str(portfolio_path), "spread-2"])
    assert res.exit_code == 0, res.output

    portfolio = load_portfolio(portfolio_path)
    assert portfolio.positions == []
