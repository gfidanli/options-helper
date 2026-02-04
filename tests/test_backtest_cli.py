from __future__ import annotations

from datetime import date
from pathlib import Path

import os

from typer.testing import CliRunner

from options_helper.backtesting.ledger import TradeLogRow
from options_helper.backtesting.runner import BacktestRun
from options_helper.cli import app


def test_backtest_run_cli_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    run = BacktestRun(
        symbol="AAA",
        contract_symbol="AAA260220C00100000",
        start=None,
        end=None,
        fill_mode="worst_case",
        slippage_factor=0.0,
        initial_cash=10000.0,
        trades=[
            TradeLogRow(
                symbol="AAA",
                contract_symbol="AAA260220C00100000",
                expiry=date(2026, 2, 20),
                strike=100.0,
                option_type="call",
                quantity=1,
                entry_date=date(2026, 1, 2),
                entry_price=1.0,
                exit_date=date(2026, 1, 3),
                exit_price=1.2,
                holding_days=1,
                pnl=20.0,
                pnl_pct=0.2,
                max_favorable=50.0,
                max_adverse=-10.0,
            )
        ],
        skips=[],
        open_position=None,
        rolls=[],
    )

    def _fake_run_backtest(*_args, **_kwargs):  # noqa: ANN001
        return run

    monkeypatch.setattr(
        "options_helper.commands.backtest._resolve_run_backtest",
        lambda: _fake_run_backtest,
    )

    reports_dir = tmp_path / "reports"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "--symbol",
            "AAA",
            "--contract-symbol",
            "AAA260220C00100000",
            "--reports-dir",
            str(reports_dir),
            "--run-id",
            "test-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (reports_dir / "test-run" / "summary.json").exists()
    assert (reports_dir / "test-run" / "report.md").exists()
    assert (reports_dir / "test-run" / "trades.csv").exists()


def test_backtest_report_latest(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    run1 = reports_dir / "run1"
    run2 = reports_dir / "run2"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)

    report1 = run1 / "report.md"
    report2 = run2 / "report.md"
    report1.write_text("# run1\n", encoding="utf-8")
    report2.write_text("# run2\n", encoding="utf-8")

    os.utime(report1, (1000, 1000))
    os.utime(report2, (2000, 2000))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "backtest",
            "report",
            "--latest",
            "--reports-dir",
            str(reports_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "# run2" in result.output
