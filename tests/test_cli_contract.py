from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from typer.testing import CliRunner


def _assert_tokens(output: str, tokens: list[str]) -> None:
    missing = [t for t in tokens if t not in output]
    assert not missing, f"Missing tokens in help output: {missing}\n\nOutput:\n{output}"


def test_cli_help_includes_core_commands() -> None:
    from options_helper.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0, res.output

    core_commands = [
        "init",
        "list",
        "add-position",
        "add-spread",
        "remove-position",
        "analyze",
        "research",
        "daily",
        "snapshot-options",
        "flow",
        "chain-report",
        "compare",
        "report-pack",
        "briefing",
        "dashboard",
        "roll-plan",
        "earnings",
        "refresh-earnings",
        "refresh-candles",
        "watch",
    ]
    command_groups = [
        "watchlists",
        "derived",
        "technicals",
        "scanner",
        "journal",
        "backtest",
    ]

    _assert_tokens(res.output, core_commands + command_groups)


def test_analyze_help_has_offline_contract_flags() -> None:
    from options_helper.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["analyze", "--help"])
    assert res.exit_code == 0, res.output

    expected_flags = [
        "--offline",
        "--as-of",
        "--offline-strict",
        "--snapshots-dir",
        "--stress-spot-pct",
        "--stress-vol-pp",
        "--stress-days",
    ]
    _assert_tokens(res.output, expected_flags)


def test_cli_help_avoids_backtest_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys
        from typer.testing import CliRunner

        from options_helper.cli import app

        heavy_modules = [
            "options_helper.technicals_backtesting.backtest.optimizer",
            "options_helper.technicals_backtesting.backtest.walk_forward",
            "options_helper.backtesting.runner",
            "options_helper.backtesting.data_source",
        ]

        runner = CliRunner()
        res = runner.invoke(app, ["--help"])
        if res.exit_code != 0:
            print(res.output)
            raise SystemExit(res.exit_code)

        imported = [m for m in heavy_modules if m in sys.modules]
        if imported:
            raise SystemExit(f"Heavy modules imported during --help: {imported}")
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")
