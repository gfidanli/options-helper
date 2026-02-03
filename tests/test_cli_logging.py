from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app


def test_cli_logging_creates_log_file(tmp_path: Path) -> None:
    runner = CliRunner()
    log_dir = tmp_path / "logs"
    watchlists_path = tmp_path / "watchlists.json"
    result = runner.invoke(
        app,
        [
            "--log-dir",
            str(log_dir),
            "watchlists",
            "list",
            "--path",
            str(watchlists_path),
        ],
    )
    assert result.exit_code == 0, result.output
    logs = list(log_dir.glob("*.log"))
    assert logs, "expected log file in log dir"
    content = logs[0].read_text(encoding="utf-8")
    assert "Start" in content
    assert "End" in content
