from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app


def test_cli_log_path_creates_and_appends(tmp_path: Path) -> None:
    runner = CliRunner()
    log_dir = tmp_path / "logs"
    log_path = tmp_path / "combined" / "job.log"
    watchlists_path = tmp_path / "watchlists.json"

    for _ in range(2):
        result = runner.invoke(
            app,
            [
                "--log-dir",
                str(log_dir),
                "--log-path",
                str(log_path),
                "watchlists",
                "list",
                "--path",
                str(watchlists_path),
            ],
        )
        assert result.exit_code == 0, result.output

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert content.count("INFO options_helper.cli: Start") == 2
    assert content.count("INFO options_helper.cli: End") == 2

    if log_dir.exists():
        assert not list(log_dir.rglob("*.log")), "expected no per-run log files when --log-path is set"
