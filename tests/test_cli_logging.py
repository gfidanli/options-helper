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
    logs = list(log_dir.rglob("*.log"))
    assert logs, "expected log file in log dir"
    content = logs[0].read_text(encoding="utf-8")
    assert "Start" in content
    assert "End" in content


def test_cli_logging_moves_legacy_flat_logs_to_date_folder(tmp_path: Path) -> None:
    runner = CliRunner()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    watchlists_path = tmp_path / "watchlists.json"

    legacy_log = log_dir / "watchlists_list_20260115T020000Z_123.log"
    legacy_log.write_text("2026-01-14 20:00:00,000 INFO options_helper.cli: Start watchlists list\n", encoding="utf-8")

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

    assert not legacy_log.exists()
    migrated_log = log_dir / "2026-01-14" / legacy_log.name
    assert migrated_log.exists(), "expected legacy root log to be moved into date partition folder"
