from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.commands import ui as ui_command


def test_ui_help_lists_launcher_flags() -> None:
    runner = CliRunner()

    res = runner.invoke(app, ["ui", "--help"])

    assert res.exit_code == 0, res.output
    assert "Streamlit portal commands" in res.output
    assert "--host" in res.output
    assert "--port" in res.output
    assert "--path" in res.output


def test_ui_launcher_shells_out_to_streamlit(monkeypatch, tmp_path: Path) -> None:
    app_path = tmp_path / "streamlit_app.py"
    app_path.write_text("import streamlit as st\nst.write('hi')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["check"] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ui_command, "_streamlit_installed", lambda: True)
    monkeypatch.setattr(ui_command.subprocess, "run", _fake_run)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "ui",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--path",
            str(app_path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert captured["command"] == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path.resolve()),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8765",
    ]
    assert captured["check"] is False


def test_ui_missing_streamlit_dependency_shows_install_guidance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_path = tmp_path / "streamlit_app.py"
    app_path.write_text("pass\n", encoding="utf-8")

    monkeypatch.setattr(ui_command, "_streamlit_installed", lambda: False)

    def _unexpected_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("subprocess.run should not be called when streamlit is missing")

    monkeypatch.setattr(ui_command.subprocess, "run", _unexpected_run)

    runner = CliRunner()
    res = runner.invoke(app, ["ui", "--path", str(app_path)])

    assert res.exit_code == 1
    assert "Streamlit is not installed" in res.output
    assert 'pip install -e ".[ui]"' in res.output
