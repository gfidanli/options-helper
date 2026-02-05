from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Streamlit portal commands.")


def register(root_app: typer.Typer) -> None:
    root_app.add_typer(app, name="ui")


@app.command("run")
def run_ui(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host interface for the Streamlit server.",
    ),
    port: int = typer.Option(
        8501,
        "--port",
        min=1,
        max=65535,
        help="Port for the Streamlit server.",
    ),
) -> None:
    """Launch the optional Streamlit portal."""
    repo_root = Path(__file__).resolve().parents[2]
    streamlit_app = repo_root / "apps" / "streamlit" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
