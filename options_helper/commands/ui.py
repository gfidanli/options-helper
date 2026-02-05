from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Streamlit portal commands.")


def register(root_app: typer.Typer) -> None:
    root_app.add_typer(app, name="ui")


def _default_streamlit_app_path() -> Path:
    return Path(__file__).resolve().parents[2] / "apps" / "streamlit" / "streamlit_app.py"


def _streamlit_installed() -> bool:
    return importlib.util.find_spec("streamlit") is not None


def _launch_ui(*, host: str, port: int, path: Path) -> None:
    streamlit_app = path.expanduser().resolve()
    if not streamlit_app.exists() or not streamlit_app.is_file():
        typer.secho(f"Streamlit app not found at: {streamlit_app}", fg=typer.colors.RED, err=True)
        typer.secho("Pass a valid app script path via --path.", err=True)
        raise typer.Exit(2)

    if not _streamlit_installed():
        typer.secho(
            'Streamlit is not installed. Install UI extras with `pip install -e ".[ui]"`.',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

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
        "--server.headless",
        "true",
        "--server.showEmailPrompt",
        "false",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.callback(invoke_without_command=True)
def ui(
    ctx: typer.Context,
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
    path: Path = typer.Option(
        _default_streamlit_app_path(),
        "--path",
        help="Path to the Streamlit app entrypoint script.",
    ),
) -> None:
    """Launch the Streamlit portal."""
    if ctx.invoked_subcommand is not None:
        return
    _launch_ui(host=host, port=port, path=path)


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
    path: Path = typer.Option(
        _default_streamlit_app_path(),
        "--path",
        help="Path to the Streamlit app entrypoint script.",
    ),
) -> None:
    """Launch the optional Streamlit portal."""
    _launch_ui(host=host, port=port, path=path)
