from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from options_helper.data.storage_runtime import get_default_duckdb_path

app = typer.Typer(help="DuckDB warehouse utilities.")


def _resolve_duckdb_path(duckdb_path: Path | None) -> Path:
    if duckdb_path is None:
        return get_default_duckdb_path()
    return Path(duckdb_path)


@app.command("init")
def db_init(
    duckdb_path: Path | None = typer.Option(
        None,
        "--duckdb-path",
        help="DuckDB file path (defaults to data/warehouse/options.duckdb).",
    ),
) -> None:
    """Ensure the DuckDB schema exists (idempotent)."""
    from options_helper.db.migrations import ensure_schema
    from options_helper.db.warehouse import DuckDBWarehouse

    console = Console()
    path = _resolve_duckdb_path(duckdb_path)
    try:
        info = ensure_schema(DuckDBWarehouse(path))
        console.print(f"{info.path} schema v{info.schema_version}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("info")
def db_info(
    duckdb_path: Path | None = typer.Option(
        None,
        "--duckdb-path",
        help="DuckDB file path (defaults to data/warehouse/options.duckdb).",
    ),
) -> None:
    """Show DuckDB schema version (0 if uninitialized)."""
    from options_helper.db.migrations import current_schema_version
    from options_helper.db.warehouse import DuckDBWarehouse

    console = Console()
    path = _resolve_duckdb_path(duckdb_path)
    try:
        version = 0
        if path.exists():
            version = current_schema_version(DuckDBWarehouse(path))
        console.print(f"{path} schema v{version}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
