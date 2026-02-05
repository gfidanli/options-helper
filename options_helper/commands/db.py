from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Sequence

import typer
from rich.console import Console
from rich.table import Table

from options_helper.data.storage_runtime import get_default_duckdb_path

app = typer.Typer(help="DuckDB warehouse utilities.")


def _resolve_duckdb_path(duckdb_path: Path | None) -> Path:
    if duckdb_path is None:
        return get_default_duckdb_path()
    return Path(duckdb_path)


def _json_default(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _stringify_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:  # noqa: BLE001
            pass
    text = str(value)
    if not text:
        return "-"
    return text


def _render_table(
    console: Console,
    *,
    title: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
    empty_text: str,
) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    if not rows:
        console.print(f"[yellow]{empty_text}[/yellow]")
        return

    table = Table(show_header=True)
    for label, _ in columns:
        table.add_column(label, overflow="fold")
    for row in rows:
        table.add_row(*[_stringify_cell(row.get(key)) for _, key in columns])
    console.print(table)


def _build_health_payload(
    *,
    path: Path,
    days: int,
    limit: int,
    stale_days: int | None,
) -> dict[str, Any]:
    from options_helper.data.observability_meta import (
        has_observability_tables,
        query_latest_runs,
        query_recent_check_failures,
        query_recent_failures,
        query_watermarks,
    )
    from options_helper.db.warehouse import DuckDBWarehouse

    payload: dict[str, Any] = {
        "database_path": str(path),
        "database_exists": path.exists(),
        "meta_tables_present": False,
        "filters": {
            "days": days,
            "limit": limit,
            "stale_days": stale_days,
        },
        "latest_runs": [],
        "recent_failures": [],
        "watermarks": [],
        "recent_failed_checks": [],
    }
    if not path.exists():
        return payload

    warehouse = DuckDBWarehouse(path)
    payload["meta_tables_present"] = has_observability_tables(warehouse)
    payload["latest_runs"] = query_latest_runs(warehouse, days=days, limit=limit)
    payload["recent_failures"] = query_recent_failures(warehouse, days=days, limit=limit)
    payload["watermarks"] = query_watermarks(warehouse, stale_days=stale_days, limit=limit)
    payload["recent_failed_checks"] = query_recent_check_failures(
        warehouse,
        days=days,
        limit=limit,
    )
    return payload


def _render_health(console: Console, payload: dict[str, Any]) -> None:
    database_path = payload["database_path"]
    database_exists = bool(payload["database_exists"])
    meta_tables_present = bool(payload["meta_tables_present"])
    filters = payload["filters"]
    stale_days = filters["stale_days"]
    stale_desc = "all" if stale_days is None else f">={stale_days}"

    console.print(f"DuckDB health: {database_path}")
    console.print(
        "Filters: days={days}, limit={limit}, stale_days={stale}".format(
            days=filters["days"],
            limit=filters["limit"],
            stale=stale_desc,
        )
    )

    if not database_exists:
        console.print("[yellow]DuckDB file does not exist yet.[/yellow]")
    elif not meta_tables_present:
        console.print("[yellow]Observability tables (meta.*) are missing.[/yellow]")

    _render_table(
        console,
        title="Latest run per job",
        rows=payload["latest_runs"],
        columns=[
            ("job", "job_name"),
            ("status", "status"),
            ("started_at", "started_at"),
            ("ended_at", "ended_at"),
            ("duration_ms", "duration_ms"),
            ("provider", "provider"),
            ("storage", "storage_backend"),
        ],
        empty_text="No run history available.",
    )
    _render_table(
        console,
        title="Recent run failures",
        rows=payload["recent_failures"],
        columns=[
            ("started_at", "started_at"),
            ("job", "job_name"),
            ("error_type", "error_type"),
            ("error_message", "error_message"),
            ("run_id", "run_id"),
        ],
        empty_text="No recent failed runs.",
    )
    _render_table(
        console,
        title="Watermarks and freshness",
        rows=payload["watermarks"],
        columns=[
            ("asset_key", "asset_key"),
            ("scope_key", "scope_key"),
            ("watermark_ts", "watermark_ts"),
            ("staleness_days", "staleness_days"),
            ("last_run_id", "last_run_id"),
        ],
        empty_text="No watermark records found.",
    )
    _render_table(
        console,
        title="Recent failed checks",
        rows=payload["recent_failed_checks"],
        columns=[
            ("checked_at", "checked_at"),
            ("asset_key", "asset_key"),
            ("partition_key", "partition_key"),
            ("check_name", "check_name"),
            ("severity", "severity"),
            ("message", "message"),
        ],
        empty_text="No recent failed checks.",
    )


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


@app.command("health")
def db_health(
    days: int = typer.Option(
        7,
        "--days",
        min=0,
        help="Lookback window for latest-runs/failures/failed-checks.",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        min=1,
        help="Max rows per section.",
    ),
    stale_days: int | None = typer.Option(
        None,
        "--stale-days",
        min=0,
        help="Only include watermarks at or above this staleness (days).",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON.",
    ),
    duckdb_path: Path | None = typer.Option(
        None,
        "--duckdb-path",
        help="DuckDB file path (defaults to data/warehouse/options.duckdb).",
    ),
) -> None:
    """Show observability health summary (not financial advice)."""
    console = Console(width=200)
    path = _resolve_duckdb_path(duckdb_path)
    try:
        payload = _build_health_payload(
            path=path,
            days=max(int(days), 0),
            limit=max(int(limit), 1),
            stale_days=None if stale_days is None else max(int(stale_days), 0),
        )
        if as_json:
            typer.echo(json.dumps(payload, indent=2, default=_json_default))
            return
        _render_health(console, payload)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
