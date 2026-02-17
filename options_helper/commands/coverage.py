from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from options_helper.data.coverage_service import build_symbol_coverage
from options_helper.data.storage_runtime import get_default_duckdb_path


def register(app: typer.Typer) -> None:
    app.command("coverage")(coverage)


def coverage(
    symbol: str = typer.Argument(..., help="Underlying symbol (e.g. SPY)."),
    days: int = typer.Option(
        60,
        "--days",
        min=1,
        help="Business-day lookback window for coverage diagnostics.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional JSON output path.",
    ),
    duckdb_path: Path | None = typer.Option(
        None,
        "--duckdb-path",
        help="DuckDB file path (defaults to data/warehouse/options.duckdb).",
    ),
) -> None:
    """Show first-class local data coverage for one symbol (not financial advice)."""
    console = Console(width=200)
    sym = _normalize_symbol(symbol)

    try:
        payload = build_symbol_coverage(
            sym,
            days=max(1, int(days)),
            duckdb_path=duckdb_path,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=_json_default))
        return

    _render_payload(console, payload)
    if out is not None:
        console.print(f"Saved JSON: {out}")


def _render_payload(console: Console, payload: dict[str, Any]) -> None:
    symbol = str(payload.get("symbol") or "")
    lookback = int(payload.get("days") or 0)
    db_path = str(payload.get("database_path") or _resolve_duckdb_path(None))
    as_of = payload.get("as_of_date") or "-"
    _render_payload_header(console, symbol=symbol, db_path=db_path, lookback=lookback, as_of=as_of)
    for note in payload.get("notes") or []:
        console.print(f"[yellow]Note:[/yellow] {note}")
    candles = dict(payload.get("candles") or {})
    snapshots = dict(payload.get("snapshots") or {})
    contracts_oi = dict(payload.get("contracts_oi") or {})
    option_bars = dict(payload.get("option_bars") or {})
    _render_candles_section(console, candles=candles, lookback=lookback)
    _render_snapshot_section(console, snapshots=snapshots, lookback=lookback)
    _render_contracts_oi_section(console, contracts_oi=contracts_oi)
    _render_oi_delta_table(console, delta_rows=contracts_oi.get("oi_delta_coverage") or [])
    _render_option_bars_section(console, option_bars=option_bars)
    _render_option_bars_status_table(console, status_counts=option_bars.get("status_counts") or {})
    _render_repair_suggestions(console, suggestions=payload.get("repair_suggestions") or [])


def _render_payload_header(console: Console, *, symbol: str, db_path: str, lookback: int, as_of: Any) -> None:
    console.print(f"[bold]Coverage for {symbol}[/bold]")
    console.print("Informational and educational use only. Not financial advice.")
    console.print(f"DuckDB: {db_path}")
    console.print(f"Lookback: {lookback} business day(s), as_of={as_of}")


def _render_candles_section(console: Console, *, candles: dict[str, Any], lookback: int) -> None:
    _render_section(
        console,
        title="Candles",
        rows=[
            ("Rows", _fmt_int(candles.get("rows_total"))),
            ("Range", _fmt_range(candles.get("start_date"), candles.get("end_date"))),
            (f"Missing business days (last {lookback})", _fmt_int(candles.get("missing_business_days"))),
            ("Missing value cells", _fmt_int(candles.get("missing_value_cells"))),
        ],
    )


def _render_snapshot_section(console: Console, *, snapshots: dict[str, Any], lookback: int) -> None:
    _render_section(
        console,
        title="Options Snapshot Headers",
        rows=[
            ("Days present", _fmt_int(snapshots.get("days_present_total"))),
            ("Range", _fmt_range(snapshots.get("start_date"), snapshots.get("end_date"))),
            (f"Missing business days (last {lookback})", _fmt_int(snapshots.get("missing_business_days"))),
            ("Avg contracts/day", _fmt_float(snapshots.get("avg_contracts_per_day"))),
            ("Non-zero contract days", _fmt_int(snapshots.get("non_zero_contract_days"))),
        ],
    )


def _render_contracts_oi_section(console: Console, *, contracts_oi: dict[str, Any]) -> None:
    _render_section(
        console,
        title="Contract + OI Snapshots",
        rows=[
            ("Contracts", _fmt_int(contracts_oi.get("contracts_total"))),
            ("Contracts with snapshots", _fmt_int(contracts_oi.get("contracts_with_snapshots"))),
            ("Contracts with OI", _fmt_int(contracts_oi.get("contracts_with_oi"))),
            ("Snapshot day coverage", _fmt_pct(contracts_oi.get("snapshot_day_coverage_ratio"))),
            ("OI day coverage", _fmt_pct(contracts_oi.get("oi_day_coverage_ratio"))),
            ("Missing snapshot days", _fmt_int(contracts_oi.get("snapshot_days_missing"))),
        ],
    )


def _render_oi_delta_table(console: Console, *, delta_rows: list[dict[str, Any]]) -> None:
    if not delta_rows:
        return
    table = Table(title="OI Delta Coverage", show_header=True)
    table.add_column("Lag")
    table.add_column("Contracts w/ OI")
    table.add_column("Contracts w/ Delta")
    table.add_column("Pairs")
    table.add_column("Coverage")
    for row in delta_rows:
        lag = int((row or {}).get("lag_days") or 0)
        table.add_row(
            f"{lag}d",
            _fmt_int((row or {}).get("contracts_with_oi")),
            _fmt_int((row or {}).get("contracts_with_delta")),
            _fmt_int((row or {}).get("pair_count")),
            _fmt_pct((row or {}).get("coverage_ratio")),
        )
    console.print(table)


def _render_option_bars_section(console: Console, *, option_bars: dict[str, Any]) -> None:
    _render_section(
        console,
        title="Option Bars Meta",
        rows=[
            ("Contracts", _fmt_int(option_bars.get("contracts_total"))),
            ("Contracts with rows", _fmt_int(option_bars.get("contracts_with_rows"))),
            ("Rows total", _fmt_int(option_bars.get("rows_total"))),
            ("Contracts covering lookback end", _fmt_int(option_bars.get("contracts_covering_lookback_end"))),
            ("Coverage at lookback end", _fmt_pct(option_bars.get("covering_lookback_end_ratio"))),
            ("Range", _fmt_range(option_bars.get("start_date"), option_bars.get("end_date"))),
        ],
    )


def _render_option_bars_status_table(console: Console, *, status_counts: dict[str, Any]) -> None:
    if not status_counts:
        return
    table = Table(title="Option Bars Status Counts", show_header=True)
    table.add_column("Status")
    table.add_column("Count")
    for key in sorted(status_counts.keys()):
        table.add_row(str(key), _fmt_int(status_counts.get(key)))
    console.print(table)


def _render_repair_suggestions(console: Console, *, suggestions: list[dict[str, Any]]) -> None:
    console.print("\n[bold]Repair Suggestions[/bold]")
    if not suggestions:
        console.print("No repair commands suggested.")
        return
    for item in suggestions:
        priority = _fmt_int((item or {}).get("priority"))
        reason = str((item or {}).get("reason") or "")
        command = str((item or {}).get("command") or "")
        note = str((item or {}).get("note") or "").strip()
        console.print(f"[{priority}] {reason}")
        console.print(f"  {command}")
        if note:
            console.print(f"  note: {note}")


def _render_section(console: Console, *, title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for metric, value in rows:
        table.add_row(metric, value)
    console.print(table)


def _normalize_symbol(value: str) -> str:
    raw = str(value or "").strip().upper()
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})
    if not cleaned:
        raise typer.BadParameter("symbol is required", param_hint="symbol")
    return cleaned


def _resolve_duckdb_path(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    return get_default_duckdb_path()


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,d}"
    except Exception:  # noqa: BLE001
        return "0"


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
        if pd_is_na(number):
            return "-"
        return f"{number:,.{digits}f}"
    except Exception:  # noqa: BLE001
        return "-"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
        if pd_is_na(number):
            return "-"
        return f"{number * 100.0:.{digits}f}%"
    except Exception:  # noqa: BLE001
        return "-"


def _fmt_range(start: Any, end: Any) -> str:
    start_txt = _fmt_date(start)
    end_txt = _fmt_date(end)
    if start_txt == "-" and end_txt == "-":
        return "-"
    return f"{start_txt} -> {end_txt}"


def _fmt_date(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return "-"
    return text[:10]


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def pd_is_na(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:  # noqa: BLE001
        return False
