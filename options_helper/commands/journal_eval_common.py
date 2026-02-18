from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import typer
from rich.console import Console


_JOURNAL_EVALUATE_JOURNAL_DIR_OPT = typer.Option(
    Path("data/journal"),
    "--journal-dir",
    help="Directory containing signal_events.jsonl.",
)
_JOURNAL_EVALUATE_CACHE_DIR_OPT = typer.Option(
    Path("data/candles"),
    "--cache-dir",
    help="Directory for cached daily candles (used for outcomes).",
)
_JOURNAL_EVALUATE_SNAPSHOTS_DIR_OPT = typer.Option(
    Path("data/options_snapshots"),
    "--snapshots-dir",
    help="Directory for options snapshot history (used for option mark outcomes).",
)
_JOURNAL_EVALUATE_HORIZONS_OPT = typer.Option(
    "1,5,20",
    "--horizons",
    help="Comma-separated trading-day horizons (e.g. 1,5,20).",
)
_JOURNAL_EVALUATE_WINDOW_OPT = typer.Option(
    252,
    "--window",
    min=1,
    max=5000,
    help="Lookback window in calendar days.",
)
_JOURNAL_EVALUATE_AS_OF_OPT = typer.Option(
    None,
    "--as-of",
    help="As-of date (YYYY-MM-DD). Defaults to today.",
)
_JOURNAL_EVALUATE_OUT_DIR_OPT = typer.Option(
    Path("data/reports/journal"),
    "--out-dir",
    help="Output directory for journal evaluation reports.",
)
_JOURNAL_EVALUATE_TOP_OPT = typer.Option(
    5,
    "--top",
    min=1,
    max=50,
    help="Top/bottom events to include in summaries.",
)


def load_filtered_events(
    *,
    store: Any,
    console: Console,
    as_of: str | None,
    window: int,
    parse_date: Callable[[str], date],
) -> tuple[list[Any], date]:
    result = store.read_events()
    if result.errors:
        console.print(f"[yellow]Warning:[/yellow] skipped {len(result.errors)} invalid journal lines.")

    events = result.events
    if not events:
        console.print("No journal events found.")
        raise typer.Exit(0)

    as_of_date = parse_date(as_of) if as_of else date.today()
    start_date = as_of_date - timedelta(days=int(window))
    filtered = [event for event in events if start_date <= event.date <= as_of_date]
    if not filtered:
        console.print("No journal events within the window.")
        raise typer.Exit(0)
    return filtered, as_of_date


def parse_horizon_values(horizons: str) -> list[int]:
    values: list[int] = []
    for part in horizons.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid horizon value: {part}") from exc
        if value <= 0:
            raise typer.BadParameter("Horizons must be positive integers.")
        values.append(value)
    if not values:
        raise typer.BadParameter("Provide at least one horizon.")
    return values


def load_history_by_symbol(
    *,
    symbols: set[str],
    candle_store: Any,
    console: Console,
) -> dict[str, pd.DataFrame]:
    history_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            history_by_symbol[symbol] = candle_store.load(symbol)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {symbol}: {exc}")
            history_by_symbol[symbol] = pd.DataFrame()
    return history_by_symbol


def make_snapshot_loader(snapshot_store: Any) -> Callable[[str, date], pd.DataFrame | None]:
    snapshot_cache: dict[tuple[str, date], pd.DataFrame] = {}

    def _snapshot_loader(symbol: str, snapshot_date: date) -> pd.DataFrame | None:
        key = (symbol.upper(), snapshot_date)
        if key in snapshot_cache:
            return snapshot_cache[key]
        try:
            frame = snapshot_store.load_day(symbol, snapshot_date)
        except Exception:  # noqa: BLE001
            frame = pd.DataFrame()
        snapshot_cache[key] = frame
        return frame

    return _snapshot_loader


def persist_report(
    *,
    report: dict[str, Any],
    as_of_date: date,
    window: int,
    out_dir: Path,
    console: Console,
    render_markdown: Callable[[dict[str, Any]], str],
) -> None:
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["as_of"] = as_of_date.isoformat()
    report["window_days"] = int(window)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{as_of_date.isoformat()}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = out_dir / f"{as_of_date.isoformat()}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    console.print(f"Saved: {json_path}")
    console.print(f"Saved: {md_path}")
