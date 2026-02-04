from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console

from options_helper.analysis.osi import normalize_underlying
from options_helper.cli_deps import build_provider
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.corporate_actions_store import CorporateActionsStore
from options_helper.data.market_types import DataFetchError
from options_helper.data.news_store import NewsStore

app = typer.Typer(help="Corporate actions + news ingestion (not financial advice).")


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


def _parse_symbols(symbols: list[str] | None, symbols_csv: str | None) -> list[str]:
    tokens: list[str] = []
    for item in symbols or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if symbols_csv:
        tokens.extend([part for part in symbols_csv.split(",") if part.strip()])
    normalized = [normalize_underlying(tok) for tok in tokens if tok]
    return sorted({tok for tok in normalized if tok})


def _parse_types(types: list[str] | None, types_csv: str | None) -> list[str]:
    tokens: list[str] = []
    for item in types or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if types_csv:
        tokens.extend([part for part in types_csv.split(",") if part.strip()])
    return sorted({tok.strip().lower() for tok in tokens if tok})


def _require_alpaca_provider() -> None:
    provider = build_provider()
    name = getattr(provider, "name", None)
    if name != "alpaca":
        raise typer.BadParameter("Events ingestion currently requires --provider alpaca.")


@app.command("refresh-corporate-actions")
def refresh_corporate_actions(
    symbol: list[str] | None = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Symbol to refresh (repeatable or comma-separated).",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated symbols to refresh.",
    ),
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)."),
    action_type: list[str] | None = typer.Option(
        None,
        "--type",
        help="Corporate action type filter (repeatable or comma-separated).",
    ),
    types: str | None = typer.Option(
        None,
        "--types",
        help="Comma-separated corporate action type filters.",
    ),
    actions_dir: Path = typer.Option(
        Path("data/events/corporate_actions"),
        "--actions-dir",
        help="Directory for cached corporate actions.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Page size limit per request (provider-specific).",
    ),
    page_limit: int | None = typer.Option(
        None,
        "--page-limit",
        help="Maximum pages to request (safety guard).",
    ),
) -> None:
    """Refresh corporate actions for symbols."""
    _require_alpaca_provider()
    console = Console()
    symbols_list = _parse_symbols(symbol, symbols)
    if not symbols_list:
        raise typer.BadParameter("Provide at least one --symbol or --symbols.")

    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt < start_dt:
        raise typer.BadParameter("End date is before start date.")

    types_list = _parse_types(action_type, types)
    client = AlpacaClient()
    store = CorporateActionsStore(actions_dir)

    try:
        actions = client.get_corporate_actions(
            symbols_list,
            start=start_dt,
            end=end_dt,
            types=types_list or None,
            limit=limit,
            page_limit=page_limit,
        )
    except DataFetchError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    by_symbol: dict[str, list[dict]] = {sym: [] for sym in symbols_list}
    for action in actions:
        sym = str(action.get("symbol") or "").upper()
        if not sym and len(symbols_list) == 1:
            sym = symbols_list[0]
        if sym in by_symbol:
            by_symbol[sym].append(action)

    for sym in symbols_list:
        meta = {
            "provider": "alpaca",
            "provider_version": client.provider_version,
            "request": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "types": types_list,
                "limit": limit,
                "page_limit": page_limit,
            },
        }
        path = store.save(sym, by_symbol.get(sym, []), meta=meta, merge=True)
        console.print(f"[green]Saved[/green] {sym} -> {path}")


@app.command("refresh-news")
def refresh_news(
    symbol: list[str] | None = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Symbol to refresh (repeatable or comma-separated).",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated symbols to refresh.",
    ),
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD)."),
    include_content: bool = typer.Option(
        False,
        "--include-content",
        help="Include full content/body when available (larger files).",
    ),
    news_dir: Path = typer.Option(
        Path("data/events/news"),
        "--news-dir",
        help="Directory for cached news.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Page size limit per request (provider-specific).",
    ),
    page_limit: int | None = typer.Option(
        None,
        "--page-limit",
        help="Maximum pages to request (safety guard).",
    ),
) -> None:
    """Refresh news for symbols."""
    _require_alpaca_provider()
    console = Console()
    symbols_list = _parse_symbols(symbol, symbols)
    if not symbols_list:
        raise typer.BadParameter("Provide at least one --symbol or --symbols.")

    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt < start_dt:
        raise typer.BadParameter("End date is before start date.")

    client = AlpacaClient()
    store = NewsStore(news_dir)

    try:
        items = client.get_news(
            symbols_list,
            start=start_dt,
            end=end_dt,
            include_content=include_content,
            limit=limit,
            page_limit=page_limit,
        )
    except DataFetchError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    target_set = set(symbols_list)
    by_symbol: dict[str, list[dict]] = {sym: [] for sym in symbols_list}
    for item in items:
        for sym in item.get("symbols", []) or []:
            if sym in target_set:
                by_symbol.setdefault(sym, []).append(item)

    for sym in symbols_list:
        meta = {
            "provider": "alpaca",
            "provider_version": client.provider_version,
            "request": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "include_content": include_content,
                "limit": limit,
                "page_limit": page_limit,
            },
        }
        paths = store.upsert_items(sym, by_symbol.get(sym, []), meta=meta)
        if not paths:
            console.print(f"[yellow]No news saved for {sym}.[/yellow]")
            continue
        console.print(f"[green]Saved[/green] {sym} -> {len(paths)} day(s)")
