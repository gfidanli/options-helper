from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from options_helper.data.alpaca_client import AlpacaClient, _load_market_tz
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.market_types import DataFetchError

from options_helper.pipelines.intraday_backfill_reporting import (
    BackfillPaths,
    append_jsonl,
    apply_symbol_summary,
    build_paths,
    ensure_run_dirs,
    initial_totals,
    record_failure,
    write_checkpoint_reports,
    write_current_symbol_status,
    write_overall_status,
    write_run_config,
    write_symbol_status,
)
from options_helper.pipelines.intraday_backfill_support import (
    iter_month_ranges,
    load_market_days,
    load_target_symbols,
    parse_date,
    process_symbol_month,
    read_exclude_symbols,
    resolve_end_day,
    resolve_first_day,
)


def run_backfill_stocks_history(
    *,
    console: Console,
    exclude_path: Path,
    out_dir: Path,
    status_dir: Path,
    run_id: str | None,
    start_date: str,
    end_date: str | None,
    max_symbols: int | None,
    feed: str,
    checkpoint_symbols: int,
    pause_at_checkpoint: bool,
) -> None:
    context = prepare_backfill_context(
        exclude_path=exclude_path,
        out_dir=out_dir,
        status_dir=status_dir,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        max_symbols=max_symbols,
        feed=feed,
        checkpoint_symbols=checkpoint_symbols,
        pause_at_checkpoint=pause_at_checkpoint,
    )
    totals = initial_totals(symbols_total=len(context["symbols"]))
    checkpoint_records: list[dict[str, Any]] = []
    symbol_summaries: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for symbol in context["symbols"]:
        records, summary = process_symbol(
            console=console,
            client=context["client"],
            store=context["store"],
            symbol=symbol,
            start_day=context["start_day"],
            end_day=context["end_day"],
            market_tz=context["market_tz"],
            market_days=context["market_days"],
            paths=context["paths"],
        )
        symbol_summaries.append(summary)
        apply_symbol_summary(totals=totals, summary=summary, month_records=records)
        totals["elapsed_seconds"] = round(time.perf_counter() - run_started, 3)
        write_overall_status(paths=context["paths"], totals=totals)

        if totals["symbols_processed"] <= checkpoint_symbols:
            checkpoint_records.extend(records)
        if checkpoint_symbols > 0 and totals["symbols_processed"] == checkpoint_symbols:
            write_checkpoint_reports(
                paths=context["paths"],
                symbol_summaries=symbol_summaries,
                month_records=checkpoint_records,
                totals=totals,
            )
            console.print(f"[cyan]Checkpoint report:[/cyan] {context['paths'].checkpoint_md}")
            if pause_at_checkpoint:
                console.print(
                    "[yellow]Paused after checkpoint.[/yellow] "
                    "Review performance/bottlenecks before continuing."
                )
                raise typer.Exit(0)

    totals["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_overall_status(paths=context["paths"], totals=totals)
    console.print(
        "[green]Backfill complete.[/green] "
        f"symbols={totals['symbols_processed']}/{totals['symbols_total']} "
        f"months={totals['months_processed']} errors={totals['months_error']}"
    )


def prepare_backfill_context(
    *,
    exclude_path: Path,
    out_dir: Path,
    status_dir: Path,
    run_id: str | None,
    start_date: str,
    end_date: str | None,
    max_symbols: int | None,
    feed: str,
    checkpoint_symbols: int,
    pause_at_checkpoint: bool,
) -> dict[str, Any]:
    start_day = parse_date(start_date)
    run_label = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = build_paths(status_dir=status_dir, run_id=run_label)
    ensure_run_dirs(paths)

    client = AlpacaClient(stock_feed=feed)
    store = IntradayStore(out_dir)
    market_tz = _load_market_tz()
    end_day = resolve_end_day(client=client, market_tz=market_tz, override=end_date)
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on/after --start-date.")
    excludes = read_exclude_symbols(exclude_path)
    symbols = load_target_symbols(client=client, excludes=excludes, max_symbols=max_symbols)
    market_days = load_market_days(client=client, start_day=start_day, end_day=end_day)

    write_run_config(
        paths=paths,
        payload=build_run_config_payload(
            run_label=run_label,
            start_day=start_day,
            end_day=end_day,
            out_dir=out_dir,
            exclude_path=exclude_path,
            feed=feed,
            checkpoint_symbols=checkpoint_symbols,
            pause_at_checkpoint=pause_at_checkpoint,
            symbols_total=len(symbols),
        ),
        symbols=symbols,
    )
    return {
        "paths": paths,
        "client": client,
        "store": store,
        "market_tz": market_tz,
        "start_day": start_day,
        "end_day": end_day,
        "symbols": symbols,
        "market_days": market_days,
    }


def process_symbol(
    *,
    console: Console,
    client: AlpacaClient,
    store: IntradayStore,
    symbol: str,
    start_day,
    end_day,
    market_tz,
    market_days,
    paths: BackfillPaths,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_days = set(store.list_days("stocks", "bars", "1Min", symbol))
    summary = {
        "symbol": symbol,
        "status": "ok",
        "first_data_day": None,
        "processed_months": 0,
        "errors": 0,
    }
    records: list[dict[str, Any]] = []

    try:
        daily = client.get_stock_bars(symbol, start=start_day, end=end_day, interval="1d", adjustment="raw")
    except DataFetchError as exc:
        summary["status"] = "error"
        summary["errors"] = 1
        record_failure(paths=paths, symbol=symbol, year_month="daily-probe", error=exc)
        write_symbol_status(paths=paths, symbol=symbol, summary=summary, records=records)
        return records, summary

    first_day = resolve_first_day(daily, market_tz=market_tz)
    if first_day is None:
        summary["status"] = "no_data"
        write_symbol_status(paths=paths, symbol=symbol, summary=summary, records=records)
        return records, summary

    if first_day < start_day:
        first_day = start_day
    summary["first_data_day"] = first_day.isoformat()

    for year_month, month_start, month_end in iter_month_ranges(first_day, end_day):
        rec = process_symbol_month(
            client=client,
            store=store,
            symbol=symbol,
            year_month=year_month,
            month_start=month_start,
            month_end=month_end,
            existing_days=existing_days,
            market_days=market_days,
            market_tz=market_tz,
        )
        records.append(rec)
        summary["processed_months"] += 1
        if rec["status"] == "error":
            summary["errors"] += 1
            record_failure(
                paths=paths,
                symbol=symbol,
                year_month=year_month,
                error=RuntimeError(str(rec.get("error_message") or "unknown")),
                error_type=rec.get("error_type"),
            )
        append_jsonl(paths.results_jsonl, rec)
        write_symbol_status(paths=paths, symbol=symbol, summary=summary, records=records)
        write_current_symbol_status(paths=paths, symbol=symbol, year_month=year_month, rec=rec)
        console.print(
            f"{symbol} {year_month} status={rec['status']} rows={rec['rows']} "
            f"fetch_s={rec['fetch_seconds']:.2f} write_s={rec['write_seconds']:.2f}"
        )

    return records, summary


def build_run_config_payload(
    *,
    run_label: str,
    start_day,
    end_day,
    out_dir: Path,
    exclude_path: Path,
    feed: str,
    checkpoint_symbols: int,
    pause_at_checkpoint: bool,
    symbols_total: int,
) -> dict[str, Any]:
    return {
        "run_id": run_label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "out_dir": str(out_dir),
        "exclude_path": str(exclude_path),
        "symbols_total": symbols_total,
        "feed": feed,
        "checkpoint_symbols": checkpoint_symbols,
        "pause_at_checkpoint": pause_at_checkpoint,
    }
