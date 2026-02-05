from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobExecutionError,
    run_derived_update_job,
)
from options_helper.data.derived import DERIVED_COLUMNS, DERIVED_SCHEMA_VERSION

app = typer.Typer(help="Persist derived metrics from local snapshots.")

JOB_COMPUTE_DERIVED = "compute_derived"
ASSET_DERIVED_DAILY = "derived_daily"
NOOP_LEDGER_WARNING = (
    "Run ledger disabled for filesystem storage backend (NoopRunLogger active)."
)

pd: object | None = None


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def _spot_from_meta(meta: dict) -> float | None:
    if not meta:
        return None
    candidates = [
        meta.get("spot"),
        (meta.get("underlying") or {}).get("regularMarketPrice"),
        (meta.get("underlying") or {}).get("regularMarketPreviousClose"),
        (meta.get("underlying") or {}).get("regularMarketOpen"),
    ]
    for v in candidates:
        try:
            if v is None:
                continue
            spot = float(v)
            if spot > 0:
                return spot
        except Exception:  # noqa: BLE001
            continue
    return None


def _is_noop_run_logger(run_logger: object) -> bool:
    return run_logger.__class__.__name__ == "NoopRunLogger"


def _coerce_iso_date(value: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def _observed_run(*, console: Console, job_name: str, args: dict[str, Any]):
    run_logger = cli_deps.build_run_logger(
        job_name=job_name,
        provider=get_default_provider_name(),
        args=args,
    )
    if _is_noop_run_logger(run_logger):
        console.print(f"[yellow]Warning:[/yellow] {NOOP_LEDGER_WARNING}")
    try:
        yield run_logger
    except typer.Exit as exc:
        exit_code = int(getattr(exc, "exit_code", 1) or 0)
        if exit_code == 0:
            run_logger.finalize_success()
        else:
            run_logger.finalize_failure(exc.__cause__ if exc.__cause__ is not None else exc)
        raise
    except Exception as exc:  # noqa: BLE001
        run_logger.finalize_failure(exc)
        raise
    else:
        run_logger.finalize_success()


@app.command("update")
def derived_update(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to update."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (writes {derived_dir}/{SYMBOL}.csv).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for realized volatility).",
    ),
) -> None:
    """Append or upsert a derived-metrics row for a symbol/day (offline)."""
    console = Console(width=200)
    with _observed_run(
        console=console,
        job_name=JOB_COMPUTE_DERIVED,
        args={
            "symbol": symbol,
            "as_of": as_of,
            "cache_dir": str(cache_dir),
            "derived_dir": str(derived_dir),
            "candle_cache_dir": str(candle_cache_dir),
        },
    ) as run_logger:
        try:
            result = run_derived_update_job(
                symbol=symbol,
                as_of=as_of,
                cache_dir=cache_dir,
                derived_dir=derived_dir,
                candle_cache_dir=candle_cache_dir,
                snapshot_store_builder=cli_deps.build_snapshot_store,
                derived_store_builder=cli_deps.build_derived_store,
                candle_store_builder=cli_deps.build_candle_store,
            )
            derived_day = _coerce_iso_date(result.as_of_date)
            bytes_written = result.output_path.stat().st_size if result.output_path.exists() else None
            run_logger.log_asset_success(
                asset_key=ASSET_DERIVED_DAILY,
                asset_kind="table",
                partition_key=result.symbol.upper(),
                rows_inserted=1,
                bytes_written=bytes_written,
                min_event_ts=derived_day,
                max_event_ts=derived_day,
            )
            if derived_day is not None:
                run_logger.upsert_watermark(
                    asset_key=ASSET_DERIVED_DAILY,
                    scope_key=result.symbol.upper(),
                    watermark_ts=derived_day,
                )
            console.print(f"Derived schema v{DERIVED_SCHEMA_VERSION} updated: {result.output_path}")
        except VisibilityJobExecutionError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc


@app.command("show")
def derived_show(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to show."),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
    ),
    last: int = typer.Option(30, "--last", min=1, max=3650, help="Show the last N rows."),
) -> None:
    """Print the last N rows of derived metrics for a symbol."""
    from rich.table import Table

    _ensure_pandas()
    console = Console(width=200)
    derived = cli_deps.build_derived_store(derived_dir)

    try:
        df = derived.load(symbol)
        if df.empty:
            console.print(f"No derived rows found for {symbol.upper()} in {derived_dir}")
            raise typer.Exit(1)

        tail = df.tail(last)
        t = Table(title=f"{symbol.upper()} derived metrics (last {min(last, len(df))})")
        for col in tail.columns:
            t.add_column(col)
        for _, row in tail.iterrows():
            t.add_row(*["" if pd.isna(v) else str(v) for v in row.tolist()])
        console.print(t)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("stats")
def derived_stats(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to analyze."),
    as_of: str = typer.Option("latest", "--as-of", help="Derived date (YYYY-MM-DD) or 'latest'."),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
    ),
    window: int = typer.Option(60, "--window", min=1, max=3650, help="Lookback window for percentiles."),
    trend_window: int = typer.Option(5, "--trend-window", min=1, max=3650, help="Lookback window for trend flags."),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/derived/{SYMBOL}/).",
    ),
) -> None:
    """Percentile ranks and trend flags from the derived-metrics history (offline)."""
    from rich.table import Table

    console = Console(width=200)
    derived = cli_deps.build_derived_store(derived_dir)

    try:
        df = derived.load(symbol)
        if df.empty:
            console.print(f"No derived rows found for {symbol.upper()} in {derived_dir}")
            raise typer.Exit(1)

        fmt = format.strip().lower()
        if fmt not in {"console", "json"}:
            raise typer.BadParameter("Invalid --format (use console|json)", param_hint="--format")

        report = compute_derived_stats(
            df,
            symbol=symbol,
            as_of=as_of,
            window=window,
            trend_window=trend_window,
            metric_columns=[c for c in DERIVED_COLUMNS if c != "date"],
        )

        if fmt == "json":
            console.print(report.model_dump_json(indent=2))
        else:
            t = Table(
                title=f"{report.symbol} derived stats (as-of {report.as_of}; pct w={window}; trend w={trend_window})"
            )
            t.add_column("metric")
            t.add_column("value", justify="right")
            t.add_column(f"pct({window})", justify="right")
            t.add_column(f"trend({trend_window})", justify="right")
            t.add_column("Δ", justify="right")
            t.add_column("Δ%", justify="right")

            for m in report.metrics:
                value = "" if m.value is None else f"{m.value:.8g}"
                pct = "" if m.percentile is None else f"{m.percentile:.1f}"
                delta = "" if m.trend_delta is None else f"{m.trend_delta:.8g}"
                delta_pct = "" if m.trend_delta_pct is None else f"{m.trend_delta_pct:.2f}"
                trend = "" if m.trend_direction is None else m.trend_direction
                t.add_row(m.name, value, pct, trend, delta, delta_pct)

            console.print(t)
            if report.warnings:
                console.print(f"[yellow]Warnings:[/yellow] {', '.join(report.warnings)}")

        if out is not None:
            base = out / "derived" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{report.as_of}_w{window}_tw{trend_window}.json"
            out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
