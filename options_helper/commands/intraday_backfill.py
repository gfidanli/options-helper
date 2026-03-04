from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.pipelines.intraday_backfill_runtime import run_backfill_stocks_history


app = typer.Typer(help="Historical intraday stock backfill utilities (not financial advice).")


def _require_alpaca_provider() -> None:
    provider = cli_deps.build_provider()
    name = getattr(provider, "name", None)
    if name != "alpaca":
        raise typer.BadParameter("This command currently requires --provider alpaca.")


_BACKFILL_EXCLUDE_PATH_OPT = typer.Option(
    Path("data/universe/exclude_symbols.txt"),
    "--exclude-path",
    help="Path to exclude symbols file (one ticker per line).",
)
_BACKFILL_OUT_DIR_OPT = typer.Option(
    Path("data/intraday"),
    "--out-dir",
    help="Base directory for intraday partitions.",
)
_BACKFILL_STATUS_DIR_OPT = typer.Option(
    Path("data/intraday_backfill_status"),
    "--status-dir",
    help="Directory for run status + checkpoint reports.",
)
_BACKFILL_RUN_ID_OPT = typer.Option(
    None,
    "--run-id",
    help="Optional run id for status folder naming (default: UTC timestamp).",
)
_BACKFILL_START_DATE_OPT = typer.Option(
    "2000-01-01",
    "--start-date",
    help="Historical start date anchor (YYYY-MM-DD).",
)
_BACKFILL_END_DATE_OPT = typer.Option(
    None,
    "--end-date",
    help="Optional explicit end date (YYYY-MM-DD). Defaults to last completed market day.",
)
_BACKFILL_MAX_SYMBOLS_OPT = typer.Option(
    None,
    "--max-symbols",
    min=1,
    help="Optional cap on symbols for staged runs.",
)
_BACKFILL_FEED_OPT = typer.Option(
    "sip",
    "--feed",
    help="Alpaca stock feed for historical bars (default: sip).",
)
_BACKFILL_CHECKPOINT_SYMBOLS_OPT = typer.Option(
    25,
    "--checkpoint-symbols",
    min=0,
    help="Emit checkpoint analysis after this many processed symbols (0 disables).",
)
_BACKFILL_PAUSE_OPT = typer.Option(
    True,
    "--pause-at-checkpoint/--no-pause-at-checkpoint",
    help="Pause after checkpoint so performance/bottlenecks can be reviewed.",
)


@app.command("stocks-history")
def stocks_history(
    exclude_path: Path = _BACKFILL_EXCLUDE_PATH_OPT,
    out_dir: Path = _BACKFILL_OUT_DIR_OPT,
    status_dir: Path = _BACKFILL_STATUS_DIR_OPT,
    run_id: str | None = _BACKFILL_RUN_ID_OPT,
    start_date: str = _BACKFILL_START_DATE_OPT,
    end_date: str | None = _BACKFILL_END_DATE_OPT,
    max_symbols: int | None = _BACKFILL_MAX_SYMBOLS_OPT,
    feed: str = _BACKFILL_FEED_OPT,
    checkpoint_symbols: int = _BACKFILL_CHECKPOINT_SYMBOLS_OPT,
    pause_at_checkpoint: bool = _BACKFILL_PAUSE_OPT,
) -> None:
    """Backfill full historical 1-minute bars by symbol and year-month."""
    _require_alpaca_provider()
    run_backfill_stocks_history(
        console=Console(width=200),
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
