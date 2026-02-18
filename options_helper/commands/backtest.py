from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.models import OptionType
from options_helper.schemas.common import utc_now

app = typer.Typer(help="Offline options backtesting.")

_SYMBOL_OPT = typer.Option(..., "--symbol", help="Underlying symbol to backtest.")
_CONTRACT_SYMBOL_OPT = typer.Option(
    None,
    "--contract-symbol",
    help="Full contractSymbol to trade (uses snapshot data).",
)
_EXPIRY_OPT = typer.Option(
    None,
    "--expiry",
    help="Contract expiry (YYYY-MM-DD) when contract-symbol is not provided.",
)
_STRIKE_OPT = typer.Option(None, "--strike", help="Contract strike when contract-symbol is not provided.")
_OPTION_TYPE_OPT = typer.Option("call", "--option-type", help="Option type (call/put).")
_START_OPT = typer.Option(None, "--start", help="Start snapshot date (YYYY-MM-DD).")
_END_OPT = typer.Option(None, "--end", help="End snapshot date (YYYY-MM-DD).")
_FILL_MODE_OPT = typer.Option(
    "worst_case",
    "--fill-mode",
    help="Fill model: worst_case (bid/ask) or mark_slippage.",
)
_SLIPPAGE_FACTOR_OPT = typer.Option(0.5, "--slippage-factor", help="Slippage factor for mark_slippage.")
_INITIAL_CASH_OPT = typer.Option(10000.0, "--initial-cash", help="Starting cash balance.")
_QUANTITY_OPT = typer.Option(1, "--quantity", help="Contracts per trade.")
_EXTENSION_LOW_PCT_OPT = typer.Option(20.0, "--extension-low-pct", help="Entry extension percentile threshold.")
_TAKE_PROFIT_PCT_OPT = typer.Option(0.8, "--take-profit-pct", help="Take-profit threshold (pct).")
_STOP_LOSS_PCT_OPT = typer.Option(0.5, "--stop-loss-pct", help="Stop-loss threshold (pct).")
_MAX_HOLDING_DAYS_OPT = typer.Option(15, "--max-holding-days", help="Time stop (days).")
_ROLL_DTE_THRESHOLD_OPT = typer.Option(
    None,
    "--roll-dte-threshold",
    help="Enable rolling when DTE <= threshold (omit to disable rolling).",
)
_ROLL_HORIZON_MONTHS_OPT = typer.Option(2, "--roll-horizon-months", help="Roll target horizon (months).")
_ROLL_SHAPE_OPT = typer.Option(
    "out-same-strike",
    "--roll-shape",
    help="Roll shape: out-same-strike | out-up | out-down.",
)
_ROLL_INTENT_OPT = typer.Option(
    "max-upside",
    "--roll-intent",
    help="Roll intent: max-upside | reduce-theta | increase-delta | de-risk.",
)
_ROLL_MIN_OI_OPT = typer.Option(0, "--roll-min-oi", help="Minimum open interest for roll candidates.")
_ROLL_MIN_VOLUME_OPT = typer.Option(0, "--roll-min-volume", help="Minimum volume for roll candidates.")
_ROLL_MAX_SPREAD_PCT_OPT = typer.Option(0.35, "--roll-max-spread-pct", help="Max spread pct gate.")
_ROLL_INCLUDE_BAD_QUOTES_OPT = typer.Option(
    False, "--roll-include-bad-quotes", help="Include bad quotes in roll selection."
)
_CACHE_DIR_OPT = typer.Option(
    Path("data/options_snapshots"),
    "--cache-dir",
    help="Directory for options chain snapshots.",
)
_CANDLE_CACHE_DIR_OPT = typer.Option(
    Path("data/candles"),
    "--candle-cache-dir",
    help="Directory for cached daily candles.",
)
_REPORTS_DIR_OPT = typer.Option(
    Path("data/reports/backtests"),
    "--reports-dir",
    help="Directory for backtest reports/artifacts.",
)
_RUN_ID_OPT = typer.Option(None, "--run-id", help="Explicit run id (default: symbol + timestamp).")


def _resolve_run_backtest():
    from options_helper.backtesting.runner import run_backtest

    return run_backtest


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


@app.command("run")
def backtest_run(
    symbol: str = _SYMBOL_OPT,
    contract_symbol: str | None = _CONTRACT_SYMBOL_OPT,
    expiry: str | None = _EXPIRY_OPT,
    strike: float | None = _STRIKE_OPT,
    option_type: OptionType = _OPTION_TYPE_OPT,
    start: str | None = _START_OPT,
    end: str | None = _END_OPT,
    fill_mode: str = _FILL_MODE_OPT,
    slippage_factor: float = _SLIPPAGE_FACTOR_OPT,
    initial_cash: float = _INITIAL_CASH_OPT,
    quantity: int = _QUANTITY_OPT,
    extension_low_pct: float = _EXTENSION_LOW_PCT_OPT,
    take_profit_pct: float = _TAKE_PROFIT_PCT_OPT,
    stop_loss_pct: float = _STOP_LOSS_PCT_OPT,
    max_holding_days: int = _MAX_HOLDING_DAYS_OPT,
    roll_dte_threshold: int | None = _ROLL_DTE_THRESHOLD_OPT,
    roll_horizon_months: int = _ROLL_HORIZON_MONTHS_OPT,
    roll_shape: str = _ROLL_SHAPE_OPT,
    roll_intent: str = _ROLL_INTENT_OPT,
    roll_min_oi: int = _ROLL_MIN_OI_OPT,
    roll_min_volume: int = _ROLL_MIN_VOLUME_OPT,
    roll_max_spread_pct: float = _ROLL_MAX_SPREAD_PCT_OPT,
    roll_include_bad_quotes: bool = _ROLL_INCLUDE_BAD_QUOTES_OPT,
    cache_dir: Path = _CACHE_DIR_OPT,
    candle_cache_dir: Path = _CANDLE_CACHE_DIR_OPT,
    reports_dir: Path = _REPORTS_DIR_OPT,
    run_id: str | None = _RUN_ID_OPT,
) -> None:
    """Run a daily backtest on offline snapshot history (not financial advice)."""
    from options_helper.data.backtesting_artifacts import write_backtest_artifacts

    console = Console(width=200)
    run_backtest = _resolve_run_backtest()
    expiry_date, start_date, end_date = _resolve_backtest_dates(contract_symbol=contract_symbol, expiry=expiry, strike=strike, start=start, end=end)
    fill_mode_norm = _normalize_fill_mode(fill_mode)
    roll_policy = _build_roll_policy(
        roll_dte_threshold=roll_dte_threshold,
        roll_horizon_months=roll_horizon_months,
        roll_shape=roll_shape,
        roll_intent=roll_intent,
        roll_min_oi=roll_min_oi,
        roll_min_volume=roll_min_volume,
        roll_max_spread_pct=roll_max_spread_pct,
        roll_include_bad_quotes=roll_include_bad_quotes,
    )
    data_source = _build_backtest_data_source(cache_dir=cache_dir, candle_cache_dir=candle_cache_dir)
    strategy = _build_backtest_strategy(extension_low_pct=extension_low_pct, take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct, max_holding_days=max_holding_days)
    result = run_backtest(
        data_source,
        symbol=symbol,
        contract_symbol=contract_symbol,
        expiry=expiry_date,
        strike=strike,
        option_type=option_type,
        strategy=strategy,
        start=start_date,
        end=end_date,
        fill_mode=fill_mode_norm,
        slippage_factor=slippage_factor,
        initial_cash=initial_cash,
        quantity=quantity,
        roll_policy=roll_policy,
    )
    run_id = _resolve_run_id(symbol=symbol, run_id=run_id)
    paths = write_backtest_artifacts(
        result,
        run_id=run_id,
        reports_dir=reports_dir,
        strategy_name=strategy.name,
        quantity=quantity,
    )
    _render_backtest_completion(console, run_id=run_id, summary_json=paths.summary_json, report_md=paths.report_md)


def _resolve_backtest_dates(
    *,
    contract_symbol: str | None,
    expiry: str | None,
    strike: float | None,
    start: str | None,
    end: str | None,
) -> tuple[date | None, date | None, date | None]:
    if contract_symbol is None and (expiry is None or strike is None):
        raise typer.BadParameter("Provide --contract-symbol or (--expiry and --strike).")
    expiry_date = _parse_date(expiry) if expiry else None
    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None
    return expiry_date, start_date, end_date


def _normalize_fill_mode(fill_mode: str) -> str:
    fill_mode_norm = fill_mode.strip().lower()
    if fill_mode_norm not in {"worst_case", "mark_slippage"}:
        raise typer.BadParameter("fill_mode must be 'worst_case' or 'mark_slippage'")
    return fill_mode_norm


def _build_roll_policy(
    *,
    roll_dte_threshold: int | None,
    roll_horizon_months: int,
    roll_shape: str,
    roll_intent: str,
    roll_min_oi: int,
    roll_min_volume: int,
    roll_max_spread_pct: float,
    roll_include_bad_quotes: bool,
):
    from options_helper.backtesting.roll import RollPolicy

    if roll_dte_threshold is None:
        return None
    if roll_shape not in {"out-same-strike", "out-up", "out-down"}:
        raise typer.BadParameter("roll_shape must be out-same-strike | out-up | out-down")
    if roll_intent not in {"max-upside", "reduce-theta", "increase-delta", "de-risk"}:
        raise typer.BadParameter("roll_intent must be max-upside | reduce-theta | increase-delta | de-risk")
    return RollPolicy(
        dte_threshold=roll_dte_threshold,
        horizon_months=roll_horizon_months,
        shape=roll_shape,  # type: ignore[arg-type]
        intent=roll_intent,  # type: ignore[arg-type]
        min_open_interest=roll_min_oi,
        min_volume=roll_min_volume,
        max_spread_pct=roll_max_spread_pct,
        include_bad_quotes=roll_include_bad_quotes,
    )


def _build_backtest_data_source(*, cache_dir: Path, candle_cache_dir: Path):
    from options_helper.backtesting.data_source import BacktestDataSource

    options_store = cli_deps.build_snapshot_store(cache_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)
    return BacktestDataSource(candle_store=candle_store, snapshot_store=options_store)


def _build_backtest_strategy(
    *,
    extension_low_pct: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_days: int,
):
    from options_helper.backtesting.strategy import BaselineLongCallStrategy

    return BaselineLongCallStrategy(
        extension_low_pct=extension_low_pct,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    )


def _resolve_run_id(*, symbol: str, run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    return f"{symbol.upper()}_{utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _render_backtest_completion(console: Console, *, run_id: str, summary_json: Path, report_md: Path) -> None:
    console.print(f"Backtest complete: {run_id}")
    console.print(f"Summary: {summary_json}")
    console.print(f"Report: {report_md}")
    console.print("Not financial advice.")


@app.command("report")
def backtest_report(
    run_id: str | None = typer.Option(None, "--run-id", help="Backtest run id."),
    latest: bool = typer.Option(False, "--latest", help="Load the most recent run in --reports-dir."),
    reports_dir: Path = typer.Option(
        Path("data/reports/backtests"),
        "--reports-dir",
        help="Directory for backtest reports/artifacts.",
    ),
) -> None:
    """Print the markdown report for a backtest run (not financial advice)."""
    console = Console(width=200)
    if latest and run_id is not None:
        raise typer.BadParameter("Provide either --run-id or --latest (not both).")
    if not latest and not run_id:
        raise typer.BadParameter("Provide --run-id or use --latest.")

    if latest:
        best: tuple[float, Path] | None = None
        for p in sorted(reports_dir.glob("*")):
            if not p.is_dir():
                continue
            report = p / "report.md"
            if not report.exists():
                continue
            try:
                mtime = float(report.stat().st_mtime)
            except Exception:  # noqa: BLE001
                continue
            if best is None or mtime > best[0]:
                best = (mtime, report)
        if best is None:
            console.print(f"[red]Error:[/red] No backtest reports found under {reports_dir}")
            raise typer.Exit(1)
        report_path = best[1]
    else:
        report_path = reports_dir / str(run_id) / "report.md"
    if not report_path.exists():
        console.print(f"[red]Error:[/red] Missing report: {report_path}")
        raise typer.Exit(1)
    console.print(report_path.read_text(encoding="utf-8"))
    console.print("Not financial advice.")
