from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console

from options_helper.cli_deps import build_candle_store, build_snapshot_store
from options_helper.models import OptionType
from options_helper.schemas.common import utc_now

app = typer.Typer(help="Offline options backtesting.")


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
    symbol: str = typer.Option(..., "--symbol", help="Underlying symbol to backtest."),
    contract_symbol: str | None = typer.Option(
        None,
        "--contract-symbol",
        help="Full contractSymbol to trade (uses snapshot data).",
    ),
    expiry: str | None = typer.Option(
        None,
        "--expiry",
        help="Contract expiry (YYYY-MM-DD) when contract-symbol is not provided.",
    ),
    strike: float | None = typer.Option(None, "--strike", help="Contract strike when contract-symbol is not provided."),
    option_type: OptionType = typer.Option("call", "--option-type", help="Option type (call/put)."),
    start: str | None = typer.Option(None, "--start", help="Start snapshot date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, "--end", help="End snapshot date (YYYY-MM-DD)."),
    fill_mode: str = typer.Option(
        "worst_case",
        "--fill-mode",
        help="Fill model: worst_case (bid/ask) or mark_slippage.",
    ),
    slippage_factor: float = typer.Option(0.5, "--slippage-factor", help="Slippage factor for mark_slippage."),
    initial_cash: float = typer.Option(10000.0, "--initial-cash", help="Starting cash balance."),
    quantity: int = typer.Option(1, "--quantity", help="Contracts per trade."),
    extension_low_pct: float = typer.Option(20.0, "--extension-low-pct", help="Entry extension percentile threshold."),
    take_profit_pct: float = typer.Option(0.8, "--take-profit-pct", help="Take-profit threshold (pct)."),
    stop_loss_pct: float = typer.Option(0.5, "--stop-loss-pct", help="Stop-loss threshold (pct)."),
    max_holding_days: int = typer.Option(15, "--max-holding-days", help="Time stop (days)."),
    roll_dte_threshold: int | None = typer.Option(
        None,
        "--roll-dte-threshold",
        help="Enable rolling when DTE <= threshold (omit to disable rolling).",
    ),
    roll_horizon_months: int = typer.Option(2, "--roll-horizon-months", help="Roll target horizon (months)."),
    roll_shape: str = typer.Option(
        "out-same-strike",
        "--roll-shape",
        help="Roll shape: out-same-strike | out-up | out-down.",
    ),
    roll_intent: str = typer.Option(
        "max-upside",
        "--roll-intent",
        help="Roll intent: max-upside | reduce-theta | increase-delta | de-risk.",
    ),
    roll_min_oi: int = typer.Option(0, "--roll-min-oi", help="Minimum open interest for roll candidates."),
    roll_min_volume: int = typer.Option(0, "--roll-min-volume", help="Minimum volume for roll candidates."),
    roll_max_spread_pct: float = typer.Option(0.35, "--roll-max-spread-pct", help="Max spread pct gate."),
    roll_include_bad_quotes: bool = typer.Option(
        False, "--roll-include-bad-quotes", help="Include bad quotes in roll selection."
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    reports_dir: Path = typer.Option(
        Path("data/reports/backtests"),
        "--reports-dir",
        help="Directory for backtest reports/artifacts.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit run id (default: symbol + timestamp)."),
) -> None:
    """Run a daily backtest on offline snapshot history (not financial advice)."""
    from options_helper.backtesting.data_source import BacktestDataSource
    from options_helper.backtesting.roll import RollPolicy
    from options_helper.backtesting.strategy import BaselineLongCallStrategy
    from options_helper.data.backtesting_artifacts import write_backtest_artifacts

    console = Console(width=200)
    run_backtest = _resolve_run_backtest()

    if contract_symbol is None:
        if expiry is None or strike is None:
            raise typer.BadParameter("Provide --contract-symbol or (--expiry and --strike).")
    expiry_date = _parse_date(expiry) if expiry else None
    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None

    fill_mode_norm = fill_mode.strip().lower()
    if fill_mode_norm not in {"worst_case", "mark_slippage"}:
        raise typer.BadParameter("fill_mode must be 'worst_case' or 'mark_slippage'")

    roll_policy = None
    if roll_dte_threshold is not None:
        if roll_shape not in {"out-same-strike", "out-up", "out-down"}:
            raise typer.BadParameter("roll_shape must be out-same-strike | out-up | out-down")
        if roll_intent not in {"max-upside", "reduce-theta", "increase-delta", "de-risk"}:
            raise typer.BadParameter("roll_intent must be max-upside | reduce-theta | increase-delta | de-risk")
        roll_policy = RollPolicy(
            dte_threshold=roll_dte_threshold,
            horizon_months=roll_horizon_months,
            shape=roll_shape,  # type: ignore[arg-type]
            intent=roll_intent,  # type: ignore[arg-type]
            min_open_interest=roll_min_oi,
            min_volume=roll_min_volume,
            max_spread_pct=roll_max_spread_pct,
            include_bad_quotes=roll_include_bad_quotes,
        )

    options_store = build_snapshot_store(cache_dir)
    candle_store = build_candle_store(candle_cache_dir)
    data_source = BacktestDataSource(candle_store=candle_store, snapshot_store=options_store)

    strategy = BaselineLongCallStrategy(
        extension_low_pct=extension_low_pct,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    )

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

    if run_id is None:
        run_id = f"{symbol.upper()}_{utc_now().strftime('%Y%m%dT%H%M%SZ')}"

    paths = write_backtest_artifacts(
        result,
        run_id=run_id,
        reports_dir=reports_dir,
        strategy_name=strategy.name,
        quantity=quantity,
    )

    console.print(f"Backtest complete: {run_id}")
    console.print(f"Summary: {paths.summary_json}")
    console.print(f"Report: {paths.report_md}")
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
