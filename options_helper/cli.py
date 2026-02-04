from __future__ import annotations

import csv
import json
import logging
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

if TYPE_CHECKING:
    import pandas as pd

from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.chain_metrics import compute_chain_report, execution_quality
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.confluence import ConfluenceInputs, ConfluenceScore, score_confluence
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.extension_scan import compute_current_extension_percentile
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.portfolio_risk import (
    PortfolioExposure,
    StressScenario,
    compute_portfolio_exposure,
    run_stress,
)
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.analysis.research import (
    Direction,
    OptionCandidate,
    TradeLevels,
    UnderlyingSetup,
    VolatilityContext,
    analyze_underlying,
    build_confluence_inputs,
    choose_expiry,
    compute_volatility_context,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.analysis.roll_plan import compute_roll_plan
from options_helper.analysis.roll_plan_multileg import compute_roll_plan_multileg
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain, black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, sma
from options_helper.analysis.journal_eval import build_journal_report, render_journal_report_markdown
from options_helper.data.candles import CandleCacheError, close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.derived import DERIVED_COLUMNS, DERIVED_SCHEMA_VERSION
from options_helper.data.earnings import EarningsRecord, safe_next_earnings_date
from options_helper.data.journal import SignalContext, SignalEvent
from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row
from options_helper.data.options_snapshotter import snapshot_full_chain_for_symbols
from options_helper.commands.backtest import app as backtest_app
from options_helper.commands.intraday import app as intraday_app
from options_helper.commands.technicals import app as technicals_app, technicals_extension_stats
from options_helper.commands.technicals_common import setup_technicals_logging
from options_helper.commands.watchlists import app as watchlists_app
from options_helper.cli_deps import (
    build_candle_store,
    build_derived_store,
    build_earnings_store,
    build_journal_store,
    build_provider,
    build_snapshot_store,
)
from options_helper.data.providers.base import MarketDataProvider
from options_helper.data.providers.runtime import reset_default_provider_name, set_default_provider_name
from options_helper.data.scanner import (
    evaluate_liquidity_for_symbols,
    prefilter_symbols,
    rank_shortlist_candidates,
    read_exclude_symbols,
    read_scanned_symbols,
    scan_symbols,
    score_shortlist_confluence,
    write_liquidity_csv,
    write_scan_csv,
    write_shortlist_csv,
    write_exclude_symbols,
    write_scanned_symbols,
    ScannerShortlistRow,
)
from options_helper.data.technical_backtesting_config import (
    ConfigError as TechnicalConfigError,
    load_technical_backtesting_config,
)
from options_helper.data.universe import UniverseError, load_universe_symbols
from options_helper.data.market_types import DataFetchError
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.models import Leg, MultiLegPosition, OptionType, Position, RiskProfile
from options_helper.observability import finalize_run_logger, setup_run_logger
from options_helper.reporting import MultiLegSummary, render_multi_leg_positions, render_positions, render_summary
from options_helper.reporting_chain import (
    render_chain_report_console,
    render_chain_report_markdown,
    render_compare_report_console,
)
from options_helper.reporting_briefing import (
    BriefingSymbolSection,
    build_briefing_payload,
    render_briefing_markdown,
    render_portfolio_table_markdown,
)
from options_helper.reporting_roll import render_roll_plan_console, render_roll_plan_multileg_console
from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.common import utc_now
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.flow import FlowArtifact
from options_helper.schemas.scanner_shortlist import (
    ScannerShortlistArtifact,
    ScannerShortlistRow as ScannerShortlistRowSchema,
)
from options_helper.storage import load_portfolio, save_portfolio, write_template
from options_helper.watchlists import build_default_watchlists, load_watchlists, save_watchlists
from options_helper.ui.dashboard import load_briefing_artifact, render_dashboard, resolve_briefing_paths
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot, compute_technical_snapshot

app = typer.Typer(add_completion=False)
app.add_typer(watchlists_app, name="watchlists")
derived_app = typer.Typer(help="Persist derived metrics from local snapshots.")
app.add_typer(derived_app, name="derived")
app.add_typer(technicals_app, name="technicals")
scanner_app = typer.Typer(help="Market opportunity scanner (not financial advice).")
app.add_typer(scanner_app, name="scanner")
journal_app = typer.Typer(help="Signal journal + outcome tracking.")
app.add_typer(journal_app, name="journal")
app.add_typer(backtest_app, name="backtest")
app.add_typer(intraday_app, name="intraday")


@app.callback()
def main(
    ctx: typer.Context,
    log_dir: Path = typer.Option(
        Path("data/logs"),
        "--log-dir",
        help="Directory to write per-command logs.",
    ),
    provider: str = typer.Option(
        "yahoo",
        "--provider",
        help="Market data provider (default: yahoo).",
    ),
) -> None:
    command_name = ctx.info_name or "options-helper"
    if ctx.invoked_subcommand:
        command_name = f"{command_name} {ctx.invoked_subcommand}"
    run_logger = setup_run_logger(log_dir, command_name)
    provider_token = set_default_provider_name(provider)

    def _on_close() -> None:
        reset_default_provider_name(provider_token)
        if run_logger is not None:
            finalize_run_logger(run_logger)

    ctx.call_on_close(_on_close)

    if run_logger is None:
        return


pd: object | None = None


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


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


def _default_position_id(symbol: str, expiry: date, strike: float, option_type: OptionType) -> str:
    suffix = "c" if option_type == "call" else "p"
    strike_str = f"{strike:g}".replace(".", "p")
    return f"{symbol.lower()}-{expiry.isoformat()}-{strike_str}{suffix}"


def _default_multileg_id(symbol: str, legs: list[Leg]) -> str:
    sorted_legs = sorted(legs, key=lambda l: (l.expiry, l.option_type, l.strike, l.side))
    tokens: list[str] = []
    for leg in sorted_legs:
        strike_str = f"{leg.strike:g}".replace(".", "p")
        token = f"{leg.side[0]}{leg.option_type[0]}{strike_str}@{leg.expiry.isoformat()}"
        tokens.append(token)
    if len(tokens) > 2:
        tokens = tokens[:2] + [f"n{len(legs)}"]
    return f"{symbol.lower()}-ml-" + "-".join(tokens)


def _parse_leg_spec(value: str) -> Leg:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 5 or len(parts) > 6:
        raise typer.BadParameter(
            "Invalid --leg format. Use side,type,expiry,strike,contracts[,ratio].",
            param_hint="--leg",
        )
    side = parts[0].lower()
    if side not in {"long", "short"}:
        raise typer.BadParameter("Invalid leg side (use long|short).", param_hint="--leg")
    opt_type = parts[1].lower()
    if opt_type not in {"call", "put"}:
        raise typer.BadParameter("Invalid leg type (use call|put).", param_hint="--leg")
    expiry = _parse_date(parts[2])
    try:
        strike = float(parts[3])
    except ValueError as exc:
        raise typer.BadParameter("Invalid leg strike (use number).", param_hint="--leg") from exc
    try:
        contracts = int(parts[4])
    except ValueError as exc:
        raise typer.BadParameter("Invalid leg contracts (use integer).", param_hint="--leg") from exc
    ratio = None
    if len(parts) == 6:
        try:
            ratio = float(parts[5])
        except ValueError as exc:
            raise typer.BadParameter("Invalid leg ratio (use number).", param_hint="--leg") from exc

    return Leg(
        side=side,  # type: ignore[arg-type]
        option_type=opt_type,  # type: ignore[arg-type]
        expiry=expiry,
        strike=strike,
        contracts=contracts,
        ratio=ratio,
    )


def _unique_id_with_suffix(existing_ids: set[str], base_id: str) -> str:
    if base_id not in existing_ids:
        return base_id
    for i in range(2, 1000):
        candidate = f"{base_id}-{i}"
        if candidate not in existing_ids:
            return candidate
    raise typer.BadParameter("Unable to generate a unique id; please supply --id.")


def _extract_float(row, key: str) -> float | None:
    if key not in row:
        return None
    val = row[key]
    try:
        if val is None:
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _extract_int(row, key: str) -> int | None:
    if key not in row:
        return None
    val = row[key]
    try:
        if val is None:
            return None
        return int(val)
    except Exception:  # noqa: BLE001
        return None


def _mark_price(*, bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    if ask is not None and ask > 0:
        return ask
    if bid is not None and bid > 0:
        return bid
    return None


def _clean_float(value: float | None) -> float | None:
    _ensure_pandas()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _clean_int(value: int | None) -> int | None:
    _ensure_pandas()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _iso_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _contract_symbol_from_row(row: pd.Series | dict | None) -> str | None:
    _ensure_pandas()
    if row is None:
        return None
    try:
        if isinstance(row, dict):
            raw = row.get("contractSymbol")
        else:
            raw = row["contractSymbol"] if "contractSymbol" in row else None
    except Exception:  # noqa: BLE001
        raw = None
    if raw is None:
        return None
    val = str(raw).strip()
    return val or None


def _build_position_journal_payload(
    metrics: PositionMetrics,
    advice: Advice,
    *,
    data_warnings: list[str],
) -> dict:
    technicals = {
        "sma20": _clean_float(metrics.sma20),
        "sma50": _clean_float(metrics.sma50),
        "rsi14": _clean_float(metrics.rsi14),
        "ema20": _clean_float(metrics.ema20),
        "ema50": _clean_float(metrics.ema50),
        "close_3d": _clean_float(metrics.close_3d),
        "rsi14_3d": _clean_float(metrics.rsi14_3d),
        "ema20_3d": _clean_float(metrics.ema20_3d),
        "ema50_3d": _clean_float(metrics.ema50_3d),
        "close_w": _clean_float(metrics.close_w),
        "rsi14_w": _clean_float(metrics.rsi14_w),
        "ema20_w": _clean_float(metrics.ema20_w),
        "ema50_w": _clean_float(metrics.ema50_w),
        "near_support_w": metrics.near_support_w,
        "breakout_w": metrics.breakout_w,
    }

    payload = {
        "position": metrics.position.model_dump(mode="json"),
        "as_of": _iso_date(metrics.as_of),
        "next_earnings_date": _iso_date(metrics.next_earnings_date),
        "metrics": {
            "underlying_price": _clean_float(metrics.underlying_price),
            "mark": _clean_float(metrics.mark),
            "bid": _clean_float(metrics.bid),
            "ask": _clean_float(metrics.ask),
            "spread": _clean_float(metrics.spread),
            "spread_pct": _clean_float(metrics.spread_pct),
            "execution_quality": metrics.execution_quality,
            "last": _clean_float(metrics.last),
            "implied_vol": _clean_float(metrics.implied_vol),
            "open_interest": _clean_int(metrics.open_interest),
            "volume": _clean_int(metrics.volume),
            "quality_label": metrics.quality_label,
            "last_trade_age_days": _clean_int(metrics.last_trade_age_days),
            "quality_warnings": list(metrics.quality_warnings or []),
            "dte": _clean_int(metrics.dte),
            "moneyness": _clean_float(metrics.moneyness),
            "pnl_abs": _clean_float(metrics.pnl_abs),
            "pnl_pct": _clean_float(metrics.pnl_pct),
            "delta": _clean_float(metrics.delta),
            "theta_per_day": _clean_float(metrics.theta_per_day),
            "technicals": technicals,
        },
        "advice": {
            "action": advice.action.value,
            "confidence": advice.confidence.value,
            "reasons": list(advice.reasons),
            "warnings": list(advice.warnings),
        },
        "data_warnings": list(data_warnings),
    }
    return payload


def _option_candidate_payload(candidate: OptionCandidate | None) -> dict | None:
    if candidate is None:
        return None
    return {
        "symbol": candidate.symbol,
        "option_type": candidate.option_type,
        "expiry": candidate.expiry.isoformat(),
        "dte": _clean_int(candidate.dte),
        "strike": _clean_float(candidate.strike),
        "mark": _clean_float(candidate.mark),
        "bid": _clean_float(candidate.bid),
        "ask": _clean_float(candidate.ask),
        "spread": _clean_float(candidate.spread),
        "spread_pct": _clean_float(candidate.spread_pct),
        "execution_quality": candidate.execution_quality,
        "last": _clean_float(candidate.last),
        "iv": _clean_float(candidate.iv),
        "delta": _clean_float(candidate.delta),
        "open_interest": _clean_int(candidate.open_interest),
        "volume": _clean_int(candidate.volume),
        "quality_score": _clean_float(candidate.quality_score),
        "quality_label": candidate.quality_label,
        "last_trade_age_days": _clean_int(candidate.last_trade_age_days),
        "rationale": list(candidate.rationale),
        "quality_warnings": list(candidate.quality_warnings),
        "warnings": list(candidate.warnings),
        "exclude": bool(candidate.exclude),
    }


def _vol_context_payload(ctx: VolatilityContext | None) -> dict | None:
    if ctx is None:
        return None
    return {
        "rv_20d": _clean_float(ctx.rv_20d),
        "iv_rv_20d": _clean_float(ctx.iv_rv_20d),
        "iv_percentile": _clean_float(ctx.iv_percentile),
        "atm_iv": _clean_float(ctx.atm_iv),
    }


def _levels_payload(levels: TradeLevels | None) -> dict | None:
    if levels is None:
        return None
    return {
        "entry": _clean_float(levels.entry),
        "pullback_entry": _clean_float(levels.pullback_entry),
        "stop": _clean_float(levels.stop),
        "notes": list(levels.notes),
    }


def _confluence_payload(score: ConfluenceScore | None) -> dict | None:
    if score is None:
        return None
    components = [comp.model_dump(mode="json") for comp in score.components]
    return {
        "total": _clean_float(score.total),
        "coverage": _clean_float(score.coverage),
        "components": components,
        "warnings": list(score.warnings),
    }


def _latest_scanner_run_dir(run_dir: Path, *, run_id: str | None = None) -> Path | None:
    if run_id:
        candidate = run_dir / run_id
        return candidate if candidate.exists() else None
    if not run_dir.exists():
        return None
    runs = [p for p in run_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name)[-1]


def _scanner_run_date(run_id: str | None) -> date | None:
    if not run_id:
        return None
    try:
        return datetime.strptime(run_id.split("_")[0], "%Y-%m-%d").date()
    except Exception:  # noqa: BLE001
        return None


def _read_scanner_shortlist(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sym = (row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "score": _clean_float(row.get("score")),
                    "coverage": _clean_float(row.get("coverage")),
                    "top_reasons": (row.get("top_reasons") or "").strip(),
                }
            )
    return rows


@derived_app.command("update")
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
    store = build_snapshot_store(cache_dir)
    derived = build_derived_store(derived_dir)
    candle_store = build_candle_store(candle_cache_dir)

    try:
        as_of_date = store.resolve_date(symbol, as_of)
        df = store.load_day(symbol, as_of_date)
        meta = store.load_meta(symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        report = compute_chain_report(
            df,
            symbol=symbol,
            as_of=as_of_date,
            spot=spot,
            expiries_mode="near",
            top=10,
            best_effort=True,
        )

        candles = candle_store.load(symbol)
        history = derived.load(symbol)
        row = DerivedRow.from_chain_report(report, candles=candles, derived_history=history)
        out_path = derived.upsert(symbol, row)
        console.print(f"Derived schema v{DERIVED_SCHEMA_VERSION} updated: {out_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@derived_app.command("show")
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
    derived = build_derived_store(derived_dir)

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


@derived_app.command("stats")
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
    derived = build_derived_store(derived_dir)

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


def _position_metrics(
    provider: MarketDataProvider | None,
    position: Position,
    *,
    risk_profile: RiskProfile,
    underlying_history: pd.DataFrame,
    underlying_last_price: float | None,
    as_of: date | None = None,
    next_earnings_date: date | None = None,
    snapshot_row: pd.Series | dict | None = None,
    cost_basis_override: float | None = None,
    include_pnl: bool = True,
    contract_sign: int = 1,
) -> PositionMetrics:
    _ensure_pandas()
    today = as_of or date.today()

    row = snapshot_row
    if row is None:
        if provider is None:
            raise ValueError("provider is required when snapshot_row is not provided")
        chain = provider.get_options_chain(position.symbol, position.expiry)
        df = chain.calls if position.option_type == "call" else chain.puts
        row = contract_row_by_strike(df, position.strike)

    bid = ask = last = iv = None
    oi = vol = None
    if row is not None:
        bid = _extract_float(row, "bid")
        ask = _extract_float(row, "ask")
        last = _extract_float(row, "lastPrice")
        iv = _extract_float(row, "impliedVolatility")
        oi = _extract_int(row, "openInterest")
        vol = _extract_int(row, "volume")

    quality_label = None
    last_trade_age_days = None
    quality_warnings: list[str] = []
    if row is not None:
        quality_df = compute_quote_quality(
            pd.DataFrame([row]),
            min_volume=risk_profile.min_volume,
            min_open_interest=risk_profile.min_open_interest,
            as_of=today,
        )
        if not quality_df.empty:
            q = quality_df.iloc[0]
            label = q.get("quality_label")
            if label is not None and not pd.isna(label):
                quality_label = str(label)
            age = q.get("last_trade_age_days")
            if age is not None and not pd.isna(age):
                try:
                    last_trade_age_days = int(age)
                except Exception:  # noqa: BLE001
                    last_trade_age_days = None
            warnings_val = q.get("quality_warnings")
            if isinstance(warnings_val, list):
                quality_warnings = [str(w) for w in warnings_val if w]

    mark = _mark_price(bid=bid, ask=ask, last=last)
    spread = spread_pct = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        spread = ask - bid
        mid = (ask + bid) / 2.0
        if mid > 0:
            spread_pct = spread / mid
    exec_quality = execution_quality(spread_pct)

    dte = (position.expiry - today).days
    dte_val = dte if dte >= 0 else 0

    underlying_price = underlying_last_price
    moneyness = None
    if underlying_price is not None:
        moneyness = underlying_price / position.strike

    pnl_abs = pnl_pct = None
    if mark is not None and include_pnl:
        basis = position.cost_basis if cost_basis_override is None else cost_basis_override
        pnl_abs = (mark - basis) * 100.0 * position.contracts
        pnl_pct = None if basis <= 0 else (mark - basis) / basis

    close_series: pd.Series | None = None
    volume_series: pd.Series | None = None
    if not underlying_history.empty and "Close" in underlying_history.columns:
        close_series = underlying_history["Close"].dropna()
    if not underlying_history.empty and "Volume" in underlying_history.columns:
        volume_series = underlying_history["Volume"].dropna()

    sma20 = sma(close_series, 20) if close_series is not None else None
    sma50 = sma(close_series, 50) if close_series is not None else None
    rsi14 = rsi(close_series, 14) if close_series is not None else None
    ema20 = ema(close_series, 20) if close_series is not None else None
    ema50 = ema(close_series, 50) if close_series is not None else None

    close_3d = rsi14_3d = ema20_3d = ema50_3d = None
    close_w = rsi14_w = ema20_w = ema50_w = None
    breakout_w = None
    near_support_w = None

    if close_series is not None and isinstance(close_series.index, pd.DatetimeIndex):
        close_3d_series = close_series.resample("3B").last().dropna()
        close_w_series = close_series.resample("W-FRI").last().dropna()

        close_3d = float(close_3d_series.iloc[-1]) if not close_3d_series.empty else None
        close_w = float(close_w_series.iloc[-1]) if not close_w_series.empty else None

        rsi14_3d = rsi(close_3d_series, 14) if not close_3d_series.empty else None
        ema20_3d = ema(close_3d_series, 20) if not close_3d_series.empty else None
        ema50_3d = ema(close_3d_series, 50) if not close_3d_series.empty else None

        rsi14_w = rsi(close_w_series, 14) if not close_w_series.empty else None
        ema20_w = ema(close_w_series, 20) if not close_w_series.empty else None
        ema50_w = ema(close_w_series, 50) if not close_w_series.empty else None

        if close_w is not None and ema50_w is not None and ema50_w != 0:
            near_support_w = abs(close_w - ema50_w) / abs(ema50_w) <= risk_profile.support_proximity_pct

        lookback = risk_profile.breakout_lookback_weeks
        breakout_price = (
            breakout_up(close_w_series, lookback)
            if position.option_type == "call"
            else breakout_down(close_w_series, lookback)
        )

        breakout_vol_ok = True
        if (
            breakout_price is True
            and volume_series is not None
            and isinstance(volume_series.index, pd.DatetimeIndex)
            and risk_profile.breakout_volume_mult > 0
        ):
            vol_w_series = volume_series.resample("W-FRI").sum().dropna()
            if len(vol_w_series) >= lookback + 1:
                last_vol = float(vol_w_series.iloc[-1])
                prev_avg = float(vol_w_series.iloc[-(lookback + 1) : -1].mean())
                if prev_avg > 0:
                    breakout_vol_ok = last_vol >= prev_avg * risk_profile.breakout_volume_mult

        if breakout_price is not None:
            breakout_w = bool(breakout_price and breakout_vol_ok)

    delta = theta_per_day = None
    if underlying_price is not None and iv is not None and dte_val > 0:
        greeks = black_scholes_greeks(
            option_type=position.option_type,
            s=underlying_price,
            k=position.strike,
            t_years=dte_val / 365.0,
            sigma=iv,
        )
        if greeks is not None:
            delta = greeks.delta
            theta_per_day = greeks.theta_per_day

    return PositionMetrics(
        position=position,
        contract_sign=contract_sign,
        underlying_price=underlying_price,
        mark=mark,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_pct=spread_pct,
        execution_quality=exec_quality,
        last=last,
        implied_vol=iv,
        open_interest=oi,
        volume=vol,
        quality_label=quality_label,
        last_trade_age_days=last_trade_age_days,
        quality_warnings=quality_warnings,
        dte=dte_val,
        moneyness=moneyness,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        sma20=sma20,
        sma50=sma50,
        rsi14=rsi14,
        ema20=ema20,
        ema50=ema50,
        close_3d=close_3d,
        rsi14_3d=rsi14_3d,
        ema20_3d=ema20_3d,
        ema50_3d=ema50_3d,
        close_w=close_w,
        rsi14_w=rsi14_w,
        ema20_w=ema20_w,
        ema50_w=ema50_w,
        near_support_w=near_support_w,
        breakout_w=breakout_w,
        delta=delta,
        theta_per_day=theta_per_day,
        as_of=today,
        next_earnings_date=next_earnings_date,
    )


@app.command("daily")
def daily_performance(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
) -> None:
    """Show best-effort daily P&L for the portfolio (based on options chain change fields)."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    provider = build_provider()

    from rich.table import Table

    table = Table(title="Daily Performance (best-effort)")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Chg", justify="right")
    table.add_column("%Chg", justify="right")
    table.add_column("Daily PnL $", justify="right")

    total_daily_pnl = 0.0
    total_prev_value = float(portfolio.cash)

    for p in portfolio.positions:
        try:
            chain = provider.get_options_chain(p.symbol, p.expiry)
            df = chain.calls if p.option_type == "call" else chain.puts
            row = contract_row_by_strike(df, p.strike)

            last = change = pct = None
            if row is not None:
                last = _extract_float(row, "lastPrice")
                change = _extract_float(row, "change")
                pct = _extract_float(row, "percentChange")

            q = compute_daily_performance_quote(
                last_price=last,
                change=change,
                percent_change_raw=pct,
                contracts=p.contracts,
            )

            if q.daily_pnl is not None:
                total_daily_pnl += q.daily_pnl
            if q.prev_close_price is not None:
                total_prev_value += q.prev_close_price * 100.0 * p.contracts
            elif q.last_price is not None:
                total_prev_value += q.last_price * 100.0 * p.contracts

            table.add_row(
                p.id,
                p.symbol,
                p.expiry.isoformat(),
                f"{p.strike:g}",
                str(p.contracts),
                "-" if q.last_price is None else f"${q.last_price:.2f}",
                "-" if q.change is None else f"{q.change:+.2f}",
                "-" if q.percent_change is None else f"{q.percent_change:+.1f}%",
                "-" if q.daily_pnl is None else f"{q.daily_pnl:+.2f}",
                style="green" if q.daily_pnl and q.daily_pnl > 0 else "red" if q.daily_pnl and q.daily_pnl < 0 else None,
            )
        except DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")

    console.print(table)

    denom = total_prev_value if total_prev_value > 0 else None
    total_pct = (total_daily_pnl / denom) if denom else None
    total_str = f"{total_daily_pnl:+.2f}"
    pct_str = "-" if total_pct is None else f"{total_pct:+.2%}"
    console.print(f"\nTotal daily PnL: ${total_str} ({pct_str})")


@app.command("snapshot-options")
def snapshot_options(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used to estimate spot).",
    ),
    window_pct: float = typer.Option(
        1.0,
        "--window-pct",
        min=0.0,
        max=2.0,
        help="Strike window around spot in --windowed mode (e.g. 1.0 = +/-100%).",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot price from daily candles.",
    ),
    require_data_date: str | None = typer.Option(
        None,
        "--require-data-date",
        help=(
            "Require each symbol's latest daily candle date to match this before snapshotting "
            "(YYYY-MM-DD or 'today'). If not met, the symbol is skipped to avoid mis-dated overwrites."
        ),
    ),
    require_data_tz: str = typer.Option(
        "America/Chicago",
        "--require-data-tz",
        help="Timezone used to interpret 'today' for --require-data-date (default: America/Chicago).",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist/--all-watchlists).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to snapshot (repeatable). When provided, snapshots those watchlists instead of portfolio positions.",
    ),
    all_watchlists: bool = typer.Option(
        False,
        "--all-watchlists",
        help="Snapshot all watchlists in the watchlists store (instead of portfolio positions).",
    ),
    all_expiries: bool = typer.Option(
        True,
        "--all-expiries/--position-expiries",
        help=(
            "Snapshot all expiries per symbol (default). Use --position-expiries to restrict "
            "(portfolio: expiries in positions; watchlists: nearest expiries unless --max-expiries)."
        ),
    ),
    full_chain: bool = typer.Option(
        True,
        "--full-chain/--windowed",
        help=(
            "Snapshot the full Yahoo options payload per expiry (writes .raw.json + a full CSV). "
            "Disables strike-window filtering. Use --windowed for smaller flow-focused snapshots."
        ),
    ),
    max_expiries: int | None = typer.Option(
        None,
        "--max-expiries",
        min=1,
        help="Optional cap on expiries per symbol (nearest first). Useful for watchlists or --full-chain.",
    ),
    risk_free_rate: float = typer.Option(
        0.0,
        "--risk-free-rate",
        help="Risk-free rate used for best-effort Black-Scholes Greeks (e.g. 0.05 = 5%).",
    ),
) -> None:
    """Save a once-daily options chain snapshot (full-chain + all expiries by default)."""
    _ensure_pandas()
    import numpy as np

    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = build_snapshot_store(cache_dir)
    provider = build_provider()
    candle_store = build_candle_store(candle_cache_dir, provider=provider)
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )

    want_full_chain = full_chain
    want_all_expiries = all_expiries

    use_watchlists = bool(watchlist) or all_watchlists

    watchlists_used: list[str] = []
    expiries_by_symbol: dict[str, set[date]] = {}
    symbols: list[str]

    if use_watchlists:
        wl = load_watchlists(watchlists_path)
        if all_watchlists:
            watchlists_used = sorted(wl.watchlists.keys())
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                console.print(f"No watchlists in {watchlists_path}")
                raise typer.Exit(0)
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise typer.BadParameter(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
            watchlists_used = sorted(set(watchlist))
    else:
        if not portfolio.positions:
            console.print("No positions.")
            raise typer.Exit(0)
        for p in portfolio.positions:
            expiries_by_symbol.setdefault(p.symbol, set()).add(p.expiry)
        symbols = sorted(expiries_by_symbol.keys())

    # Snapshot folder date should reflect the data period (latest available daily candle),
    # not the wall-clock run date. This matters for pre-market runs where the latest
    # daily candle is still yesterday's close.
    dates_used: set[date] = set()

    required_date: date | None = None
    if require_data_date is not None:
        spec = require_data_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_data_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(
                f"Invalid --require-data-date/--require-data-tz: {exc}",
                param_hint="--require-data-date",
            ) from exc

    mode = "watchlists" if use_watchlists else "portfolio"
    console.print(
        f"Snapshotting options chains for {len(symbols)} symbol(s) "
        f"({mode}, {'full-chain' if want_full_chain else 'windowed'})..."
    )

    # If the user snapshots watchlists with --position-expiries and doesn't explicitly cap expiries,
    # default to the nearest couple expiries to keep runtime and storage sane.
    effective_max_expiries = max_expiries
    if use_watchlists and not want_all_expiries and effective_max_expiries is None:
        effective_max_expiries = 2

    for symbol in symbols:
        history = candle_store.get_daily_history(symbol, period=spot_period)
        spot = last_close(history)
        data_date: date | None = history.index.max().date() if not history.empty else None
        if spot is None:
            try:
                underlying = provider.get_underlying(symbol, period=spot_period, interval="1d")
                spot = underlying.last_price
                if data_date is None and underlying.history is not None and not underlying.history.empty:
                    try:
                        data_date = underlying.history.index.max().date()
                    except Exception:  # noqa: BLE001
                        pass
            except DataFetchError:
                spot = None

        if spot is None or spot <= 0:
            console.print(f"[yellow]Warning:[/yellow] {symbol}: missing spot price; skipping snapshot.")
            continue

        if required_date is not None and data_date != required_date:
            got = "-" if data_date is None else data_date.isoformat()
            console.print(
                f"[yellow]Warning:[/yellow] {symbol}: candle date {got} != required {required_date.isoformat()}; "
                "skipping snapshot to avoid mis-dated overwrite."
            )
            continue

        effective_snapshot_date = data_date or date.today()
        dates_used.add(effective_snapshot_date)

        strike_min = spot * (1.0 - window_pct)
        strike_max = spot * (1.0 + window_pct)

        total_contracts = 0
        missing_bid_ask = 0
        stale_quotes = 0
        spread_pcts: list[float] = []

        meta = {
            "spot": spot,
            "spot_period": spot_period,
            "full_chain": want_full_chain,
            "all_expiries": want_all_expiries,
            "risk_free_rate": risk_free_rate,
            "window_pct": None if want_full_chain else window_pct,
            "strike_min": None if want_full_chain else strike_min,
            "strike_max": None if want_full_chain else strike_max,
            "snapshot_date": effective_snapshot_date.isoformat(),
            "symbol_source": mode,
            "watchlists": watchlists_used,
            "provider": provider_name,
        }
        if provider_version:
            meta["provider_version"] = provider_version

        expiries: list[date]
        if not use_watchlists and not want_all_expiries:
            expiries = sorted(expiries_by_symbol.get(symbol, set()))
        else:
            expiries = provider.list_option_expiries(symbol)
            if not expiries:
                console.print(f"[yellow]Warning:[/yellow] {symbol}: no listed option expiries; skipping snapshot.")
                continue
            if effective_max_expiries is not None:
                expiries = expiries[:effective_max_expiries]

        for exp in expiries:
            if want_full_chain:
                try:
                    raw = provider.get_options_chain_raw(symbol, exp)
                except DataFetchError as exc:
                    console.print(
                        f"[yellow]Warning:[/yellow] {symbol} {exp.isoformat()}: {exc}; skipping snapshot."
                    )
                    continue

                # Capture the full payload (raw) + a denormalized CSV for convenience.
                meta_with_underlying = dict(meta)
                meta_with_underlying["underlying"] = raw.get("underlying", {})

                calls = pd.DataFrame(raw.get("calls", []))
                puts = pd.DataFrame(raw.get("puts", []))
                calls["optionType"] = "call"
                puts["optionType"] = "put"
                calls["expiry"] = exp.isoformat()
                puts["expiry"] = exp.isoformat()

                df = pd.concat([calls, puts], ignore_index=True)
                df = add_black_scholes_greeks_to_chain(
                    df,
                    spot=spot,
                    expiry=exp,
                    as_of=effective_snapshot_date,
                    r=risk_free_rate,
                )

                quality = compute_quote_quality(
                    df,
                    min_volume=0,
                    min_open_interest=0,
                    as_of=effective_snapshot_date,
                )
                total_contracts += len(df)
                if not quality.empty:
                    q_warn = quality["quality_warnings"].tolist()
                    missing_bid_ask += sum("quote_missing_bid_ask" in w for w in q_warn if isinstance(w, list))
                    stale_quotes += sum("quote_stale" in w for w in q_warn if isinstance(w, list))
                    spread_series = pd.to_numeric(quality["spread_pct"], errors="coerce")
                    spread_series = spread_series.where(spread_series >= 0)
                    spread_pcts.extend(spread_series.dropna().tolist())

                store.save_expiry_snapshot(
                    symbol,
                    effective_snapshot_date,
                    expiry=exp,
                    snapshot=df,
                    meta=meta_with_underlying,
                )
                store.save_expiry_snapshot_raw(
                    symbol,
                    effective_snapshot_date,
                    expiry=exp,
                    raw=raw,
                )
                console.print(f"{symbol} {exp.isoformat()}: saved {len(df)} contracts (full)")
                continue

            # Default: windowed flow snapshot (compact columns).
            try:
                chain = provider.get_options_chain(symbol, exp)
            except DataFetchError as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] {symbol} {exp.isoformat()}: {exc}; skipping snapshot."
                )
                continue

            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["optionType"] = "call"
            puts["optionType"] = "put"
            calls["expiry"] = exp.isoformat()
            puts["expiry"] = exp.isoformat()

            df = pd.concat([calls, puts], ignore_index=True)
            if "strike" in df.columns:
                df = df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]

            df = add_black_scholes_greeks_to_chain(
                df,
                spot=spot,
                expiry=exp,
                as_of=effective_snapshot_date,
                r=risk_free_rate,
            )

            keep = [
                "contractSymbol",
                "optionType",
                "expiry",
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "change",
                "percentChange",
                "volume",
                "openInterest",
                "impliedVolatility",
                "inTheMoney",
                "bs_price",
                "bs_delta",
                "bs_gamma",
                "bs_theta_per_day",
                "bs_vega",
            ]
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

            store.save_expiry_snapshot(symbol, effective_snapshot_date, expiry=exp, snapshot=df, meta=meta)
            console.print(f"{symbol} {exp.isoformat()}: saved {len(df)} contracts")

        if want_full_chain and total_contracts > 0:
            spread_median = float(np.nanmedian(spread_pcts)) if spread_pcts else None
            spread_worst = float(np.nanmax(spread_pcts)) if spread_pcts else None
            store._upsert_meta(
                store._day_dir(symbol, effective_snapshot_date),
                {
                    "quote_quality": {
                        "contracts": int(total_contracts),
                        "missing_bid_ask_count": int(missing_bid_ask),
                        "missing_bid_ask_pct": float(missing_bid_ask / total_contracts),
                        "spread_pct_median": spread_median,
                        "spread_pct_worst": spread_worst,
                        "stale_quotes": int(stale_quotes),
                        "stale_pct": float(stale_quotes / total_contracts),
                    }
                },
            )

    if dates_used:
        days = ", ".join(sorted({d.isoformat() for d in dates_used}))
        console.print(f"Snapshot complete. Data date(s): {days}.")


@app.command("flow")
def flow_report(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist/--all-watchlists).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to include (repeatable). When provided, reports flow for those watchlists instead of portfolio positions.",
    ),
    all_watchlists: bool = typer.Option(
        False,
        "--all-watchlists",
        help="Report flow for all watchlists in the watchlists store (instead of portfolio positions).",
    ),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Only report a single symbol (overrides portfolio/watchlists selection).",
    ),
    window: int = typer.Option(
        1,
        "--window",
        min=1,
        max=30,
        help="Number of snapshot-to-snapshot deltas to net (requires N+1 snapshots).",
    ),
    group_by: str = typer.Option(
        "contract",
        "--group-by",
        help="Aggregation mode: contract|strike|expiry|expiry-strike",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top contracts per symbol to display."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/flow/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Report OI/volume deltas from locally captured snapshots (single-day or windowed)."""
    _ensure_pandas()
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = build_snapshot_store(cache_dir)
    use_watchlists = bool(watchlist) or all_watchlists
    if use_watchlists:
        wl = load_watchlists(watchlists_path)
        if all_watchlists:
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                console.print(f"No watchlists in {watchlists_path}")
                raise typer.Exit(0)
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise typer.BadParameter(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
    else:
        symbols = sorted({p.symbol for p in portfolio.positions})
        if not symbols and symbol is None:
            console.print("No positions.")
            raise typer.Exit(0)

    if symbol is not None:
        symbols = [symbol.upper()]

    pos_keys = {(p.symbol, p.expiry.isoformat(), float(p.strike), p.option_type) for p in portfolio.positions}

    from rich.table import Table

    group_by_norm = group_by.strip().lower()
    valid_group_by = {"contract", "strike", "expiry", "expiry-strike"}
    if group_by_norm not in valid_group_by:
        raise typer.BadParameter(
            f"Invalid --group-by (use {', '.join(sorted(valid_group_by))})",
            param_hint="--group-by",
        )
    group_by_val = cast(FlowGroupBy, group_by_norm)

    for sym in symbols:
        need = window + 1
        dates = store.latest_dates(sym, n=need)
        if len(dates) < need:
            console.print(f"[yellow]No flow data for {sym}:[/yellow] need at least {need} snapshots.")
            continue

        pair_flows: list[pd.DataFrame] = []
        for prev_date, today_date in zip(dates[:-1], dates[1:], strict=False):
            today_df = store.load_day(sym, today_date)
            prev_df = store.load_day(sym, prev_date)
            if today_df.empty or prev_df.empty:
                console.print(f"[yellow]No flow data for {sym}:[/yellow] empty snapshot(s) in window.")
                pair_flows = []
                break

            spot = _spot_from_meta(store.load_meta(sym, today_date))
            pair_flows.append(compute_flow(today_df, prev_df, spot=spot))

        if not pair_flows:
            continue

        start_date, end_date = dates[0], dates[-1]

        # Backward-compatible view: window=1 + per-contract list.
        if window == 1 and group_by_norm == "contract":
            prev_date, today_date = dates[-2], dates[-1]
            flow = pair_flows[-1]
            summary = summarize_flow(flow)

            console.print(
                f"\n[bold]{sym}[/bold] flow {prev_date.isoformat()} → {today_date.isoformat()} | "
                f"calls ΔOI$={summary['calls_delta_oi_notional']:,.0f} | puts ΔOI$={summary['puts_delta_oi_notional']:,.0f}"
            )

            if flow.empty:
                console.print("No flow rows.")
                continue

            if "deltaOI_notional" in flow.columns:
                flow = flow.assign(_abs=flow["deltaOI_notional"].abs())
                flow = flow.sort_values("_abs", ascending=False).drop(columns=["_abs"])

            table = Table(title=f"{sym} top {top} contracts by |ΔOI_notional|")
            table.add_column("*")
            table.add_column("Expiry")
            table.add_column("Type")
            table.add_column("Strike", justify="right")
            table.add_column("ΔOI", justify="right")
            table.add_column("OI", justify="right")
            table.add_column("Vol", justify="right")
            table.add_column("ΔOI$", justify="right")
            table.add_column("Class")

            for _, row in flow.head(top).iterrows():
                expiry = str(row.get("expiry", "-"))
                opt_type = str(row.get("optionType", "-"))
                strike = row.get("strike")
                strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
                key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
                in_port = key in pos_keys if strike_val is not None else False

                table.add_row(
                    "*" if in_port else "",
                    expiry,
                    opt_type,
                    "-" if strike_val is None else f"{strike_val:g}",
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("openInterest")) else f"{row.get('openInterest'):.0f}",
                    "-" if pd.isna(row.get("volume")) else f"{row.get('volume'):.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    str(row.get("flow_class", "-")),
                )

            console.print(table)

            if out is not None:
                net = aggregate_flow_window(pair_flows, group_by="contract")
                net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
                sort_cols = ["_abs"]
                ascending = [False]
                for c in ["expiry", "strike", "optionType", "contractSymbol"]:
                    if c in net.columns:
                        sort_cols.append(c)
                        ascending.append(True)
                net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

                base = out / "flow" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                out_path = base / f"{prev_date.isoformat()}_to_{today_date.isoformat()}_w1_contract.json"
                artifact_net = net.rename(
                    columns={
                        "contractSymbol": "contract_symbol",
                        "optionType": "option_type",
                        "deltaOI": "delta_oi",
                        "deltaOI_notional": "delta_oi_notional",
                        "size": "n_pairs",
                    }
                )
                payload = FlowArtifact(
                    schema_version=1,
                    generated_at=utc_now(),
                    as_of=today_date.isoformat(),
                    symbol=sym.upper(),
                    from_date=prev_date.isoformat(),
                    to_date=today_date.isoformat(),
                    window=1,
                    group_by="contract",
                    snapshot_dates=[prev_date.isoformat(), today_date.isoformat()],
                    net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                ).to_dict()
                if strict:
                    FlowArtifact.model_validate(payload)
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                console.print(f"\nSaved: {out_path}")
            continue

        # Windowed + aggregated views.
        net = aggregate_flow_window(pair_flows, group_by=group_by_val)
        if net.empty:
            console.print(f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()})")
            console.print("No net flow rows.")
            continue

        calls_premium = float(net[net["optionType"] == "call"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
        puts_premium = float(net[net["optionType"] == "put"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0

        console.print(
            f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()}) | "
            f"group-by={group_by_norm} | calls ΔOI$={calls_premium:,.0f} | puts ΔOI$={puts_premium:,.0f}"
        )

        net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
        sort_cols = ["_abs"]
        ascending = [False]
        for c in ["expiry", "strike", "optionType", "contractSymbol"]:
            if c in net.columns:
                sort_cols.append(c)
                ascending.append(True)

        net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

        def _render_zone_table(title: str) -> Table:
            t = Table(title=title)
            if group_by_norm == "contract":
                t.add_column("*")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                t.add_column("Expiry")
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                t.add_column("Strike", justify="right")
            t.add_column("Type")
            t.add_column("Net ΔOI", justify="right")
            t.add_column("Net ΔOI$", justify="right")
            t.add_column("Net Δ$", justify="right")
            t.add_column("N", justify="right")
            return t

        def _add_zone_row(t: Table, row) -> None:
            expiry = str(row.get("expiry", "-"))
            opt_type = str(row.get("optionType", "-"))
            strike = row.get("strike")
            strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
            key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
            in_port = key in pos_keys if strike_val is not None else False

            cells: list[str] = []
            if group_by_norm == "contract":
                cells.append("*" if in_port else "")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                cells.append(expiry)
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                cells.append("-" if strike_val is None else f"{strike_val:g}")
            cells.extend(
                [
                    opt_type,
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    "-" if pd.isna(row.get("delta_notional")) else f"{row.get('delta_notional'):+.0f}",
                    "-" if pd.isna(row.get("size")) else f"{int(row.get('size')):d}",
                ]
            )
            t.add_row(*cells)

        building = net[net["deltaOI_notional"] > 0].head(top)
        unwinding = net[net["deltaOI_notional"] < 0].head(top)

        t_build = _render_zone_table(f"{sym} building zones (top {top} by |net ΔOI$|)")
        for _, row in building.iterrows():
            _add_zone_row(t_build, row)
        console.print(t_build)

        t_unwind = _render_zone_table(f"{sym} unwinding zones (top {top} by |net ΔOI$|)")
        for _, row in unwinding.iterrows():
            _add_zone_row(t_unwind, row)
        console.print(t_unwind)

        if out is not None:
            base = out / "flow" / sym.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{start_date.isoformat()}_to_{end_date.isoformat()}_w{window}_{group_by_norm}.json"
            artifact_net = net.rename(
                columns={
                    "contractSymbol": "contract_symbol",
                    "optionType": "option_type",
                    "deltaOI": "delta_oi",
                    "deltaOI_notional": "delta_oi_notional",
                    "size": "n_pairs",
                }
            )
            payload = FlowArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=end_date.isoformat(),
                symbol=sym.upper(),
                from_date=start_date.isoformat(),
                to_date=end_date.isoformat(),
                window=window,
                group_by=group_by_norm,
                snapshot_dates=[d.isoformat() for d in dates],
                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
            ).to_dict()
            if strict:
                FlowArtifact.model_validate(payload)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")


@app.command("chain-report")
def chain_report(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to report on."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    format: str = typer.Option(
        "console",
        "--format",
        help="Output format: console|md|json",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/chains/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top strikes to show for walls/gamma."),
    include_expiry: list[str] = typer.Option(
        [],
        "--include-expiry",
        help="Include a specific expiry date (repeatable). When provided, overrides --expiries selection.",
    ),
    expiries: str = typer.Option(
        "near",
        "--expiries",
        help="Expiry selection mode: near|monthly|all (ignored when --include-expiry is used).",
    ),
    best_effort: bool = typer.Option(
        False,
        "--best-effort",
        help="Don't fail hard on missing fields; emit warnings and partial outputs.",
    ),
) -> None:
    """Offline options chain dashboard from local snapshot files."""
    console = Console()
    store = build_snapshot_store(cache_dir)

    try:
        as_of_date = store.resolve_date(symbol, as_of)
        df = store.load_day(symbol, as_of_date)
        meta = store.load_meta(symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        fmt = format.strip().lower()
        if fmt not in {"console", "md", "json"}:
            raise typer.BadParameter("Invalid --format (use console|md|json)", param_hint="--format")

        expiries_mode = expiries.strip().lower()
        if expiries_mode not in {"near", "monthly", "all"}:
            raise typer.BadParameter("Invalid --expiries (use near|monthly|all)", param_hint="--expiries")

        include_dates = [_parse_date(x) for x in include_expiry] if include_expiry else None

        report = compute_chain_report(
            df,
            symbol=symbol,
            as_of=as_of_date,
            spot=spot,
            expiries_mode=expiries_mode,  # type: ignore[arg-type]
            include_expiries=include_dates,
            top=top,
            best_effort=best_effort,
        )
        report_artifact = ChainReportArtifact(
            generated_at=utc_now(),
            **report.model_dump(),
        )
        if strict:
            ChainReportArtifact.model_validate(report_artifact.to_dict())

        if fmt == "console":
            render_chain_report_console(console, report)
        elif fmt == "md":
            console.print(render_chain_report_markdown(report))
        else:
            console.print(report_artifact.model_dump_json(indent=2))

        if out is not None:
            base = out / "chains" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            json_path = base / f"{as_of_date.isoformat()}.json"
            json_path.write_text(report_artifact.model_dump_json(indent=2), encoding="utf-8")

            # Write Markdown alongside JSON (human-friendly artifact).
            md_path = base / f"{as_of_date.isoformat()}.md"
            md_path.write_text(render_chain_report_markdown(report), encoding="utf-8")

            console.print(f"\nSaved: {json_path}")
            console.print(f"Saved: {md_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("compare")
def compare_snapshots(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to diff."),
    from_spec: str = typer.Option(
        "-1",
        "--from",
        help="From snapshot date (YYYY-MM-DD) or a negative offset relative to --to (e.g. -1).",
    ),
    to_spec: str = typer.Option("latest", "--to", help="To snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top rows to include per section."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/compare/{SYMBOL}/).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Diff two snapshot dates for a symbol (offline)."""
    console = Console()
    store = build_snapshot_store(cache_dir)

    try:
        to_date = store.resolve_date(symbol, to_spec)

        from_spec_norm = from_spec.strip().lower()
        if from_spec_norm.startswith("-") and from_spec_norm[1:].isdigit():
            from_date = store.resolve_relative_date(symbol, to_date=to_date, offset=int(from_spec_norm))
        else:
            from_date = store.resolve_date(symbol, from_spec_norm)

        df_from = store.load_day(symbol, from_date)
        df_to = store.load_day(symbol, to_date)
        meta_from = store.load_meta(symbol, from_date)
        meta_to = store.load_meta(symbol, to_date)

        spot_from = _spot_from_meta(meta_from)
        spot_to = _spot_from_meta(meta_to)
        if spot_from is None or spot_to is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        diff, report_from, report_to = compute_compare_report(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            from_df=df_from,
            to_df=df_to,
            spot_from=spot_from,
            spot_to=spot_to,
            top=top,
        )

        render_compare_report_console(console, diff)

        if out is not None:
            base = out / "compare" / symbol.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}.json"
            payload = CompareArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=to_date.isoformat(),
                symbol=symbol.upper(),
                from_report=report_from.model_dump(),
                to_report=report_to.model_dump(),
                diff=diff.model_dump(),
            ).to_dict()
            if strict:
                CompareArtifact.model_validate(payload)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("report-pack")
def report_pack(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (used for default paths)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (symbol source).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to include (repeatable). Default: positions, monitor, Scanner - Shortlist.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for technicals artifacts).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (writes {derived_dir}/{SYMBOL}.csv).",
    ),
    out: Path = typer.Option(
        Path("data/reports"),
        "--out",
        help="Output root for saved artifacts (writes under chains/compare/flow/derived/technicals).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest' (per-symbol)."),
    compare_from: str = typer.Option(
        "-1",
        "--compare-from",
        help="Compare spec: -1|-5|YYYY-MM-DD|none (relative offsets are per-symbol).",
    ),
    require_snapshot_date: str | None = typer.Option(
        None,
        "--require-snapshot-date",
        help="Only include symbols whose resolved --as-of snapshot date matches this (YYYY-MM-DD or 'today').",
    ),
    require_snapshot_tz: str = typer.Option(
        "America/Chicago",
        "--require-snapshot-tz",
        help="Timezone used to interpret 'today' for --require-snapshot-date (default: America/Chicago).",
    ),
    chain: bool = typer.Option(True, "--chain/--no-chain", help="Generate chain-report artifacts."),
    compare: bool = typer.Option(True, "--compare/--no-compare", help="Generate compare artifacts."),
    flow: bool = typer.Option(True, "--flow/--no-flow", help="Generate flow artifacts (contract + expiry-strike)."),
    derived: bool = typer.Option(True, "--derived/--no-derived", help="Upsert derived rows + write derived stats artifacts."),
    technicals: bool = typer.Option(
        True,
        "--technicals/--no-technicals",
        help="Generate technicals extension-stats artifacts (offline, from candle cache).",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (used for extension-stats artifacts).",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top rows/strikes to include in reports."),
    derived_window: int = typer.Option(60, "--derived-window", min=1, max=3650, help="Derived stats lookback window."),
    derived_trend_window: int = typer.Option(
        5, "--derived-trend-window", min=1, max=3650, help="Derived stats trend lookback window."
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Optional symmetric tail threshold for technicals extension-stats (e.g. 5 => low<=5, high>=95).",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Optional rolling window (years) for extension percentiles in technicals extension-stats.",
    ),
) -> None:
    """
    Offline report pack from local snapshots/candles.

    Generates per-symbol artifacts under `--out`:
    - chains/{SYMBOL}/{YYYY-MM-DD}.json + .md
    - compare/{SYMBOL}/{FROM}_to_{TO}.json
    - flow/{SYMBOL}/{FROM}_to_{TO}_w1_{group_by}.json
    - derived/{SYMBOL}/{ASOF}_w{N}_tw{M}.json
    - technicals/extension/{SYMBOL}/{ASOF}.json + .md
    """
    _ensure_pandas()
    console = Console(width=200)
    _ = load_portfolio(portfolio_path)  # validates the file exists/loads; used by cron scripts.

    wl = load_watchlists(watchlists_path)
    watchlists_used = watchlist[:] if watchlist else ["positions", "monitor", "Scanner - Shortlist"]
    symbols: set[str] = set()
    for name in watchlists_used:
        syms = wl.get(name)
        if not syms:
            console.print(f"[yellow]Warning:[/yellow] watchlist '{name}' missing/empty in {watchlists_path}")
            continue
        symbols.update(syms)

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        console.print("[yellow]No symbols selected (empty watchlists).[/yellow]")
        raise typer.Exit(0)

    store = build_snapshot_store(cache_dir)
    derived_store = build_derived_store(derived_dir)
    candle_store = build_candle_store(candle_cache_dir)

    required_date: date | None = None
    if require_snapshot_date is not None:
        spec = require_snapshot_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_snapshot_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(
                f"Invalid --require-snapshot-date/--require-snapshot-tz: {exc}",
                param_hint="--require-snapshot-date",
            ) from exc

    compare_norm = compare_from.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    out = out.expanduser()
    (out / "chains").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    (out / "flow").mkdir(parents=True, exist_ok=True)
    (out / "derived").mkdir(parents=True, exist_ok=True)
    (out / "technicals" / "extension").mkdir(parents=True, exist_ok=True)

    console.print(
        "Running offline report pack for "
        f"{len(symbols)} symbol(s) from watchlists: {', '.join([repr(x) for x in watchlists_used])}"
    )

    counts = {
        "symbols_total": len(symbols),
        "symbols_ok": 0,
        "chain_ok": 0,
        "compare_ok": 0,
        "flow_ok": 0,
        "derived_ok": 0,
        "technicals_ok": 0,
        "skipped_required_date": 0,
    }

    for sym in sorted(symbols):
        try:
            to_date = store.resolve_date(sym, as_of)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] {sym}: no snapshots ({exc})")
            continue

        if required_date is not None and to_date != required_date:
            counts["skipped_required_date"] += 1
            continue

        df_to = store.load_day(sym, to_date)
        meta_to = store.load_meta(sym, to_date)
        spot_to = _spot_from_meta(meta_to)
        if spot_to is None:
            console.print(f"[yellow]Warning:[/yellow] {sym}: missing spot in meta.json for {to_date.isoformat()}")
            continue

        # 1) Chain report (and derived update/stats based on it)
        chain_report_model = None
        if chain or derived:
            try:
                chain_report_model = compute_chain_report(
                    df_to,
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    expiries_mode="near",
                    top=top,
                    best_effort=True,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: chain-report failed: {exc}")
                chain_report_model = None

        if chain and chain_report_model is not None:
            try:
                base = out / "chains" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                json_path = base / f"{to_date.isoformat()}.json"
                md_path = base / f"{to_date.isoformat()}.md"
                chain_artifact = ChainReportArtifact(
                    generated_at=utc_now(),
                    **chain_report_model.model_dump(),
                )
                if strict:
                    ChainReportArtifact.model_validate(chain_artifact.to_dict())
                json_path.write_text(chain_artifact.model_dump_json(indent=2), encoding="utf-8")
                md_path.write_text(render_chain_report_markdown(chain_report_model), encoding="utf-8")
                counts["chain_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: failed writing chain artifacts: {exc}")

        if derived and chain_report_model is not None:
            try:
                candles = candle_store.load(sym)
                history = derived_store.load(sym)
                row = DerivedRow.from_chain_report(chain_report_model, candles=candles, derived_history=history)
                derived_store.upsert(sym, row)
                df_derived = derived_store.load(sym)
                if not df_derived.empty:
                    stats = compute_derived_stats(
                        df_derived,
                        symbol=sym,
                        as_of="latest",
                        window=derived_window,
                        trend_window=derived_trend_window,
                        metric_columns=[c for c in DERIVED_COLUMNS if c != "date"],
                    )
                    base = out / "derived" / sym.upper()
                    base.mkdir(parents=True, exist_ok=True)
                    stats_path = base / f"{stats.as_of}_w{derived_window}_tw{derived_trend_window}.json"
                    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
                    counts["derived_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: derived update/stats failed: {exc}")

        # 2) Compare + flow (requires previous snapshot)
        if compare_enabled and (compare or flow):
            try:
                from_date: date
                if compare_norm.startswith("-") and compare_norm[1:].isdigit():
                    from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
                else:
                    from_date = store.resolve_date(sym, compare_norm)

                df_from = store.load_day(sym, from_date)
                meta_from = store.load_meta(sym, from_date)
                spot_from = _spot_from_meta(meta_from)
                if spot_from is None:
                    raise ValueError("missing spot in from-date meta.json")

                if compare:
                    diff, report_from, report_to = compute_compare_report(
                        symbol=sym,
                        from_date=from_date,
                        to_date=to_date,
                        from_df=df_from,
                        to_df=df_to,
                        spot_from=spot_from,
                        spot_to=spot_to,
                        top=top,
                    )
                    base = out / "compare" / sym.upper()
                    base.mkdir(parents=True, exist_ok=True)
                    out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}.json"
                    payload = CompareArtifact(
                        schema_version=1,
                        generated_at=utc_now(),
                        as_of=to_date.isoformat(),
                        symbol=sym.upper(),
                        from_report=report_from.model_dump(),
                        to_report=report_to.model_dump(),
                        diff=diff.model_dump(),
                    ).to_dict()
                    if strict:
                        CompareArtifact.model_validate(payload)
                    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    counts["compare_ok"] += 1

                if flow:
                    pair_flow = compute_flow(df_to, df_from, spot=spot_to)
                    if not pair_flow.empty:
                        for group_by in ("contract", "expiry-strike"):
                            net = aggregate_flow_window([pair_flow], group_by=cast(FlowGroupBy, group_by))
                            base = out / "flow" / sym.upper()
                            base.mkdir(parents=True, exist_ok=True)
                            out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}_w1_{group_by}.json"
                            artifact_net = net.rename(
                                columns={
                                    "contractSymbol": "contract_symbol",
                                    "optionType": "option_type",
                                    "deltaOI": "delta_oi",
                                    "deltaOI_notional": "delta_oi_notional",
                                    "size": "n_pairs",
                                }
                            )
                            payload = FlowArtifact(
                                schema_version=1,
                                generated_at=utc_now(),
                                as_of=to_date.isoformat(),
                                symbol=sym.upper(),
                                from_date=from_date.isoformat(),
                                to_date=to_date.isoformat(),
                                window=1,
                                group_by=group_by,
                                snapshot_dates=[from_date.isoformat(), to_date.isoformat()],
                                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                            ).to_dict()
                            if strict:
                                FlowArtifact.model_validate(payload)
                            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                        counts["flow_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: compare/flow skipped: {exc}")

        # 3) Technicals extension stats artifacts (offline, candle cache)
        if technicals:
            try:
                technicals_extension_stats(
                    symbol=sym,
                    ohlc_path=None,
                    cache_dir=candle_cache_dir,
                    config_path=technicals_config,
                    tail_pct=tail_pct,
                    percentile_window_years=percentile_window_years,
                    out=out / "technicals" / "extension",
                    write_json=True,
                    write_md=True,
                    print_to_console=False,
                )
                counts["technicals_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: technicals extension-stats failed: {exc}")

        counts["symbols_ok"] += 1

    console.print(
        "Report pack complete: "
        f"symbols ok={counts['symbols_ok']}/{counts['symbols_total']} | "
        f"chain={counts['chain_ok']} compare={counts['compare_ok']} flow={counts['flow_ok']} "
        f"derived={counts['derived_ok']} technicals={counts['technicals_ok']} | "
        f"skipped(required_date)={counts['skipped_required_date']}"
    )


@app.command("briefing")
def briefing(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to include (repeatable). Adds to portfolio symbols.",
    ),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Only include a single symbol (overrides portfolio/watchlists selection).",
    ),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    compare: str = typer.Option(
        "-1",
        "--compare",
        help="Compare spec: -1|-5|YYYY-MM-DD|none (relative offsets are per-symbol).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for technical context).",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (canonical indicator definitions).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output path (Markdown) or directory. Default: data/reports/daily/{ASOF}.md",
    ),
    print_to_console: bool = typer.Option(
        False,
        "--print/--no-print",
        help="Print the briefing to the console (in addition to writing files).",
    ),
    write_json: bool = typer.Option(
        True,
        "--write-json/--no-write-json",
        help="Write a JSON version of the briefing alongside the Markdown (LLM-friendly).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    update_derived: bool = typer.Option(
        True,
        "--update-derived/--no-update-derived",
        help="Update derived metrics for included symbols (per-symbol CSV).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used when --update-derived).",
    ),
    top: int = typer.Option(3, "--top", min=1, max=10, help="Top rows to include in compare/flow sections."),
) -> None:
    """Generate a daily Markdown briefing for portfolio + optional watchlists (offline-first)."""
    _ensure_pandas()
    console = Console(width=200)
    portfolio = load_portfolio(portfolio_path)
    rp = portfolio.risk_profile

    positions_by_symbol: dict[str, list[Position]] = {}
    for p in portfolio.positions:
        positions_by_symbol.setdefault(p.symbol.upper(), []).append(p)

    portfolio_symbols = sorted({p.symbol.upper() for p in portfolio.positions})
    watch_symbols: list[str] = []
    watchlist_symbols_by_name: dict[str, list[str]] = {}
    if watchlist:
        try:
            wl = load_watchlists(watchlists_path)
            for name in watchlist:
                wl_symbols = wl.get(name)
                watch_symbols.extend(wl_symbols)
                watchlist_symbols_by_name[name] = wl_symbols
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] failed to load watchlists: {exc}")

    symbols = sorted(set(portfolio_symbols).union({s.upper() for s in watch_symbols if s}))
    if symbol is not None:
        symbols = [symbol.upper().strip()]

    if symbol is not None:
        sym = symbols[0] if symbols else ""
        symbol_sources_map: dict[str, set[str]] = {}
        if sym:
            if sym in portfolio_symbols:
                symbol_sources_map.setdefault(sym, set()).add("portfolio")
            symbol_sources_map.setdefault(sym, set()).add("manual")
        symbol_sources_payload = [
            {"symbol": sym, "sources": sorted(symbol_sources_map.get(sym, set()))}
            for sym in symbols
        ]
        watchlists_payload: list[dict[str, object]] = []
    else:
        symbol_sources_map = {}
        for sym in portfolio_symbols:
            symbol_sources_map.setdefault(sym, set()).add("portfolio")
        for name, syms in watchlist_symbols_by_name.items():
            for sym in syms:
                symbol_sources_map.setdefault(sym, set()).add(f"watchlist:{name}")

        symbol_sources_payload = [
            {"symbol": sym, "sources": sorted(symbol_sources_map.get(sym, set()))} for sym in symbols
        ]
        watchlists_payload = [
            {"name": name, "symbols": watchlist_symbols_by_name.get(name, [])}
            for name in watchlist
            if name in watchlist_symbols_by_name
        ]

    if not symbols:
        console.print("[red]Error:[/red] no symbols selected (empty portfolio and no watchlists)")
        raise typer.Exit(1)

    store = build_snapshot_store(cache_dir)
    derived_store = build_derived_store(derived_dir)
    candle_store = build_candle_store(candle_cache_dir)
    earnings_store = build_earnings_store(Path("data/earnings"))

    technicals_cfg: dict | None = None
    technicals_cfg_error: str | None = None
    try:
        technicals_cfg = load_technical_backtesting_config(technicals_config)
    except Exception as exc:  # noqa: BLE001
        technicals_cfg_error = str(exc)
    confluence_cfg = None
    confluence_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    if confluence_cfg_error:
        console.print(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")

    # Cache day snapshots for portfolio marks (best-effort).
    day_cache: dict[str, tuple[date, pd.DataFrame]] = {}
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}

    sections: list[BriefingSymbolSection] = []
    resolved_to_dates: list[date] = []
    compare_norm = compare.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    def _trend_from_weekly_flag(flag: bool | None) -> str | None:
        if flag is True:
            return "up"
        if flag is False:
            return "down"
        return None

    def _extension_percentile_from_snapshot(snapshot: TechnicalSnapshot | None) -> float | None:
        if snapshot is None or snapshot.extension_percentiles is None:
            return None
        daily = snapshot.extension_percentiles.daily
        if daily is None or not daily.current_percentiles:
            return None
        items: list[tuple[float, float]] = []
        for key, value in daily.current_percentiles.items():
            try:
                items.append((float(key), float(value)))
            except Exception:  # noqa: BLE001
                continue
        if not items:
            return None
        return sorted(items, key=lambda t: t[0])[-1][1]

    def _net_flow_delta_oi_notional(flow_net: pd.DataFrame | None) -> float | None:
        if flow_net is None or flow_net.empty:
            return None
        if "deltaOI_notional" not in flow_net.columns or "optionType" not in flow_net.columns:
            return None
        df = flow_net.copy()
        df["deltaOI_notional"] = pd.to_numeric(df["deltaOI_notional"], errors="coerce")
        df["optionType"] = df["optionType"].astype(str).str.lower()
        calls = df[df["optionType"] == "call"]["deltaOI_notional"].dropna()
        puts = df[df["optionType"] == "put"]["deltaOI_notional"].dropna()
        if calls.empty and puts.empty:
            return None
        return float(calls.sum()) - float(puts.sum())

    for sym in symbols:
        errors: list[str] = []
        warnings: list[str] = []
        chain = None
        compare_report = None
        flow_net = None
        technicals = None
        candles = None
        derived_updated = False
        derived_row = None
        confluence_score = None
        quote_quality = None
        next_earnings_date = safe_next_earnings_date(earnings_store, sym)
        next_earnings_by_symbol[sym] = next_earnings_date

        try:
            to_date = store.resolve_date(sym, as_of)
            resolved_to_dates.append(to_date)

            event_warnings: set[str] = set()
            base_risk = earnings_event_risk(
                today=to_date,
                expiry=None,
                next_earnings_date=next_earnings_date,
                warn_days=rp.earnings_warn_days,
                avoid_days=rp.earnings_avoid_days,
            )
            event_warnings.update(base_risk["warnings"])
            for pos in positions_by_symbol.get(sym, []):
                pos_risk = earnings_event_risk(
                    today=to_date,
                    expiry=pos.expiry,
                    next_earnings_date=next_earnings_date,
                    warn_days=rp.earnings_warn_days,
                    avoid_days=rp.earnings_avoid_days,
                )
                event_warnings.update(pos_risk["warnings"])
            if event_warnings:
                warnings.extend(sorted(event_warnings))

            df_to = store.load_day(sym, to_date)
            meta_to = store.load_meta(sym, to_date)
            spot_to = _spot_from_meta(meta_to)
            quote_quality = meta_to.get("quote_quality") if isinstance(meta_to, dict) else None
            if spot_to is None:
                raise ValueError("missing spot price in meta.json (run snapshot-options first)")

            day_cache[sym] = (to_date, df_to)

            if technicals_cfg is None:
                if technicals_cfg_error is not None:
                    warnings.append(f"technicals unavailable: {technicals_cfg_error}")
            else:
                try:
                    candles = candle_store.load(sym)
                    if candles.empty:
                        warnings.append("technicals unavailable: missing candle cache (run refresh-candles)")
                    else:
                        technicals = compute_technical_snapshot(candles, technicals_cfg)
                        if technicals is None:
                            warnings.append("technicals unavailable: insufficient candle history / warmup")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"technicals unavailable: {exc}")

            if candles is None:
                candles = pd.DataFrame()
            candles_by_symbol[sym] = candles

            chain = compute_chain_report(
                df_to,
                symbol=sym,
                as_of=to_date,
                spot=spot_to,
                expiries_mode="near",
                top=10,
                best_effort=True,
            )

            if update_derived:
                if candles is None:
                    candles = candle_store.load(sym)
                history = derived_store.load(sym)
                row = DerivedRow.from_chain_report(chain, candles=candles, derived_history=history)
                try:
                    derived_store.upsert(sym, row)
                    derived_updated = True
                    derived_row = row
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"derived update failed: {exc}")

            if compare_enabled:
                from_date: date | None = None
                if compare_norm.startswith("-") and compare_norm[1:].isdigit():
                    try:
                        from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"compare unavailable: {exc}")
                else:
                    from_date = store.resolve_date(sym, compare_norm)

                if from_date is not None and from_date != to_date:
                    df_from = store.load_day(sym, from_date)
                    meta_from = store.load_meta(sym, from_date)
                    spot_from = _spot_from_meta(meta_from)
                    if spot_from is None:
                        warnings.append("compare unavailable: missing spot in from-date meta.json")
                    elif df_from.empty or df_to.empty:
                        warnings.append("compare unavailable: missing snapshot CSVs for from/to date")
                    else:
                        compare_report, _, _ = compute_compare_report(
                            symbol=sym,
                            from_date=from_date,
                            to_date=to_date,
                            from_df=df_from,
                            to_df=df_to,
                            spot_from=spot_from,
                            spot_to=spot_to,
                            top=top,
                        )

                        try:
                            flow = compute_flow(df_to, df_from, spot=spot_to)
                            flow_net = aggregate_flow_window([flow], group_by="strike")
                        except Exception:  # noqa: BLE001
                            warnings.append("flow unavailable: compute failed")

            try:
                trend = _trend_from_weekly_flag(technicals.weekly_trend_up if technicals is not None else None)
                ext_pct = _extension_percentile_from_snapshot(technicals)
                flow_notional = _net_flow_delta_oi_notional(flow_net)
                iv_rv = derived_row.iv_rv_20d if derived_row is not None else None
                inputs = ConfluenceInputs(
                    weekly_trend=trend,
                    extension_percentile=ext_pct,
                    rsi_divergence=None,
                    flow_delta_oi_notional=flow_notional,
                    iv_rv_20d=iv_rv,
                )
                confluence_score = score_confluence(inputs, cfg=confluence_cfg)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"confluence unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        as_of_label = "-" if sym not in day_cache else day_cache[sym][0].isoformat()
        sections.append(
            BriefingSymbolSection(
                symbol=sym,
                as_of=as_of_label,
                chain=chain,
                compare=compare_report,
                flow_net=flow_net,
                technicals=technicals,
                confluence=confluence_score,
                errors=errors,
                warnings=warnings,
                quote_quality=quote_quality,
                derived_updated=derived_updated,
                derived=derived_row,
                next_earnings_date=next_earnings_date,
            )
        )

    if not resolved_to_dates:
        console.print("[red]Error:[/red] no snapshots found for selected symbols")
        raise typer.Exit(1)
    report_date = max(resolved_to_dates).isoformat()
    portfolio_rows: list[dict[str, str]] = []
    portfolio_rows_payload: list[dict[str, object]] = []
    portfolio_rows_with_pnl: list[tuple[float, dict[str, str]]] = []
    portfolio_metrics: list[PositionMetrics] = []
    for p in portfolio.positions:
        sym = p.symbol.upper()
        to_date, df_to = day_cache.get(sym, (None, pd.DataFrame()))

        mark = None
        spr_pct = None
        snapshot_row = None
        if not df_to.empty:
            sub = df_to.copy()
            if "expiry" in sub.columns:
                sub = sub[sub["expiry"].astype(str) == p.expiry.isoformat()]
            if "optionType" in sub.columns:
                sub = sub[sub["optionType"].astype(str).str.lower() == p.option_type]
            if "strike" in sub.columns:
                strike = pd.to_numeric(sub["strike"], errors="coerce")
                sub = sub.assign(_strike=strike)
                sub = sub[(sub["_strike"] - float(p.strike)).abs() < 1e-9]
            if not sub.empty:
                snapshot_row = sub.iloc[0]
                bid = _extract_float(snapshot_row, "bid")
                ask = _extract_float(snapshot_row, "ask")
                mark = _mark_price(bid=bid, ask=ask, last=_extract_float(snapshot_row, "lastPrice"))
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    if mid > 0:
                        spr_pct = (ask - bid) / mid

        history = candles_by_symbol.get(sym)
        if history is None or history.empty:
            try:
                history = candle_store.load(sym)
            except Exception:  # noqa: BLE001
                history = pd.DataFrame()
            candles_by_symbol[sym] = history

        try:
            last_price = close_asof(history, to_date) if to_date is not None else last_close(history)
            metrics = _position_metrics(
                None,
                p,
                risk_profile=rp,
                underlying_history=history,
                underlying_last_price=last_price,
                as_of=to_date,
                next_earnings_date=next_earnings_by_symbol.get(sym),
                snapshot_row=snapshot_row if snapshot_row is not None else {},
            )
            portfolio_metrics.append(metrics)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] portfolio exposure skipped for {p.id}: {exc}")

        pnl_abs = None
        pnl_pct = None
        if mark is not None:
            pnl_abs = (mark - p.cost_basis) * 100.0 * p.contracts
            pnl_pct = ((mark / p.cost_basis) - 1.0) if p.cost_basis > 0 else None

        portfolio_rows.append(
            {
                "id": p.id,
                "symbol": sym,
                "type": p.option_type,
                "expiry": p.expiry.isoformat(),
                "strike": f"{p.strike:g}",
                "ct": str(p.contracts),
                "cost": f"{p.cost_basis:.2f}",
                "mark": "-" if mark is None else f"{mark:.2f}",
                "pnl_$": "-" if pnl_abs is None else f"{pnl_abs:+.0f}",
                "pnl_%": "-" if pnl_pct is None else f"{pnl_pct * 100.0:+.1f}%",
                "spr_%": "-" if spr_pct is None else f"{spr_pct * 100.0:.1f}%",
                "as_of": "-" if to_date is None else to_date.isoformat(),
            }
        )
        portfolio_rows_payload.append(
            {
                "id": p.id,
                "symbol": sym,
                "option_type": p.option_type,
                "expiry": p.expiry.isoformat(),
                "strike": float(p.strike),
                "contracts": int(p.contracts),
                "cost_basis": float(p.cost_basis),
                "mark": None if mark is None else float(mark),
                "pnl": None if pnl_abs is None else float(pnl_abs),
                "pnl_pct": None if pnl_pct is None else float(pnl_pct),
                "spr_pct": None if spr_pct is None else float(spr_pct),
                "as_of": None if to_date is None else to_date.isoformat(),
            }
        )
        pnl_sort = float(pnl_pct) if pnl_pct is not None else float("-inf")
        portfolio_rows_with_pnl.append((pnl_sort, portfolio_rows[-1]))

    portfolio_table_md = None
    if portfolio_rows:
        # Sort by pnl% descending; rows without pnl% go last.
        portfolio_rows_sorted = [row for _, row in sorted(portfolio_rows_with_pnl, key=lambda r: r[0], reverse=True)]
        include_spread = any(r.get("spr_%") not in (None, "-") for r in portfolio_rows_sorted)
        portfolio_table_md = render_portfolio_table_markdown(portfolio_rows_sorted, include_spread=include_spread)

    portfolio_exposure = None
    portfolio_stress = None
    if portfolio_metrics:
        portfolio_exposure = compute_portfolio_exposure(portfolio_metrics)
        portfolio_stress = run_stress(
            portfolio_exposure,
            _build_stress_scenarios(stress_spot_pct=[], stress_vol_pp=5.0, stress_days=7),
        )

    md = render_briefing_markdown(
        report_date=report_date,
        portfolio_path=str(portfolio_path),
        symbol_sections=sections,
        portfolio_table_md=portfolio_table_md,
        top=top,
    )

    if out is None:
        out_path = Path("data/reports/daily") / f"{report_date}.md"
    else:
        out_path = out
        if out_path.suffix.lower() != ".md":
            out_path = out_path / f"{report_date}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    console.print(f"Saved: {out_path}")

    if write_json:
        payload = build_briefing_payload(
            report_date=report_date,
            as_of=report_date,
            portfolio_path=str(portfolio_path),
            symbol_sections=sections,
            top=top,
            technicals_config=str(technicals_config),
            portfolio_exposure=portfolio_exposure,
            portfolio_stress=portfolio_stress,
            portfolio_rows=portfolio_rows_payload,
            symbol_sources=symbol_sources_payload,
            watchlists=watchlists_payload,
        )
        if strict:
            BriefingArtifact.model_validate(payload)
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        console.print(f"Saved: {json_path}")

    if print_to_console:
        try:
            from rich.markdown import Markdown

            console.print(Markdown(md))
        except Exception:  # noqa: BLE001
            console.print(md)


@app.command("dashboard")
def dashboard(
    report_date: str = typer.Option(
        "latest",
        "--date",
        help="Briefing date (YYYY-MM-DD) or 'latest'.",
    ),
    reports_dir: Path = typer.Option(
        Path("data/reports"),
        "--reports-dir",
        help="Reports root (expects {reports_dir}/daily/{DATE}.json).",
    ),
    scanner_run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--scanner-run-dir",
        help="Scanner runs directory (for shortlist view).",
    ),
    scanner_run_id: str | None = typer.Option(
        None,
        "--scanner-run-id",
        help="Specific scanner run id to display (defaults to latest for the date).",
    ),
    max_shortlist_rows: int = typer.Option(
        20,
        "--max-shortlist-rows",
        min=1,
        max=200,
        help="Max rows to show in the scanner shortlist table.",
    ),
) -> None:
    """Render a read-only daily dashboard from briefing JSON + artifacts."""
    console = Console(width=200)
    try:
        paths = resolve_briefing_paths(reports_dir, report_date)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        artifact = load_briefing_artifact(paths.json_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] failed to load briefing JSON: {exc}")
        raise typer.Exit(1) from exc

    console.print(f"Briefing JSON: {paths.json_path}")
    render_dashboard(
        artifact=artifact,
        console=console,
        reports_dir=reports_dir,
        scanner_run_dir=scanner_run_dir,
        scanner_run_id=scanner_run_id,
        max_shortlist_rows=max_shortlist_rows,
    )


@app.command("roll-plan")
def roll_plan(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (positions + risk profile)."),
    position_id: str = typer.Option(..., "--id", help="Position id to plan a roll for."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    intent: str = typer.Option(
        "max-upside",
        "--intent",
        help="Intent: max-upside|reduce-theta|increase-delta|de-risk",
    ),
    horizon_months: int = typer.Option(..., "--horizon-months", min=1, max=60),
    shape: str = typer.Option(
        "out-same-strike",
        "--shape",
        help="Roll shape: out-same-strike|out-up|out-down",
    ),
    top: int = typer.Option(10, "--top", min=1, max=50, help="Number of candidates to display."),
    max_debit: float | None = typer.Option(
        None,
        "--max-debit",
        help="Max roll debit in dollars (total for position size).",
    ),
    min_credit: float | None = typer.Option(
        None,
        "--min-credit",
        help="Min roll credit in dollars (total for position size).",
    ),
    min_open_interest: int | None = typer.Option(
        None,
        "--min-open-interest",
        help="Override minimum open interest liquidity gate (default from risk profile).",
    ),
    min_volume: int | None = typer.Option(
        None,
        "--min-volume",
        help="Override minimum volume liquidity gate (default from risk profile).",
    ),
    include_bad_quotes: bool = typer.Option(
        False,
        "--include-bad-quotes",
        help="Include candidates with bad quote quality (best-effort).",
    ),
) -> None:
    """Propose and rank roll candidates for a single position using offline snapshots."""
    console = Console(width=200)

    portfolio = load_portfolio(portfolio_path)
    position = next((p for p in portfolio.positions if p.id == position_id), None)
    if position is None:
        raise typer.BadParameter(f"No position found with id: {position_id}", param_hint="--id")

    intent_norm = intent.strip().lower()
    if intent_norm not in {"max-upside", "reduce-theta", "increase-delta", "de-risk"}:
        raise typer.BadParameter(
            "Invalid --intent (use max-upside|reduce-theta|increase-delta|de-risk)",
            param_hint="--intent",
        )

    shape_norm = shape.strip().lower()
    if shape_norm not in {"out-same-strike", "out-up", "out-down"}:
        raise typer.BadParameter("Invalid --shape (use out-same-strike|out-up|out-down)", param_hint="--shape")

    rp = portfolio.risk_profile
    min_oi = rp.min_open_interest if min_open_interest is None else int(min_open_interest)
    min_vol = rp.min_volume if min_volume is None else int(min_volume)

    store = build_snapshot_store(cache_dir)
    earnings_store = build_earnings_store(Path("data/earnings"))
    next_earnings_date = safe_next_earnings_date(earnings_store, position.symbol)

    try:
        as_of_date = store.resolve_date(position.symbol, as_of)
        df = store.load_day(position.symbol, as_of_date)
        meta = store.load_meta(position.symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        if isinstance(position, MultiLegPosition):
            report = compute_roll_plan_multileg(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                horizon_months=horizon_months,
                min_open_interest=min_oi,
                min_volume=min_vol,
                top=top,
                include_bad_quotes=include_bad_quotes,
                max_debit=max_debit,
                min_credit=min_credit,
            )
            render_roll_plan_multileg_console(console, report)
        else:
            report = compute_roll_plan(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                intent=intent_norm,  # type: ignore[arg-type]
                horizon_months=horizon_months,
                shape=shape_norm,  # type: ignore[arg-type]
                min_open_interest=min_oi,
                min_volume=min_vol,
                max_debit=max_debit,
                min_credit=min_credit,
                top=top,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
                include_bad_quotes=include_bad_quotes,
            )

            render_roll_plan_console(console, report)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("earnings")
def earnings(
    symbol: str = typer.Argument(..., help="Ticker symbol (e.g. IREN)."),
    refresh: bool = typer.Option(False, "--refresh", help="Fetch from yfinance and update the local cache."),
    set_date: str | None = typer.Option(
        None,
        "--set",
        help="Manually set the next earnings date (YYYY-MM-DD). Overrides cached value.",
    ),
    clear: bool = typer.Option(False, "--clear", help="Delete cached earnings for the symbol."),
    cache_dir: Path = typer.Option(
        Path("data/earnings"),
        "--cache-dir",
        help="Directory for cached earnings dates.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the cached record as JSON."),
) -> None:
    """Show/cache the next earnings date (best-effort; Yahoo can be wrong/stale)."""
    console = Console(width=120)
    store = build_earnings_store(cache_dir)
    sym = symbol.upper().strip()

    if clear:
        deleted = store.delete(sym)
        console.print(f"Deleted: {sym}" if deleted else f"No cache found for: {sym}")
        return

    record: EarningsRecord | None = None

    if set_date is not None:
        d = _parse_date(set_date)
        record = EarningsRecord.manual(symbol=sym, next_earnings_date=d, note="Set via CLI --set.")
        out_path = store.save(record)
        console.print(f"Saved: {out_path}")
    else:
        record = store.load(sym)
        if refresh or record is None:
            try:
                ev = build_provider().get_next_earnings_event(sym)
            except DataFetchError as exc:
                console.print(f"[red]Data error:[/red] {exc}")
                raise typer.Exit(1)
            record = EarningsRecord(
                symbol=sym,
                fetched_at=datetime.now(tz=timezone.utc),
                source=ev.source,
                next_earnings_date=ev.next_date,
                window_start=ev.window_start,
                window_end=ev.window_end,
                raw=ev.raw,
                notes=[],
            )
            out_path = store.save(record)
            console.print(f"Saved: {out_path}")

    if record is None:
        console.print(f"No earnings record for {sym}.")
        raise typer.Exit(1)

    if json_out:
        console.print_json(json.dumps(record.model_dump(mode='json'), sort_keys=True))
        return

    today = date.today()
    if record.next_earnings_date is None:
        console.print(f"{sym} next earnings date: [yellow]unknown[/yellow] (source={record.source})")
        return

    days = (record.next_earnings_date - today).days
    suffix = "today" if days == 0 else f"in {days} day(s)"
    console.print(f"{sym} next earnings: [bold]{record.next_earnings_date.isoformat()}[/bold] ({suffix})")
    if record.window_start or record.window_end:
        console.print(
            f"Earnings window: {record.window_start.isoformat() if record.window_start else '-'}"
            f" → {record.window_end.isoformat() if record.window_end else '-'}"
        )
    console.print(f"Source: {record.source} (fetched_at={record.fetched_at.isoformat()})")


@app.command("refresh-earnings")
def refresh_earnings(
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (symbols source).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to refresh (default: all watchlists).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/earnings"),
        "--cache-dir",
        help="Directory for cached earnings dates.",
    ),
) -> None:
    """Fetch and cache next earnings dates for symbols in watchlists (best-effort)."""
    console = Console(width=120)
    wl = load_watchlists(watchlists_path)

    symbols: set[str] = set()
    if watchlist:
        for name in watchlist:
            symbols.update(wl.get(name))
    else:
        for syms in (wl.watchlists or {}).values():
            symbols.update(syms or [])

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        console.print("No symbols found (no watchlists or empty watchlist selection).")
        raise typer.Exit(0)

    store = build_earnings_store(cache_dir)
    provider = build_provider()

    ok = 0
    err = 0
    unknown = 0

    console.print(f"Refreshing earnings for {len(symbols)} symbol(s)...")
    for sym in sorted(symbols):
        try:
            ev = provider.get_next_earnings_event(sym)
            record = EarningsRecord(
                symbol=sym,
                fetched_at=datetime.now(tz=timezone.utc),
                source=ev.source,
                next_earnings_date=ev.next_date,
                window_start=ev.window_start,
                window_end=ev.window_end,
                raw=ev.raw,
                notes=[],
            )
            path = store.save(record)
            ok += 1
            if record.next_earnings_date is None:
                unknown += 1
                console.print(f"[yellow]Warning:[/yellow] {sym}: next earnings unknown (saved {path})")
            else:
                console.print(f"{sym}: {record.next_earnings_date.isoformat()} (saved {path})")
        except DataFetchError as exc:
            err += 1
            console.print(f"[red]Error:[/red] {sym}: {exc}")
        except Exception as exc:  # noqa: BLE001
            err += 1
            console.print(f"[red]Error:[/red] {sym}: {exc}")

    console.print(f"Done. ok={ok} unknown={unknown} errors={err}")


@app.command()
def research(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (risk profile + candle cache config)."),
    symbol: str | None = typer.Option(None, "--symbol", help="Run research for a single symbol."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: str = typer.Option(
        "watchlist",
        "--watchlist",
        help="Watchlist name to research (ignored when --symbol is provided).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    period: str = typer.Option(
        "5y",
        "--period",
        help="Daily candle period to ensure cached for research (yfinance period format).",
    ),
    window_pct: float = typer.Option(
        0.30,
        "--window-pct",
        min=0.0,
        max=2.0,
        help="Strike window around spot for option selection (e.g. 0.30 = +/-30%).",
    ),
    short_min_dte: int = typer.Option(30, "--short-min-dte"),
    short_max_dte: int = typer.Option(90, "--short-max-dte"),
    long_min_dte: int = typer.Option(365, "--long-min-dte"),
    long_max_dte: int = typer.Option(1500, "--long-max-dte"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the research output to a .txt report."),
    output_dir: Path = typer.Option(
        Path("data/research"),
        "--output-dir",
        help="Directory for saved research reports (ignored when --no-save).",
    ),
    include_bad_quotes: bool = typer.Option(
        False,
        "--include-bad-quotes",
        help="Include candidates with bad quote quality (best-effort).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used for IV percentile context).",
    ),
) -> None:
    """Recommend short-dated (30-90d) and long-dated (LEAPS) contracts based on technicals."""
    _ensure_pandas()
    portfolio = load_portfolio(portfolio_path)
    rp = portfolio.risk_profile

    if symbol:
        symbols = [symbol.strip().upper()]
    else:
        wl = load_watchlists(watchlists_path)
        symbols = wl.get(watchlist)
        if not symbols:
            raise typer.BadParameter(f"Watchlist '{watchlist}' is empty or missing in {watchlists_path}")

    provider = build_provider()
    candle_store = build_candle_store(candle_cache_dir, provider=provider)
    earnings_store = build_earnings_store(Path("data/earnings"))
    derived_store = build_derived_store(derived_dir)
    confluence_cfg = None
    confluence_cfg_error = None
    technicals_cfg = None
    technicals_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    try:
        technicals_cfg = load_technical_backtesting_config()
    except TechnicalConfigError as exc:
        technicals_cfg_error = str(exc)
    console = Console()

    from rich.table import Table

    report_buffer = None
    report_console = None
    symbol_outputs: dict[str, str] = {}
    symbol_candle_dates: dict[str, date] = {}
    symbol_candle_datetimes: dict[str, datetime] = {}
    if save:
        import io

        report_buffer = io.StringIO()
        report_console = Console(file=report_buffer, width=200, force_terminal=False)

    symbol_console: Console | None = None

    def emit(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        console.print(*args, **kwargs)
        if report_console is not None:
            report_console.print(*args, **kwargs)
        if symbol_console is not None:
            symbol_console.print(*args, **kwargs)

    def _format_confluence(score: ConfluenceScore) -> tuple[str, str, str]:
        total = f"{score.total:.0f}"
        coverage = f"{score.coverage * 100.0:.0f}%"
        parts = []
        for comp in score.components:
            if comp.score is None:
                continue
            if abs(comp.score) < 1e-9:
                continue
            parts.append(f"{comp.name} {comp.score:+.0f}")
        detail = ", ".join(parts) if parts else "neutral"
        return total, coverage, detail

    if confluence_cfg_error:
        emit(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")
    if technicals_cfg_error:
        emit(f"[yellow]Warning:[/yellow] technicals config unavailable: {technicals_cfg_error}")

    cached_history: dict[str, pd.DataFrame] = {}
    cached_setup: dict[str, UnderlyingSetup] = {}
    cached_extension_pct: dict[str, float | None] = {}
    pre_confluence: dict[str, ConfluenceScore] = {}

    for sym in symbols:
        history = candle_store.get_daily_history(sym, period=period)
        cached_history[sym] = history
        setup = analyze_underlying(sym, history=history, risk_profile=rp)
        cached_setup[sym] = setup

        ext_pct = None
        if technicals_cfg is not None and history is not None and not history.empty:
            try:
                ext_result = compute_current_extension_percentile(history, technicals_cfg)
                ext_pct = ext_result.percentile
            except Exception:  # noqa: BLE001
                ext_pct = None
        cached_extension_pct[sym] = ext_pct

        inputs = build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None)
        pre_confluence[sym] = score_confluence(inputs, cfg=confluence_cfg)

    if len(symbols) > 1:
        def _sort_key(sym: str) -> tuple[float, float, str]:
            score = pre_confluence.get(sym)
            coverage = score.coverage if score is not None else -1.0
            total = score.total if score is not None else -1.0
            return (-coverage, -total, sym)

        symbols = sorted(symbols, key=_sort_key)

    for sym in symbols:
        symbol_buffer = None
        symbol_console = None
        if save:
            import io

            symbol_buffer = io.StringIO()
            symbol_console = Console(file=symbol_buffer, width=200, force_terminal=False)

        history = cached_history.get(sym)
        if history is None:
            history = candle_store.get_daily_history(sym, period=period)
        if not history.empty:
            last_ts = history.index.max()
            # Candle store normalizes to tz-naive DatetimeIndex.
            symbol_candle_dates[sym] = last_ts.date()
            symbol_candle_datetimes[sym] = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
        as_of_date = symbol_candle_dates.get(sym)
        next_earnings_date = safe_next_earnings_date(earnings_store, sym)
        setup = cached_setup.get(sym)
        if setup is None:
            setup = analyze_underlying(sym, history=history, risk_profile=rp)
        ext_pct = cached_extension_pct.get(sym)

        emit(f"\n[bold]{sym}[/bold] — setup: {setup.direction.value}")
        for r in setup.reasons:
            emit(f"  - {r}")

        if setup.spot is None:
            emit("  - No spot price; skipping option selection.")
            continue

        levels = suggest_trade_levels(setup, history=history, risk_profile=rp)
        if levels.entry is not None:
            # Percent change relative to the latest close (setup.spot).
            pct_from_close = None
            try:
                if setup.spot:
                    pct_from_close = (float(levels.entry) / float(setup.spot) - 1.0) * 100.0
            except Exception:  # noqa: BLE001
                pct_from_close = None

            if pct_from_close is None:
                emit(f"  - Suggested entry (underlying): ${levels.entry:.2f}")
            else:
                emit(f"  - Suggested entry (underlying): ${levels.entry:.2f} ({pct_from_close:+.2f}%)")
        if levels.pullback_entry is not None:
            emit(f"  - Pullback entry (underlying): ${levels.pullback_entry:.2f}")
        if levels.stop is not None:
            emit(f"  - Suggested stop (underlying): ${levels.stop:.2f}")
        for note in levels.notes:
            emit(f"    - {note}")

        expiry_strs = [d.isoformat() for d in provider.list_option_expiries(sym)]
        if not expiry_strs:
            emit("  - No listed option expirations found.")
            continue

        expiry_as_of = as_of_date or date.today()
        short_exp = choose_expiry(
            expiry_strs, min_dte=short_min_dte, max_dte=short_max_dte, target_dte=60, today=expiry_as_of
        )
        long_exp = choose_expiry(
            expiry_strs, min_dte=long_min_dte, max_dte=long_max_dte, target_dte=540, today=expiry_as_of
        )
        if long_exp is None:
            # Fallback: pick the farthest expiry that still qualifies as "long".
            parsed = []
            for s in expiry_strs:
                try:
                    exp = date.fromisoformat(s)
                except ValueError:
                    continue
                dte = (exp - expiry_as_of).days
                parsed.append((dte, exp))
            parsed = [t for t in parsed if t[0] >= long_min_dte]
            if parsed:
                _, long_exp = max(parsed, key=lambda t: t[0])

        if setup.direction == Direction.NEUTRAL:
            confluence_score = score_confluence(
                build_confluence_inputs(
                    setup,
                    extension_percentile=ext_pct,
                    vol_context=None,
                ),
                cfg=confluence_cfg,
            )
            total, coverage, detail = _format_confluence(confluence_score)
            emit(f"  - Confluence score: {total} (coverage {coverage})")
            emit(f"    - Components: {detail}")
            if confluence_score.warnings:
                emit(f"    - Confluence warnings: {', '.join(confluence_score.warnings)}")
            emit("  - No strong directional setup; skipping contract recommendations.")
            continue

        opt_type: OptionType = "call" if setup.direction == Direction.BULLISH else "put"
        min_oi = rp.min_open_interest
        min_vol = rp.min_volume
        derived_history = derived_store.load(sym)
        vol_context = None

        table = Table(title=f"{sym} option ideas (best-effort)")
        table.add_column("Horizon")
        table.add_column("Expiry")
        table.add_column("DTE", justify="right")
        table.add_column("Type")
        table.add_column("Strike", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("IV", justify="right")
        table.add_column("IV/RV20", justify="right")
        table.add_column("IV pct", justify="right")
        table.add_column("OI", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("Spr%", justify="right")
        table.add_column("Exec", justify="right")
        table.add_column("Quality", justify="right")
        table.add_column("Stale", justify="right")
        table.add_column("Why")

        def _fmt_stale(age_days: int | None) -> str:
            if age_days is None:
                return "-"
            age = int(age_days)
            return f"{age}d" if age > 5 else "-"

        def _fmt_iv_rv(ctx) -> str:  # type: ignore[no-untyped-def]
            if ctx is None or ctx.iv_rv_20d is None:
                return "-"
            return f"{ctx.iv_rv_20d:.2f}x"

        def _fmt_iv_pct(ctx) -> str:  # type: ignore[no-untyped-def]
            if ctx is None or ctx.iv_percentile is None:
                return "-"
            return f"{ctx.iv_percentile:.0f}"

        if short_exp is not None:
            chain = provider.get_options_chain(sym, short_exp)
            if vol_context is None:
                vol_context = compute_volatility_context(
                    history=history,
                    spot=setup.spot,
                    calls=chain.calls,
                    puts=chain.puts,
                    derived_history=derived_history,
                )
            df = chain.calls if opt_type == "call" else chain.puts
            target_delta = 0.40 if opt_type == "call" else -0.40
            short_pick = select_option_candidate(
                df,
                symbol=sym,
                option_type=opt_type,
                expiry=short_exp,
                spot=setup.spot,
                target_delta=target_delta,
                window_pct=window_pct,
                min_open_interest=min_oi,
                min_volume=min_vol,
                as_of=expiry_as_of,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
                include_bad_quotes=include_bad_quotes,
            )
            if short_pick is not None:
                if short_pick.exclude:
                    warn = ", ".join(short_pick.warnings) if short_pick.warnings else "earnings_unknown"
                    emit(f"  - Excluded 30–90d candidate due to earnings_avoid_days ({warn}).")
                else:
                    why = "; ".join(short_pick.rationale[:2])
                    if short_pick.warnings:
                        why = f"{why}; Warnings: {', '.join(short_pick.warnings)}"
                    table.add_row(
                        "30–90d",
                        short_pick.expiry.isoformat(),
                        str(short_pick.dte),
                        short_pick.option_type,
                        f"{short_pick.strike:g}",
                        "-" if short_pick.mark is None else f"${short_pick.mark:.2f}",
                        "-" if short_pick.delta is None else f"{short_pick.delta:+.2f}",
                        "-" if short_pick.iv is None else f"{short_pick.iv:.1%}",
                        _fmt_iv_rv(vol_context),
                        _fmt_iv_pct(vol_context),
                        "-" if short_pick.open_interest is None else str(short_pick.open_interest),
                        "-" if short_pick.volume is None else str(short_pick.volume),
                        "-" if short_pick.spread_pct is None else f"{short_pick.spread_pct:.1%}",
                        "-" if short_pick.execution_quality is None else short_pick.execution_quality,
                        "-" if short_pick.quality_label is None else short_pick.quality_label,
                        _fmt_stale(short_pick.last_trade_age_days),
                        why,
                    )
                    if short_pick.warnings:
                        emit(f"  - Earnings warnings (30–90d): {', '.join(short_pick.warnings)}")
                    if short_pick.quality_warnings:
                        emit(f"  - Quote warnings (30–90d): {', '.join(short_pick.quality_warnings)}")
        else:
            emit(f"  - No expiries found in {short_min_dte}-{short_max_dte} DTE range.")

        if long_exp is not None:
            chain = provider.get_options_chain(sym, long_exp)
            if vol_context is None:
                vol_context = compute_volatility_context(
                    history=history,
                    spot=setup.spot,
                    calls=chain.calls,
                    puts=chain.puts,
                    derived_history=derived_history,
                )
            df = chain.calls if opt_type == "call" else chain.puts
            target_delta = 0.70 if opt_type == "call" else -0.70
            long_pick = select_option_candidate(
                df,
                symbol=sym,
                option_type=opt_type,
                expiry=long_exp,
                spot=setup.spot,
                target_delta=target_delta,
                window_pct=window_pct,
                min_open_interest=min_oi,
                min_volume=min_vol,
                as_of=expiry_as_of,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
                include_bad_quotes=include_bad_quotes,
            )
            if long_pick is not None:
                if long_pick.exclude:
                    warn = ", ".join(long_pick.warnings) if long_pick.warnings else "earnings_unknown"
                    emit(f"  - Excluded LEAPS candidate due to earnings_avoid_days ({warn}).")
                else:
                    why = "; ".join(long_pick.rationale[:2] + ["Longer DTE reduces theta pressure."])
                    if long_pick.warnings:
                        why = f"{why}; Warnings: {', '.join(long_pick.warnings)}"
                    table.add_row(
                        "LEAPS",
                        long_pick.expiry.isoformat(),
                        str(long_pick.dte),
                        long_pick.option_type,
                        f"{long_pick.strike:g}",
                        "-" if long_pick.mark is None else f"${long_pick.mark:.2f}",
                        "-" if long_pick.delta is None else f"{long_pick.delta:+.2f}",
                        "-" if long_pick.iv is None else f"{long_pick.iv:.1%}",
                        _fmt_iv_rv(vol_context),
                        _fmt_iv_pct(vol_context),
                        "-" if long_pick.open_interest is None else str(long_pick.open_interest),
                        "-" if long_pick.volume is None else str(long_pick.volume),
                        "-" if long_pick.spread_pct is None else f"{long_pick.spread_pct:.1%}",
                        "-" if long_pick.execution_quality is None else long_pick.execution_quality,
                        "-" if long_pick.quality_label is None else long_pick.quality_label,
                        _fmt_stale(long_pick.last_trade_age_days),
                        why,
                    )
                    if long_pick.warnings:
                        emit(f"  - Earnings warnings (LEAPS): {', '.join(long_pick.warnings)}")
                    if long_pick.quality_warnings:
                        emit(f"  - Quote warnings (LEAPS): {', '.join(long_pick.quality_warnings)}")
        else:
            emit(f"  - No expiries found in {long_min_dte}-{long_max_dte} DTE range.")

        confluence_score = score_confluence(
            build_confluence_inputs(
                setup,
                extension_percentile=ext_pct,
                vol_context=vol_context,
            ),
            cfg=confluence_cfg,
        )
        total, coverage, detail = _format_confluence(confluence_score)
        emit(f"  - Confluence score: {total} (coverage {coverage})")
        emit(f"    - Components: {detail}")
        if confluence_score.warnings:
            emit(f"    - Confluence warnings: {', '.join(confluence_score.warnings)}")

        emit(table)

        if symbol_buffer is not None:
            symbol_outputs[sym] = symbol_buffer.getvalue().lstrip()

    def _render_ticker_entry(*, sym: str, candle_day: date, run_dt: datetime, body: str) -> str:
        run_ts = run_dt.strftime("%Y-%m-%d %H:%M:%S")
        header = f"=== {candle_day.isoformat()} ===\nrun_at: {run_ts}\ncandles_through: {candle_day.isoformat()}\n"
        return f"{header}\n{body.strip()}\n"

    def _parse_ticker_entries(text: str) -> dict[str, str]:
        import re

        pattern = re.compile(r"^=== (\\d{4}-\\d{2}-\\d{2}) ===$", re.M)
        matches = list(pattern.finditer(text))
        if not matches:
            return {}

        entries: dict[str, str] = {}
        for idx, match in enumerate(matches):
            day = match.group(1)
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            entries[day] = text[start:end].strip()
        return entries

    def _upsert_ticker_entry(*, path: Path, candle_day: date, new_entry: str) -> None:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        entries = _parse_ticker_entries(existing)
        entries[candle_day.isoformat()] = new_entry.strip()
        ordered_days = sorted(entries.keys(), reverse=True)
        out = "\n\n".join(entries[d] for d in ordered_days).rstrip() + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out, encoding="utf-8")

    if save and report_buffer is not None:
        run_dt = datetime.now()
        candle_dt = max(symbol_candle_datetimes.values()) if symbol_candle_datetimes else run_dt
        candle_day = candle_dt.date()
        candle_stamp = candle_dt.strftime("%Y-%m-%d_%H%M%S")
        run_stamp = run_dt.strftime("%Y-%m-%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"research-{candle_stamp}-{run_stamp}.txt"
        header = (
            f"run_at: {run_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"candles_through: {candle_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"symbols: {', '.join(symbols)}\n\n"
        )
        out_path.write_text(header + report_buffer.getvalue().lstrip(), encoding="utf-8")

        tickers_dir = output_dir / "tickers"
        for sym, body in symbol_outputs.items():
            sym_day = symbol_candle_dates.get(sym) or candle_day
            entry = _render_ticker_entry(sym=sym, candle_day=sym_day, run_dt=run_dt, body=body)
            _upsert_ticker_entry(path=tickers_dir / f"{sym}.txt", candle_day=sym_day, new_entry=entry)

        console.print(f"\nSaved research report to {out_path}", soft_wrap=True)


@app.command("refresh-candles")
def refresh_candles(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (used to include position underlyings)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (symbols are included if present).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    period: str = typer.Option(
        "5y",
        "--period",
        help="Daily candle period to ensure cached (yfinance period format).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to include (default: all watchlists).",
    ),
) -> None:
    """Refresh cached daily candles for portfolio symbols and watchlists."""
    portfolio = load_portfolio(portfolio_path)
    symbols: set[str] = {p.symbol.upper() for p in portfolio.positions}

    wl = load_watchlists(watchlists_path)
    if watchlist:
        for name in watchlist:
            symbols.update(wl.get(name))
    else:
        for syms in (wl.watchlists or {}).values():
            symbols.update(syms or [])

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        Console().print("No symbols found (no positions and no watchlists).")
        raise typer.Exit(0)

    provider = build_provider()
    store = build_candle_store(candle_cache_dir, provider=provider)
    console = Console()
    console.print(f"Refreshing daily candles for {len(symbols)} symbol(s)...")

    for sym in sorted(symbols):
        try:
            hist = store.get_daily_history(sym, period=period)
            if hist.empty:
                console.print(f"[yellow]Warning:[/yellow] {sym}: no candles returned.")
            else:
                last_dt = hist.index.max()
                console.print(f"{sym}: cached through {last_dt.date().isoformat()}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {sym}: {exc}")


@scanner_app.command("run")
def scanner_run(
    universe: str = typer.Option(
        "file:data/universe/sec_company_tickers.json",
        "--universe",
        help="Universe source: us-all/us-equities/us-etfs or file:/path/to/list.txt.",
    ),
    universe_cache_dir: Path = typer.Option(
        Path("data/universe"),
        "--universe-cache-dir",
        help="Directory for cached universe lists.",
    ),
    universe_refresh_days: int = typer.Option(
        1,
        "--universe-refresh-days",
        help="Refresh universe cache if older than this many days.",
    ),
    max_symbols: int | None = typer.Option(
        None,
        "--max-symbols",
        min=1,
        help="Optional cap on number of symbols scanned (for dev/testing).",
    ),
    prefilter_mode: str = typer.Option(
        "default",
        "--prefilter-mode",
        help="Prefilter mode: default, aggressive, or none.",
    ),
    exclude_path: Path = typer.Option(
        Path("data/universe/exclude_symbols.txt"),
        "--exclude-path",
        help="Path to exclude symbols file (one ticker per line).",
    ),
    scanned_path: Path = typer.Option(
        Path("data/scanner/scanned_symbols.txt"),
        "--scanned-path",
        help="Path to scanned symbols file (one ticker per line).",
    ),
    skip_scanned: bool = typer.Option(
        True,
        "--skip-scanned/--no-skip-scanned",
        help="Skip symbols already recorded in the scanned file.",
    ),
    write_scanned: bool = typer.Option(
        True,
        "--write-scanned/--no-write-scanned",
        help="Persist scanned symbols so future runs skip them.",
    ),
    write_error_excludes: bool = typer.Option(
        True,
        "--write-error-excludes/--no-write-error-excludes",
        help="Persist symbols that error to the exclude file.",
    ),
    exclude_statuses: str = typer.Option(
        "error,no_candles",
        "--exclude-statuses",
        help="Comma-separated scan statuses to add to the exclude file.",
    ),
    error_flush_every: int = typer.Option(
        50,
        "--error-flush-every",
        min=1,
        help="Flush exclude file after this many new error symbols.",
    ),
    scanned_flush_every: int = typer.Option(
        250,
        "--scanned-flush-every",
        min=1,
        help="Flush scanned file after this many new symbols.",
    ),
    scan_period: str = typer.Option(
        "max",
        "--scan-period",
        help="Candle period to pull for the scan (yfinance period format).",
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Symmetric tail threshold percentile (e.g. 2.5 => low<=2.5, high>=97.5).",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Rolling window (years) for extension percentiles (default: auto 1y/3y).",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    all_watchlist_name: str = typer.Option(
        "Scanner - All",
        "--all-watchlist-name",
        help="Watchlist name for all tail symbols (replaced each run).",
    ),
    shortlist_watchlist_name: str = typer.Option(
        "Scanner - Shortlist",
        "--shortlist-watchlist-name",
        help="Watchlist name for liquid short list (replaced each run).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    options_cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--options-cache-dir",
        help="Directory for options chain snapshots.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot from daily candles for snapshotting.",
    ),
    backfill: bool = typer.Option(
        True,
        "--backfill/--no-backfill",
        help="Backfill max candle history for tail symbols.",
    ),
    snapshot_options: bool = typer.Option(
        True,
        "--snapshot-options/--no-snapshot-options",
        help="Snapshot full options chain for all expiries on tail symbols.",
    ),
    risk_free_rate: float = typer.Option(
        0.0,
        "--risk-free-rate",
        help="Risk-free rate used for best-effort Black-Scholes Greeks (e.g. 0.05 = 5%).",
    ),
    liquidity_min_dte: int = typer.Option(
        60,
        "--liquidity-min-dte",
        help="Minimum DTE for liquidity screening.",
    ),
    liquidity_min_volume: int = typer.Option(
        10,
        "--liquidity-min-volume",
        help="Minimum volume for liquidity screening.",
    ),
    liquidity_min_oi: int = typer.Option(
        500,
        "--liquidity-min-oi",
        help="Minimum open interest for liquidity screening.",
    ),
    run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--run-dir",
        help="Output root for scanner runs.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run id (default: timestamp).",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Max concurrent workers for scan (default: auto).",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        min=1,
        help="Batch size for scan requests.",
    ),
    batch_sleep_seconds: float = typer.Option(
        0.25,
        "--batch-sleep-seconds",
        min=0.0,
        help="Sleep between batches (seconds) to be polite to data sources.",
    ),
    reports_out: Path = typer.Option(
        Path("data/reports/technicals/extension"),
        "--reports-out",
        help="Output root for Extension Percentile Stats reports.",
    ),
    run_reports: bool = typer.Option(
        True,
        "--run-reports/--no-run-reports",
        help="Generate Extension Percentile Stats reports for shortlist symbols.",
    ),
    write_scan: bool = typer.Option(
        True,
        "--write-scan/--no-write-scan",
        help="Write scan CSV under the run directory.",
    ),
    write_liquidity: bool = typer.Option(
        True,
        "--write-liquidity/--no-write-liquidity",
        help="Write liquidity CSV under the run directory.",
    ),
    write_shortlist: bool = typer.Option(
        True,
        "--write-shortlist/--no-write-shortlist",
        help="Write shortlist CSV under the run directory.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--config",
        help="Config path.",
    ),
) -> None:
    """Scan the market for extension tails and build watchlists (not financial advice)."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)
    confluence_cfg = None
    confluence_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    if confluence_cfg_error:
        console.print(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")

    ext_cfg = cfg.get("extension_percentiles", {})
    tail_high_cfg = float(ext_cfg.get("tail_high_pct", 97.5))
    tail_low_cfg = float(ext_cfg.get("tail_low_pct", 2.5))
    if tail_pct is None:
        tail_low_pct = tail_low_cfg
        tail_high_pct = tail_high_cfg
    else:
        tp = float(tail_pct)
        if tp < 0.0 or tp >= 50.0:
            raise typer.BadParameter("--tail-pct must be >= 0 and < 50")
        tail_low_pct = tp
        tail_high_pct = 100.0 - tp

    if tail_low_pct >= tail_high_pct:
        raise typer.BadParameter("Tail thresholds must satisfy low < high")

    try:
        symbols = load_universe_symbols(
            universe,
            cache_dir=universe_cache_dir,
            refresh_days=universe_refresh_days,
        )
    except UniverseError as exc:
        console.print(f"[red]Universe error:[/red] {exc}")
        raise typer.Exit(1)

    symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})

    exclude_symbols = read_exclude_symbols(exclude_path) if exclude_path else set()
    if exclude_symbols:
        console.print(f"Loaded {len(exclude_symbols)} excluded symbol(s) from {exclude_path}")

    scanned_symbols: set[str] = set()
    if scanned_path and (skip_scanned or write_scanned):
        scanned_symbols = read_scanned_symbols(scanned_path)
        if scanned_symbols:
            console.print(f"Loaded {len(scanned_symbols)} scanned symbol(s) from {scanned_path}")

    filtered, dropped = prefilter_symbols(
        symbols,
        mode=prefilter_mode,
        exclude=exclude_symbols,
        scanned=scanned_symbols if skip_scanned else None,
    )
    dropped_n = sum(dropped.values())
    if dropped_n:
        console.print(f"Prefiltered symbols: dropped {dropped_n} ({dropped})")
    symbols = filtered

    if max_symbols is not None:
        symbols = symbols[: int(max_symbols)]

    if not symbols:
        console.print("[yellow]No symbols found in universe.[/yellow]")
        raise typer.Exit(0)

    run_stamp = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = run_dir / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    new_error_symbols: set[str] = set()
    new_scanned_symbols: set[str] = set()

    status_set = {s.strip().lower() for s in exclude_statuses.split(",") if s.strip()}

    def _row_callback(row) -> None:  # noqa: ANN001
        if write_scanned and scanned_path is not None:
            sym = row.symbol
            if sym not in scanned_symbols:
                scanned_symbols.add(sym)
                new_scanned_symbols.add(sym)
                if len(new_scanned_symbols) >= int(scanned_flush_every):
                    write_scanned_symbols(scanned_path, scanned_symbols)
                    new_scanned_symbols.clear()
        if not write_error_excludes or exclude_path is None:
            return
        if str(row.status).strip().lower() not in status_set:
            return
        sym = row.symbol
        if sym not in exclude_symbols:
            exclude_symbols.add(sym)
            new_error_symbols.add(sym)
            if len(new_error_symbols) >= int(error_flush_every):
                write_exclude_symbols(exclude_path, exclude_symbols)
                new_error_symbols.clear()

    console.print(
        f"Scanning {len(symbols)} symbol(s) from `{universe}` (tail {tail_low_pct:.1f}/{tail_high_pct:.1f})..."
    )
    provider = build_provider()
    candle_store = build_candle_store(candle_cache_dir, provider=provider)
    scan_rows, tail_symbols = scan_symbols(
        symbols,
        candle_store=candle_store,
        cfg=cfg,
        scan_period=scan_period,
        tail_low_pct=float(tail_low_pct),
        tail_high_pct=float(tail_high_pct),
        percentile_window_years=percentile_window_years,
        workers=workers,
        batch_size=batch_size,
        batch_sleep_seconds=batch_sleep_seconds,
        row_callback=_row_callback,
    )

    scan_as_of_dates: list[date] = []
    for row in scan_rows:
        if not row.asof:
            continue
        try:
            scan_as_of_dates.append(date.fromisoformat(row.asof))
        except ValueError:
            continue
    scan_as_of = max(scan_as_of_dates).isoformat() if scan_as_of_dates else date.today().isoformat()

    if write_error_excludes and new_error_symbols and exclude_path is not None:
        write_exclude_symbols(exclude_path, exclude_symbols)
        console.print(f"Wrote {len(new_error_symbols)} new excluded symbol(s) to {exclude_path}")

    if write_scanned and new_scanned_symbols and scanned_path is not None:
        write_scanned_symbols(scanned_path, scanned_symbols)
        console.print(f"Wrote {len(new_scanned_symbols)} new scanned symbol(s) to {scanned_path}")

    if write_scan:
        scan_path = run_root / "scan.csv"
        write_scan_csv(scan_rows, scan_path)
        console.print(f"Wrote scan CSV: {scan_path}")

    wl = load_watchlists(watchlists_path)
    wl.set(all_watchlist_name, tail_symbols)
    save_watchlists(watchlists_path, wl)
    console.print(f"Updated watchlist `{all_watchlist_name}` ({len(tail_symbols)} symbol(s))")

    if not tail_symbols:
        wl.set(shortlist_watchlist_name, [])
        save_watchlists(watchlists_path, wl)
        console.print("[yellow]No tail symbols found; shortlist cleared.[/yellow]")
        if write_liquidity:
            liquidity_path = run_root / "liquidity.csv"
            write_liquidity_csv([], liquidity_path)
            console.print(f"Wrote liquidity CSV: {liquidity_path}")
        if write_shortlist:
            shortlist_csv = run_root / "shortlist.csv"
            write_shortlist_csv([], shortlist_csv)
            console.print(f"Wrote shortlist CSV: {shortlist_csv}")
            shortlist_json = run_root / "shortlist.json"
            payload = ScannerShortlistArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=scan_as_of,
                run_id=run_stamp,
                universe=universe,
                tail_low_pct=float(tail_low_pct),
                tail_high_pct=float(tail_high_pct),
                all_watchlist_name=all_watchlist_name,
                shortlist_watchlist_name=shortlist_watchlist_name,
                rows=[],
            ).to_dict()
            if strict:
                ScannerShortlistArtifact.model_validate(payload)
            shortlist_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"Wrote shortlist JSON: {shortlist_json}")
        shortlist_md = run_root / "shortlist.md"
        lines = [
            f"# Scanner Shortlist — {run_stamp}",
            "",
            f"- Universe: `{universe}`",
            f"- Tail threshold: `{tail_low_pct:.1f}` / `{tail_high_pct:.1f}`",
            f"- Tail watchlist: `{all_watchlist_name}`",
            f"- Shortlist watchlist: `{shortlist_watchlist_name}`",
            "- Ranking: `scanner score` (desc)",
            "- Symbols: `0`",
            "",
            "Not financial advice.",
            "",
            "## Symbols",
            "- (empty)",
        ]
        shortlist_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        console.print(f"Wrote shortlist summary: {shortlist_md}")
        console.print("Not financial advice.")
        return

    if backfill:
        console.print(f"Backfilling candles for {len(tail_symbols)} tail symbol(s)...")
        for sym in tail_symbols:
            try:
                candle_store.get_daily_history(sym, period="max")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: candle backfill failed: {exc}")

    if snapshot_options:
        console.print(f"Snapshotting full options chains for {len(tail_symbols)} tail symbol(s)...")
        snapshot_results = snapshot_full_chain_for_symbols(
            tail_symbols,
            cache_dir=options_cache_dir,
            candle_cache_dir=candle_cache_dir,
            spot_period=spot_period,
            max_expiries=None,
            risk_free_rate=risk_free_rate,
            symbol_source="scanner",
            watchlists=[all_watchlist_name],
            provider=provider,
        )
        ok = sum(1 for r in snapshot_results if r.status == "ok")
        console.print(f"Options snapshots complete: {ok}/{len(snapshot_results)} ok")

    options_store = build_snapshot_store(options_cache_dir)
    liquidity_rows, shortlist_symbols = evaluate_liquidity_for_symbols(
        tail_symbols,
        store=options_store,
        min_dte=liquidity_min_dte,
        min_volume=liquidity_min_volume,
        min_open_interest=liquidity_min_oi,
    )

    rank_results = {}
    if shortlist_symbols:
        rank_cfg = None
        if confluence_cfg is not None and isinstance(confluence_cfg, dict):
            rank_cfg = confluence_cfg.get("scanner_rank")
        derived_store = build_derived_store(derived_dir)
        rank_results = rank_shortlist_candidates(
            shortlist_symbols,
            candle_store=candle_store,
            rank_cfg=rank_cfg,
            scan_rows=scan_rows,
            liquidity_rows=liquidity_rows,
            derived_store=derived_store,
            period=scan_period,
        )

    scan_percentiles = {row.symbol.upper(): row.percentile for row in scan_rows}
    confluence_scores = score_shortlist_confluence(
        shortlist_symbols,
        candle_store=candle_store,
        confluence_cfg=confluence_cfg,
        extension_percentiles=scan_percentiles,
        period=scan_period,
    )
    if rank_results:
        def _shortlist_rank_key(sym: str) -> tuple[float, float, str]:
            score = rank_results.get(sym)
            total = score.score if score is not None else -1.0
            coverage = score.coverage if score is not None else -1.0
            return (-total, -coverage, sym)

        shortlist_symbols = sorted(shortlist_symbols, key=_shortlist_rank_key)
    elif confluence_scores:
        def _shortlist_sort_key(sym: str) -> tuple[float, float, str]:
            score = confluence_scores.get(sym)
            coverage = score.coverage if score is not None else -1.0
            total = score.total if score is not None else -1.0
            return (-coverage, -total, sym)

        shortlist_symbols = sorted(shortlist_symbols, key=_shortlist_sort_key)

    if write_liquidity:
        liquidity_path = run_root / "liquidity.csv"
        write_liquidity_csv(liquidity_rows, liquidity_path)
        console.print(f"Wrote liquidity CSV: {liquidity_path}")

    if write_shortlist:
        shortlist_csv = run_root / "shortlist.csv"
        rows: list[ScannerShortlistRow] = []
        schema_rows: list[ScannerShortlistRowSchema] = []
        for sym in shortlist_symbols:
            rank = rank_results.get(sym)
            reasons = "; ".join(rank.top_reasons) if rank is not None else ""
            rows.append(
                ScannerShortlistRow(
                    symbol=sym,
                    score=rank.score if rank is not None else None,
                    coverage=rank.coverage if rank is not None else None,
                    top_reasons=reasons,
                )
            )
            schema_rows.append(
                ScannerShortlistRowSchema(
                    symbol=sym,
                    score=rank.score if rank is not None else None,
                    coverage=rank.coverage if rank is not None else None,
                    top_reasons=reasons or None,
                )
            )
        write_shortlist_csv(rows, shortlist_csv)
        console.print(f"Wrote shortlist CSV: {shortlist_csv}")
        shortlist_json = run_root / "shortlist.json"
        payload = ScannerShortlistArtifact(
            schema_version=1,
            generated_at=utc_now(),
            as_of=scan_as_of,
            run_id=run_stamp,
            universe=universe,
            tail_low_pct=float(tail_low_pct),
            tail_high_pct=float(tail_high_pct),
            all_watchlist_name=all_watchlist_name,
            shortlist_watchlist_name=shortlist_watchlist_name,
            rows=schema_rows,
        ).to_dict()
        if strict:
            ScannerShortlistArtifact.model_validate(payload)
        shortlist_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote shortlist JSON: {shortlist_json}")

    wl.set(shortlist_watchlist_name, shortlist_symbols)
    save_watchlists(watchlists_path, wl)
    console.print(f"Updated watchlist `{shortlist_watchlist_name}` ({len(shortlist_symbols)} symbol(s))")

    if run_reports and shortlist_symbols:
        console.print(f"Running Extension Percentile Stats for {len(shortlist_symbols)} symbol(s)...")
        for sym in shortlist_symbols:
            try:
                technicals_extension_stats(
                    symbol=sym,
                    ohlc_path=None,
                    cache_dir=candle_cache_dir,
                    config_path=config_path,
                    tail_pct=tail_pct,
                    percentile_window_years=percentile_window_years,
                    out=reports_out,
                    write_json=True,
                    write_md=True,
                    print_to_console=False,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: extension-stats failed: {exc}")

    shortlist_md = run_root / "shortlist.md"
    lines = [
        f"# Scanner Shortlist — {run_stamp}",
        "",
        f"- Universe: `{universe}`",
        f"- Tail threshold: `{tail_low_pct:.1f}` / `{tail_high_pct:.1f}`",
        f"- Tail watchlist: `{all_watchlist_name}`",
        f"- Shortlist watchlist: `{shortlist_watchlist_name}`",
        "- Ranking: `scanner score` (desc)",
        f"- Symbols: `{len(shortlist_symbols)}`",
        "",
        "Not financial advice.",
        "",
        "## Symbols",
    ]
    if shortlist_symbols:
        for sym in shortlist_symbols:
            parts: list[str] = []
            rank = rank_results.get(sym)
            if rank is not None:
                parts.append(f"scanner {rank.score:.0f}, cov {rank.coverage * 100.0:.0f}%")
            score = confluence_scores.get(sym)
            if score is not None:
                parts.append(f"confluence {score.total:.0f}, cov {score.coverage * 100.0:.0f}%")
            if parts:
                lines.append(f"- `{sym}` ({'; '.join(parts)}) → `{reports_out / sym}`")
            else:
                lines.append(f"- `{sym}` → `{reports_out / sym}`")
    else:
        lines.append("- (empty)")
    shortlist_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    console.print(f"Wrote shortlist summary: {shortlist_md}")
    console.print("Not financial advice.")


@app.command()
def init(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    force: bool = typer.Option(False, "--force", help="Overwrite if file exists."),
) -> None:
    """Create a starter portfolio JSON file."""
    write_template(portfolio_path, force=force)
    Console().print(f"Wrote template portfolio to {portfolio_path}")


@app.command("list")
def list_positions(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
) -> None:
    """List positions in the portfolio file."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        return

    # Minimal list (no fetch)
    from rich.table import Table

    table = Table(title="Portfolio Positions")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Type")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Cost", justify="right")
    for p in portfolio.positions:
        table.add_row(
            p.id,
            p.symbol,
            p.option_type,
            p.expiry.isoformat(),
            f"{p.strike:g}",
            str(p.contracts),
            f"${p.cost_basis:.2f}",
        )
    console.print(table)


@app.command("add-position")
def add_position(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str = typer.Option(..., "--symbol"),
    expiry: str = typer.Option(..., "--expiry", help="Expiry date, e.g. 2026-04-17."),
    strike: float = typer.Option(..., "--strike"),
    option_type: OptionType = typer.Option(..., "--type", case_sensitive=False),
    contracts: int = typer.Option(1, "--contracts"),
    cost_basis: float = typer.Option(..., "--cost-basis", help="Premium per share (e.g. 0.45)."),
    position_id: str | None = typer.Option(None, "--id", help="Optional position id."),
    opened_at: str | None = typer.Option(None, "--opened-at", help="Optional open date (YYYY-MM-DD)."),
) -> None:
    """Add a position to the portfolio JSON."""
    portfolio = load_portfolio(portfolio_path)

    expiry_date = _parse_date(expiry)
    opened_at_date = _parse_date(opened_at) if opened_at else None

    symbol = symbol.upper()
    option_type = option_type.lower()  # type: ignore[assignment]

    pid = position_id or _default_position_id(symbol, expiry_date, strike, option_type)
    if any(p.id == pid for p in portfolio.positions):
        raise typer.BadParameter(f"Position id already exists: {pid}")

    position = Position(
        id=pid,
        symbol=symbol,
        option_type=option_type,
        expiry=expiry_date,
        strike=float(strike),
        contracts=int(contracts),
        cost_basis=float(cost_basis),
        opened_at=opened_at_date,
    )

    portfolio.positions.append(position)
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Added {pid}")


@app.command("add-spread")
def add_spread(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str = typer.Option(..., "--symbol"),
    legs: list[str] = typer.Option(
        ...,
        "--leg",
        help="Repeatable leg spec: side,type,expiry,strike,contracts[,ratio].",
    ),
    net_debit: float | None = typer.Option(
        None,
        "--net-debit",
        help="Net debit in dollars for the whole structure (optional).",
    ),
    position_id: str | None = typer.Option(None, "--id", help="Optional position id."),
    opened_at: str | None = typer.Option(None, "--opened-at", help="Optional open date (YYYY-MM-DD)."),
) -> None:
    """Add a multi-leg (spread) position to the portfolio JSON."""
    portfolio = load_portfolio(portfolio_path)

    if len(legs) < 2:
        raise typer.BadParameter("Provide at least two --leg values.", param_hint="--leg")

    parsed_legs = [_parse_leg_spec(value) for value in legs]
    symbol = symbol.upper()
    opened_at_date = _parse_date(opened_at) if opened_at else None

    base_id = position_id or _default_multileg_id(symbol, parsed_legs)
    existing_ids = {p.id for p in portfolio.positions}
    if position_id is not None:
        if position_id in existing_ids:
            raise typer.BadParameter(f"Position id already exists: {position_id}", param_hint="--id")
        pid = position_id
    else:
        pid = _unique_id_with_suffix(existing_ids, base_id)

    position = MultiLegPosition(
        id=pid,
        symbol=symbol,
        legs=parsed_legs,
        net_debit=None if net_debit is None else float(net_debit),
        opened_at=opened_at_date,
    )

    portfolio.positions.append(position)
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Added {pid}")


@app.command("remove-position")
def remove_position(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    position_id: str = typer.Argument(..., help="Position id to remove."),
) -> None:
    """Remove a position by id."""
    portfolio = load_portfolio(portfolio_path)
    before = len(portfolio.positions)
    portfolio.positions = [p for p in portfolio.positions if p.id != position_id]
    after = len(portfolio.positions)
    if before == after:
        raise typer.BadParameter(f"No position found with id: {position_id}")
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Removed {position_id}")


@app.command()
def analyze(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    period: str = typer.Option("2y", "--period", help="Underlying history period (yfinance)."),
    interval: str = typer.Option("1d", "--interval", help="Underlying history interval (yfinance)."),
    offline: bool = typer.Option(
        False,
        "--offline/--online",
        help="Run from local snapshots + candle cache for deterministic as-of outputs.",
    ),
    as_of: str = typer.Option(
        "latest",
        "--as-of",
        help="Snapshot date (YYYY-MM-DD) or 'latest' (used with --offline).",
    ),
    offline_strict: bool = typer.Option(
        False,
        "--offline-strict",
        help="Fail if any position is missing snapshot coverage (used with --offline).",
    ),
    snapshots_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--snapshots-dir",
        help="Directory containing options snapshot folders (used with --offline).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--cache-dir",
        help="Directory for locally cached daily candles.",
    ),
    stress_spot_pct: list[float] = typer.Option(
        [],
        "--stress-spot-pct",
        help="Spot shock percent (repeatable). Use 5 for 5%% or 0.05 for 5%%.",
    ),
    stress_vol_pp: float = typer.Option(
        5.0,
        "--stress-vol-pp",
        help="Volatility shock in IV points (e.g. 5 = 5pp). Set to 0 to disable.",
    ),
    stress_days: int = typer.Option(
        7,
        "--stress-days",
        help="Time decay stress in days. Set to 0 to disable.",
    ),
) -> None:
    """Fetch data and print metrics + rule-based advice."""
    _ensure_pandas()
    if interval != "1d":
        raise typer.BadParameter("Only --interval 1d is supported for now (cache uses daily candles).")

    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    snapshot_store: OptionsSnapshotStore | None = None
    if offline:
        snapshot_store = build_snapshot_store(snapshots_dir)

    provider = None if offline else build_provider()
    candle_store = build_candle_store(cache_dir, provider=provider)
    earnings_store = build_earnings_store(Path("data/earnings"))

    history_by_symbol: dict[str, pd.DataFrame] = {}
    last_price_by_symbol: dict[str, float | None] = {}
    as_of_by_symbol: dict[str, date | None] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}
    snapshot_day_by_symbol: dict[str, pd.DataFrame] = {}
    chain_cache: dict[tuple[str, date], OptionsChain] = {}
    for sym in sorted({p.symbol for p in portfolio.positions}):
        history = pd.DataFrame()
        snapshot_date: date | None = None

        if offline:
            try:
                history = candle_store.load(sym)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {sym}: {exc}")
                history = pd.DataFrame()

            if snapshot_store is not None:
                try:
                    snapshot_date = snapshot_store.resolve_date(sym, as_of)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[yellow]Warning:[/yellow] snapshot date resolve failed for {sym}: {exc}")
                    snapshot_date = None

            if snapshot_date is None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                last_ts = history.index.max()
                if last_ts is not None and not pd.isna(last_ts):
                    snapshot_date = last_ts.date()

            if snapshot_date is not None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                history = history.loc[history.index <= pd.Timestamp(snapshot_date)]

            snap_df = pd.DataFrame()
            if snapshot_store is not None and snapshot_date is not None:
                try:
                    snap_df = snapshot_store.load_day(sym, snapshot_date)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[yellow]Warning:[/yellow] snapshot load failed for {sym}: {exc}")
                    snap_df = pd.DataFrame()
            snapshot_day_by_symbol[sym] = snap_df
        else:
            try:
                history = candle_store.get_daily_history(sym, period=period)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Candle cache error:[/red] {sym}: {exc}")
                history = pd.DataFrame()

        history_by_symbol[sym] = history

        if offline:
            as_of_by_symbol[sym] = snapshot_date
            last_price_by_symbol[sym] = (
                close_asof(history, snapshot_date) if snapshot_date is not None else last_close(history)
            )
        else:
            last_price_by_symbol[sym] = last_close(history)
            as_of_date = None
            if not history.empty and isinstance(history.index, pd.DatetimeIndex):
                last_ts = history.index.max()
                if last_ts is not None and not pd.isna(last_ts):
                    as_of_date = last_ts.date()
            as_of_by_symbol[sym] = as_of_date

        next_earnings_by_symbol[sym] = safe_next_earnings_date(earnings_store, sym)

    single_metrics: list[PositionMetrics] = []
    all_metrics: list[PositionMetrics] = []
    multi_leg_summaries: list[MultiLegSummary] = []
    advice_by_id: dict[str, Advice] = {}
    offline_missing: list[str] = []

    for p in portfolio.positions:
        try:
            if isinstance(p, MultiLegPosition):
                leg_metrics: list[PositionMetrics] = []
                net_mark_total = 0.0
                net_mark_ready = True
                dte_vals: list[int] = []
                warnings: list[str] = []
                low_oi = False
                low_vol = False
                bad_spread = False
                quote_flags = False
                missing_as_of_warned = False
                missing_day_warned = False

                for idx, leg in enumerate(p.legs, start=1):
                    snapshot_row = None
                    if offline:
                        snap_date = as_of_by_symbol.get(p.symbol)
                        df_snap = snapshot_day_by_symbol.get(p.symbol, pd.DataFrame())
                        row = None

                        if snap_date is None:
                            if not missing_as_of_warned:
                                offline_missing.append(f"{p.id}: missing offline as-of date for {p.symbol}")
                                missing_as_of_warned = True
                        elif df_snap.empty:
                            if not missing_day_warned:
                                offline_missing.append(
                                    f"{p.id}: missing snapshot day data for {p.symbol} (as-of {snap_date.isoformat()})"
                                )
                                missing_day_warned = True
                        else:
                            row = find_snapshot_row(
                                df_snap,
                                expiry=leg.expiry,
                                strike=leg.strike,
                                option_type=leg.option_type,
                            )
                            if row is None:
                                offline_missing.append(
                                    f"{p.id}: missing snapshot row for {p.symbol} {leg.expiry.isoformat()} "
                                    f"{leg.option_type} {leg.strike:g} (as-of {snap_date.isoformat()})"
                                )

                        snapshot_row = row if row is not None else {}
                    else:
                        row = None
                        if provider is not None:
                            key = (p.symbol, leg.expiry)
                            chain = chain_cache.get(key)
                            if chain is None:
                                chain = provider.get_options_chain(p.symbol, leg.expiry)
                                chain_cache[key] = chain
                            df_chain = chain.calls if leg.option_type == "call" else chain.puts
                            row = contract_row_by_strike(df_chain, leg.strike)
                        snapshot_row = row if row is not None else {}

                    leg_position = Position(
                        id=f"{p.id}:leg{idx}",
                        symbol=p.symbol,
                        option_type=leg.option_type,
                        expiry=leg.expiry,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        cost_basis=0.0,
                        opened_at=p.opened_at,
                    )

                    metrics = _position_metrics(
                        provider,
                        leg_position,
                        risk_profile=portfolio.risk_profile,
                        underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                        underlying_last_price=last_price_by_symbol.get(p.symbol),
                        as_of=as_of_by_symbol.get(p.symbol),
                        next_earnings_date=next_earnings_by_symbol.get(p.symbol),
                        snapshot_row=snapshot_row,
                        include_pnl=False,
                        contract_sign=1 if leg.side == "long" else -1,
                    )
                    leg_metrics.append(metrics)
                    all_metrics.append(metrics)

                    if metrics.mark is None:
                        net_mark_ready = False
                    else:
                        net_mark_total += metrics.mark * leg.signed_contracts * 100.0

                    if metrics.dte is not None:
                        dte_vals.append(metrics.dte)
                    if (
                        metrics.open_interest is not None
                        and metrics.open_interest < portfolio.risk_profile.min_open_interest
                    ):
                        low_oi = True
                    if metrics.volume is not None and metrics.volume < portfolio.risk_profile.min_volume:
                        low_vol = True
                    if metrics.execution_quality == "bad":
                        bad_spread = True
                    if metrics.quality_warnings:
                        quote_flags = True

                net_mark = net_mark_total if net_mark_ready else None
                if net_mark is None:
                    warnings.append("missing_leg_marks")
                if p.net_debit is None:
                    warnings.append("missing_net_debit")
                if low_oi:
                    warnings.append("low_open_interest_leg")
                if low_vol:
                    warnings.append("low_volume_leg")
                if bad_spread:
                    warnings.append("bad_spread_leg")
                if quote_flags:
                    warnings.append("quote_quality_leg")

                net_pnl_abs = net_pnl_pct = None
                if net_mark is not None and p.net_debit is not None:
                    net_pnl_abs = net_mark - p.net_debit
                    if p.net_debit > 0:
                        net_pnl_pct = net_pnl_abs / p.net_debit

                dte_min = min(dte_vals) if dte_vals else None
                dte_max = max(dte_vals) if dte_vals else None

                multi_leg_summaries.append(
                    MultiLegSummary(
                        position=p,
                        leg_metrics=leg_metrics,
                        net_mark=net_mark,
                        net_pnl_abs=net_pnl_abs,
                        net_pnl_pct=net_pnl_pct,
                        dte_min=dte_min,
                        dte_max=dte_max,
                        warnings=warnings,
                    )
                )
                continue

            snapshot_row = None
            if offline:
                snap_date = as_of_by_symbol.get(p.symbol)
                df_snap = snapshot_day_by_symbol.get(p.symbol, pd.DataFrame())
                row = None

                if snap_date is None:
                    offline_missing.append(f"{p.id}: missing offline as-of date for {p.symbol}")
                elif df_snap.empty:
                    offline_missing.append(
                        f"{p.id}: missing snapshot day data for {p.symbol} (as-of {snap_date.isoformat()})"
                    )
                else:
                    row = find_snapshot_row(
                        df_snap,
                        expiry=p.expiry,
                        strike=p.strike,
                        option_type=p.option_type,
                    )
                    if row is None:
                        offline_missing.append(
                            f"{p.id}: missing snapshot row for {p.symbol} {p.expiry.isoformat()} "
                            f"{p.option_type} {p.strike:g} (as-of {snap_date.isoformat()})"
                        )

                snapshot_row = row if row is not None else {}

            metrics = _position_metrics(
                provider,
                p,
                risk_profile=portfolio.risk_profile,
                underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                underlying_last_price=last_price_by_symbol.get(p.symbol),
                as_of=as_of_by_symbol.get(p.symbol),
                next_earnings_date=next_earnings_by_symbol.get(p.symbol),
                snapshot_row=snapshot_row,
            )
            single_metrics.append(metrics)
            all_metrics.append(metrics)
            advice_by_id[p.id] = advise(metrics, portfolio)
        except DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")

    if not single_metrics and not multi_leg_summaries:
        raise typer.Exit(1)

    if single_metrics:
        render_positions(console, portfolio, single_metrics, advice_by_id)
    if multi_leg_summaries:
        render_multi_leg_positions(console, multi_leg_summaries)

    exposure = compute_portfolio_exposure(all_metrics)
    _render_portfolio_risk(
        console,
        exposure,
        stress_spot_pct=stress_spot_pct,
        stress_vol_pp=stress_vol_pp,
        stress_days=stress_days,
    )

    if offline_missing:
        for msg in offline_missing:
            console.print(f"[yellow]Warning:[/yellow] {msg}")
        if offline_strict:
            raise typer.Exit(1)


@journal_app.command("log")
def journal_log(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    positions: bool = typer.Option(True, "--positions/--no-positions", help="Log position signals."),
    research: bool = typer.Option(False, "--research/--no-research", help="Log research recommendations."),
    scanner: bool = typer.Option(False, "--scanner/--no-scanner", help="Log scanner shortlist entries."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    offline: bool = typer.Option(
        True,
        "--offline/--online",
        help="Use offline snapshots + candle cache when available.",
    ),
    offline_strict: bool = typer.Option(
        False,
        "--offline-strict",
        help="Fail if any position is missing snapshot coverage (used with --offline).",
    ),
    snapshots_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--snapshots-dir",
        help="Directory containing options snapshot folders (used with --offline).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--cache-dir",
        help="Directory for locally cached daily candles.",
    ),
    journal_dir: Path = typer.Option(
        Path("data/journal"),
        "--journal-dir",
        help="Directory for journal signal events (writes signal_events.jsonl).",
    ),
    period: str = typer.Option("2y", "--period", help="Underlying history period (online only)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used for research).",
    ),
    research_watchlist: str = typer.Option(
        "watchlist",
        "--research-watchlist",
        help="Watchlist name to research (ignored when --research-symbol is provided).",
    ),
    research_symbol: str | None = typer.Option(None, "--research-symbol", help="Run research for a single symbol."),
    research_top: int = typer.Option(
        10,
        "--research-top",
        min=1,
        max=500,
        help="Max research symbols to log after ranking.",
    ),
    research_period: str = typer.Option(
        "5y",
        "--research-period",
        help="Daily candle period to ensure cached for research (yfinance period format).",
    ),
    research_window_pct: float = typer.Option(
        0.30,
        "--research-window-pct",
        min=0.0,
        max=2.0,
        help="Strike window around spot for research option selection.",
    ),
    research_short_min_dte: int = typer.Option(30, "--research-short-min-dte"),
    research_short_max_dte: int = typer.Option(90, "--research-short-max-dte"),
    research_long_min_dte: int = typer.Option(365, "--research-long-min-dte"),
    research_long_max_dte: int = typer.Option(1500, "--research-long-max-dte"),
    research_include_bad_quotes: bool = typer.Option(
        False,
        "--research-include-bad-quotes",
        help="Include research candidates with bad quote quality (best-effort).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used for IV percentile context).",
    ),
    scanner_run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--scanner-run-dir",
        help="Scanner runs directory (used to load shortlist.csv).",
    ),
    scanner_run_id: str | None = typer.Option(
        None,
        "--scanner-run-id",
        help="Specific scanner run id to load (defaults to latest).",
    ),
    scanner_top: int = typer.Option(
        20,
        "--scanner-top",
        min=1,
        max=500,
        help="Max scanner shortlist symbols to log.",
    ),
) -> None:
    """Append journal signal events (positions, research, scanner) best-effort."""
    _ensure_pandas()
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    if not positions and not research and not scanner:
        console.print("No journal contexts selected.")
        raise typer.Exit(0)

    if positions and not portfolio.positions:
        if research or scanner:
            console.print("[yellow]Warning:[/yellow] No positions found; skipping position logging.")
            positions = False
        else:
            console.print("No positions.")
            raise typer.Exit(0)

    provider: MarketDataProvider | None = None
    if (positions and not offline) or research:
        provider = build_provider()

    candle_store = build_candle_store(cache_dir, provider=provider)
    earnings_store = build_earnings_store(Path("data/earnings"))
    journal_store = build_journal_store(journal_dir)

    events: list[SignalEvent] = []
    counts = {"position": 0, "research": 0, "scanner": 0}
    offline_missing: list[str] = []

    if positions:
        snapshot_store: OptionsSnapshotStore | None = None
        if offline:
            snapshot_store = build_snapshot_store(snapshots_dir)

        history_by_symbol: dict[str, pd.DataFrame] = {}
        last_price_by_symbol: dict[str, float | None] = {}
        as_of_by_symbol: dict[str, date | None] = {}
        next_earnings_by_symbol: dict[str, date | None] = {}
        snapshot_day_by_symbol: dict[str, pd.DataFrame] = {}

        for sym in sorted({p.symbol for p in portfolio.positions}):
            history = pd.DataFrame()
            snapshot_date: date | None = None

            if offline:
                try:
                    history = candle_store.load(sym)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {sym}: {exc}")
                    history = pd.DataFrame()

                if snapshot_store is not None:
                    try:
                        snapshot_date = snapshot_store.resolve_date(sym, as_of)
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"[yellow]Warning:[/yellow] snapshot date resolve failed for {sym}: {exc}")
                        snapshot_date = None

                if snapshot_date is None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                    last_ts = history.index.max()
                    if last_ts is not None and not pd.isna(last_ts):
                        snapshot_date = last_ts.date()

                if snapshot_date is not None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                    history = history.loc[history.index <= pd.Timestamp(snapshot_date)]

                snap_df = pd.DataFrame()
                if snapshot_store is not None and snapshot_date is not None:
                    try:
                        snap_df = snapshot_store.load_day(sym, snapshot_date)
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"[yellow]Warning:[/yellow] snapshot load failed for {sym}: {exc}")
                        snap_df = pd.DataFrame()
                snapshot_day_by_symbol[sym] = snap_df
            else:
                try:
                    history = candle_store.get_daily_history(sym, period=period)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]Candle cache error:[/red] {sym}: {exc}")
                    history = pd.DataFrame()

            history_by_symbol[sym] = history

            if offline:
                as_of_by_symbol[sym] = snapshot_date
                last_price_by_symbol[sym] = (
                    close_asof(history, snapshot_date) if snapshot_date is not None else last_close(history)
                )
            else:
                last_price_by_symbol[sym] = last_close(history)
                as_of_date = None
                if not history.empty and isinstance(history.index, pd.DatetimeIndex):
                    last_ts = history.index.max()
                    if last_ts is not None and not pd.isna(last_ts):
                        as_of_date = last_ts.date()
                as_of_by_symbol[sym] = as_of_date

            next_earnings_by_symbol[sym] = safe_next_earnings_date(earnings_store, sym)

        for p in portfolio.positions:
            try:
                snapshot_row = None
                snapshot_date = None
                contract_symbol = None
                data_warnings: list[str] = []

                if offline:
                    snapshot_date = as_of_by_symbol.get(p.symbol)
                    df_snap = snapshot_day_by_symbol.get(p.symbol, pd.DataFrame())
                    row = None

                    if snapshot_date is None:
                        msg = f"{p.id}: missing offline as-of date for {p.symbol}"
                        offline_missing.append(msg)
                        data_warnings.append(msg)
                    elif df_snap.empty:
                        msg = (
                            f"{p.id}: missing snapshot day data for {p.symbol} "
                            f"(as-of {snapshot_date.isoformat()})"
                        )
                        offline_missing.append(msg)
                        data_warnings.append(msg)
                    else:
                        row = find_snapshot_row(
                            df_snap,
                            expiry=p.expiry,
                            strike=p.strike,
                            option_type=p.option_type,
                        )
                        if row is None:
                            msg = (
                                f"{p.id}: missing snapshot row for {p.symbol} {p.expiry.isoformat()} "
                                f"{p.option_type} {p.strike:g} (as-of {snapshot_date.isoformat()})"
                            )
                            offline_missing.append(msg)
                            data_warnings.append(msg)

                    snapshot_row = row if row is not None else {}
                    contract_symbol = _contract_symbol_from_row(row)

                metrics = _position_metrics(
                    None if offline else provider,
                    p,
                    risk_profile=portfolio.risk_profile,
                    underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                    underlying_last_price=last_price_by_symbol.get(p.symbol),
                    as_of=as_of_by_symbol.get(p.symbol),
                    next_earnings_date=next_earnings_by_symbol.get(p.symbol),
                    snapshot_row=snapshot_row,
                )
                advice = advise(metrics, portfolio)
                payload = _build_position_journal_payload(metrics, advice, data_warnings=data_warnings)

                event_date = metrics.as_of or date.today()
                events.append(
                    SignalEvent(
                        date=event_date,
                        symbol=p.symbol,
                        context=SignalContext.POSITION,
                        payload=payload,
                        snapshot_date=snapshot_date,
                        contract_symbol=contract_symbol,
                    )
                )
                counts["position"] += 1
            except DataFetchError as exc:
                console.print(f"[red]Data error:[/red] {exc}")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Unexpected error:[/red] {exc}")

    if research:
        rp = portfolio.risk_profile
        derived_store = build_derived_store(derived_dir)
        confluence_cfg = None
        technicals_cfg = None
        confluence_cfg_error = None
        technicals_cfg_error = None

        try:
            confluence_cfg = load_confluence_config()
        except ConfluenceConfigError as exc:
            confluence_cfg_error = str(exc)

        try:
            technicals_cfg = load_technical_backtesting_config()
        except TechnicalConfigError as exc:
            technicals_cfg_error = str(exc)

        if confluence_cfg_error:
            console.print(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")
        if technicals_cfg_error:
            console.print(f"[yellow]Warning:[/yellow] technicals config unavailable: {technicals_cfg_error}")

        if research_symbol:
            research_symbols = [research_symbol.strip().upper()]
        else:
            wl = load_watchlists(watchlists_path)
            research_symbols = wl.get(research_watchlist)
            if not research_symbols:
                console.print(
                    f"[yellow]Warning:[/yellow] research watchlist '{research_watchlist}' is empty or missing."
                )
                research_symbols = []

        cached_history: dict[str, pd.DataFrame] = {}
        cached_setup: dict[str, UnderlyingSetup] = {}
        cached_extension_pct: dict[str, float | None] = {}
        pre_confluence: dict[str, ConfluenceScore] = {}

        for sym in research_symbols:
            try:
                history = candle_store.get_daily_history(sym, period=research_period)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: candle cache error: {exc}")
                history = pd.DataFrame()
            cached_history[sym] = history

            setup = analyze_underlying(sym, history=history, risk_profile=rp)
            cached_setup[sym] = setup

            ext_pct = None
            if technicals_cfg is not None and history is not None and not history.empty:
                try:
                    ext_result = compute_current_extension_percentile(history, technicals_cfg)
                    ext_pct = ext_result.percentile
                except Exception:  # noqa: BLE001
                    ext_pct = None
            cached_extension_pct[sym] = ext_pct

            inputs = build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None)
            pre_confluence[sym] = score_confluence(inputs, cfg=confluence_cfg)

        if len(research_symbols) > 1:
            def _sort_key(sym: str) -> tuple[float, float, str]:
                score = pre_confluence.get(sym)
                coverage = score.coverage if score is not None else -1.0
                total = score.total if score is not None else -1.0
                return (-coverage, -total, sym)

            research_symbols = sorted(research_symbols, key=_sort_key)

        if research_top and research_symbols:
            research_symbols = research_symbols[:research_top]

        for sym in research_symbols:
            try:
                history = cached_history.get(sym, pd.DataFrame())
                setup = cached_setup.get(sym) or analyze_underlying(sym, history=history, risk_profile=rp)
                ext_pct = cached_extension_pct.get(sym)

                as_of_date = None
                if not history.empty and isinstance(history.index, pd.DatetimeIndex):
                    last_ts = history.index.max()
                    if last_ts is not None and not pd.isna(last_ts):
                        as_of_date = last_ts.date()

                next_earnings_date = safe_next_earnings_date(earnings_store, sym)
                levels = suggest_trade_levels(setup, history=history, risk_profile=rp)

                warnings: list[str] = []
                short_pick = None
                long_pick = None
                confluence_score = None
                vol_context = None

                if setup.spot is None:
                    warnings.append("no_spot_price")
                else:
                    expiry_strs = [d.isoformat() for d in provider.list_option_expiries(sym)] if provider else []
                    if not expiry_strs:
                        warnings.append("no_listed_expiries")
                    else:
                        expiry_as_of = as_of_date or date.today()
                        short_exp = choose_expiry(
                            expiry_strs,
                            min_dte=research_short_min_dte,
                            max_dte=research_short_max_dte,
                            target_dte=60,
                            today=expiry_as_of,
                        )
                        long_exp = choose_expiry(
                            expiry_strs,
                            min_dte=research_long_min_dte,
                            max_dte=research_long_max_dte,
                            target_dte=540,
                            today=expiry_as_of,
                        )
                        if long_exp is None:
                            parsed = []
                            for s in expiry_strs:
                                try:
                                    exp = date.fromisoformat(s)
                                except ValueError:
                                    continue
                                dte = (exp - expiry_as_of).days
                                parsed.append((dte, exp))
                            parsed = [t for t in parsed if t[0] >= research_long_min_dte]
                            if parsed:
                                _, long_exp = max(parsed, key=lambda t: t[0])

                        if setup.direction == Direction.NEUTRAL:
                            confluence_score = score_confluence(
                                build_confluence_inputs(
                                    setup,
                                    extension_percentile=ext_pct,
                                    vol_context=None,
                                ),
                                cfg=confluence_cfg,
                            )
                        else:
                            opt_type: OptionType = "call" if setup.direction == Direction.BULLISH else "put"
                            min_oi = rp.min_open_interest
                            min_vol = rp.min_volume
                            derived_history = derived_store.load(sym)

                            if short_exp is not None and provider is not None:
                                chain = provider.get_options_chain(sym, short_exp)
                                if vol_context is None:
                                    vol_context = compute_volatility_context(
                                        history=history,
                                        spot=setup.spot,
                                        calls=chain.calls,
                                        puts=chain.puts,
                                        derived_history=derived_history,
                                    )
                                df = chain.calls if opt_type == "call" else chain.puts
                                target_delta = 0.40 if opt_type == "call" else -0.40
                                short_pick = select_option_candidate(
                                    df,
                                    symbol=sym,
                                    option_type=opt_type,
                                    expiry=short_exp,
                                    spot=setup.spot,
                                    target_delta=target_delta,
                                    window_pct=research_window_pct,
                                    min_open_interest=min_oi,
                                    min_volume=min_vol,
                                    as_of=expiry_as_of,
                                    next_earnings_date=next_earnings_date,
                                    earnings_warn_days=rp.earnings_warn_days,
                                    earnings_avoid_days=rp.earnings_avoid_days,
                                    include_bad_quotes=research_include_bad_quotes,
                                )
                            else:
                                warnings.append("no_short_expiry")

                            if long_exp is not None and provider is not None:
                                chain = provider.get_options_chain(sym, long_exp)
                                if vol_context is None:
                                    vol_context = compute_volatility_context(
                                        history=history,
                                        spot=setup.spot,
                                        calls=chain.calls,
                                        puts=chain.puts,
                                        derived_history=derived_history,
                                    )
                                df = chain.calls if opt_type == "call" else chain.puts
                                target_delta = 0.70 if opt_type == "call" else -0.70
                                long_pick = select_option_candidate(
                                    df,
                                    symbol=sym,
                                    option_type=opt_type,
                                    expiry=long_exp,
                                    spot=setup.spot,
                                    target_delta=target_delta,
                                    window_pct=research_window_pct,
                                    min_open_interest=min_oi,
                                    min_volume=min_vol,
                                    as_of=expiry_as_of,
                                    next_earnings_date=next_earnings_date,
                                    earnings_warn_days=rp.earnings_warn_days,
                                    earnings_avoid_days=rp.earnings_avoid_days,
                                    include_bad_quotes=research_include_bad_quotes,
                                )
                            else:
                                warnings.append("no_long_expiry")

                            confluence_score = score_confluence(
                                build_confluence_inputs(
                                    setup,
                                    extension_percentile=ext_pct,
                                    vol_context=vol_context,
                                ),
                                cfg=confluence_cfg,
                            )

                if confluence_score is None:
                    confluence_score = pre_confluence.get(sym)

                payload = {
                    "as_of": _iso_date(as_of_date),
                    "setup": {
                        "direction": setup.direction.value,
                        "spot": _clean_float(setup.spot),
                        "reasons": list(setup.reasons),
                        "daily_rsi": _clean_float(setup.daily_rsi),
                        "daily_stoch_rsi": _clean_float(setup.daily_stoch_rsi),
                        "weekly_rsi": _clean_float(setup.weekly_rsi),
                        "weekly_breakout": setup.weekly_breakout,
                    },
                    "extension_percentile": _clean_float(ext_pct),
                    "levels": _levels_payload(levels),
                    "vol_context": _vol_context_payload(vol_context),
                    "confluence": _confluence_payload(confluence_score),
                    "short_candidate": _option_candidate_payload(short_pick),
                    "long_candidate": _option_candidate_payload(long_pick),
                    "warnings": warnings,
                }

                event_date = as_of_date or date.today()
                events.append(
                    SignalEvent(
                        date=event_date,
                        symbol=sym,
                        context=SignalContext.RESEARCH,
                        payload=payload,
                        snapshot_date=None,
                        contract_symbol=None,
                    )
                )
                counts["research"] += 1
            except DataFetchError as exc:
                console.print(f"[red]Research data error:[/red] {sym}: {exc}")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Research error:[/red] {sym}: {exc}")

    if scanner:
        run_root = _latest_scanner_run_dir(scanner_run_dir, run_id=scanner_run_id)
        if run_root is None:
            console.print("[yellow]Warning:[/yellow] No scanner runs found; skipping scanner logging.")
        else:
            rows = _read_scanner_shortlist(run_root / "shortlist.csv")
            if not rows:
                console.print("[yellow]Warning:[/yellow] Scanner shortlist.csv empty or missing.")
            else:
                if scanner_top and len(rows) > scanner_top:
                    rows = rows[:scanner_top]
                run_date = _scanner_run_date(run_root.name) or date.today()
                for idx, row in enumerate(rows, start=1):
                    sym = str(row.get("symbol") or "").strip().upper()
                    if not sym:
                        continue
                    payload = {
                        "rank": idx,
                        "score": row.get("score"),
                        "coverage": row.get("coverage"),
                        "top_reasons": row.get("top_reasons"),
                        "run_id": run_root.name,
                        "run_path": str(run_root),
                    }
                    events.append(
                        SignalEvent(
                            date=run_date,
                            symbol=sym,
                            context=SignalContext.SCANNER,
                            payload=payload,
                            snapshot_date=None,
                            contract_symbol=None,
                        )
                    )
                    counts["scanner"] += 1

    if not events:
        console.print("No journal events written.")
        raise typer.Exit(0)

    logged = journal_store.append_events(events)
    console.print(
        f"Logged {logged} signal(s) to {journal_store.path()} "
        f"(positions={counts['position']}, research={counts['research']}, scanner={counts['scanner']})"
    )

    if offline_missing:
        for msg in offline_missing:
            console.print(f"[yellow]Warning:[/yellow] {msg}")
        if offline_strict:
            raise typer.Exit(1)


@journal_app.command("evaluate")
def journal_evaluate(
    journal_dir: Path = typer.Option(
        Path("data/journal"),
        "--journal-dir",
        help="Directory containing signal_events.jsonl.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--cache-dir",
        help="Directory for cached daily candles (used for outcomes).",
    ),
    snapshots_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--snapshots-dir",
        help="Directory for options snapshot history (used for option mark outcomes).",
    ),
    horizons: str = typer.Option(
        "1,5,20",
        "--horizons",
        help="Comma-separated trading-day horizons (e.g. 1,5,20).",
    ),
    window: int = typer.Option(
        252,
        "--window",
        min=1,
        max=5000,
        help="Lookback window in calendar days.",
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="As-of date (YYYY-MM-DD). Defaults to today.",
    ),
    out_dir: Path = typer.Option(
        Path("data/reports/journal"),
        "--out-dir",
        help="Output directory for journal evaluation reports.",
    ),
    top: int = typer.Option(
        5,
        "--top",
        min=1,
        max=50,
        help="Top/bottom events to include in summaries.",
    ),
) -> None:
    """Evaluate journal outcomes across horizons (offline, deterministic)."""
    _ensure_pandas()
    console = Console()
    store = build_journal_store(journal_dir)
    result = store.read_events()
    if result.errors:
        console.print(f"[yellow]Warning:[/yellow] skipped {len(result.errors)} invalid journal lines.")

    events = result.events
    if not events:
        console.print("No journal events found.")
        raise typer.Exit(0)

    as_of_date = _parse_date(as_of) if as_of else date.today()
    start_date = as_of_date - timedelta(days=int(window))
    events = [e for e in events if start_date <= e.date <= as_of_date]
    if not events:
        console.print("No journal events within the window.")
        raise typer.Exit(0)

    horizon_vals: list[int] = []
    for part in horizons.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid horizon value: {part}") from exc
        if val <= 0:
            raise typer.BadParameter("Horizons must be positive integers.")
        horizon_vals.append(val)
    if not horizon_vals:
        raise typer.BadParameter("Provide at least one horizon.")

    symbols = {e.symbol for e in events if e.symbol}
    candle_store = build_candle_store(cache_dir)
    history_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            history_by_symbol[sym] = candle_store.load(sym)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {sym}: {exc}")
            history_by_symbol[sym] = pd.DataFrame()

    snapshot_store = build_snapshot_store(snapshots_dir)
    snapshot_cache: dict[tuple[str, date], pd.DataFrame] = {}

    def _snapshot_loader(symbol: str, snapshot_date: date) -> pd.DataFrame | None:
        key = (symbol.upper(), snapshot_date)
        if key in snapshot_cache:
            return snapshot_cache[key]
        try:
            df = snapshot_store.load_day(symbol, snapshot_date)
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()
        snapshot_cache[key] = df
        return df

    report = build_journal_report(
        events,
        history_by_symbol=history_by_symbol,
        horizons=horizon_vals,
        snapshot_loader=_snapshot_loader,
        top_n=top,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["as_of"] = as_of_date.isoformat()
    report["window_days"] = int(window)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{as_of_date.isoformat()}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = out_dir / f"{as_of_date.isoformat()}.md"
    md_path.write_text(render_journal_report_markdown(report), encoding="utf-8")

    console.print(f"Saved: {json_path}")
    console.print(f"Saved: {md_path}")


@app.command()
def watch(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    minutes: int = typer.Option(15, "--minutes", help="Polling interval in minutes."),
) -> None:
    """Continuously re-run analysis at a fixed interval."""
    if minutes <= 0:
        raise typer.BadParameter("--minutes must be > 0")

    console = Console()
    console.print(f"Watching {portfolio_path} every {minutes} minute(s). Ctrl+C to stop.")
    while True:
        try:
            analyze(portfolio_path)
        except typer.Exit:
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Watch error:[/red] {exc}")
        time.sleep(minutes * 60)


def _normalize_pct(val: float) -> float:
    try:
        val_f = float(val)
    except Exception:  # noqa: BLE001
        return 0.0
    return val_f / 100.0 if abs(val_f) > 1.0 else val_f


def _normalize_pp(val: float) -> float:
    try:
        val_f = float(val)
    except Exception:  # noqa: BLE001
        return 0.0
    return val_f / 100.0 if abs(val_f) > 1.0 else val_f


def _build_stress_scenarios(
    *,
    stress_spot_pct: list[float],
    stress_vol_pp: float,
    stress_days: int,
) -> list[StressScenario]:
    scenarios: list[StressScenario] = []
    spot_values = stress_spot_pct or [0.05]
    seen_spot: set[float] = set()
    for raw in spot_values:
        pct = abs(_normalize_pct(raw))
        if pct <= 0 or pct in seen_spot:
            continue
        seen_spot.add(pct)
        scenarios.append(StressScenario(name=f"Spot {pct:+.0%}", spot_pct=pct))
        scenarios.append(StressScenario(name=f"Spot {-pct:+.0%}", spot_pct=-pct))

    vol_pp = _normalize_pp(stress_vol_pp)
    if vol_pp != 0:
        pp_label = vol_pp * 100.0
        scenarios.append(StressScenario(name=f"IV {pp_label:+.1f}pp", vol_pp=vol_pp))
        scenarios.append(StressScenario(name=f"IV {-pp_label:+.1f}pp", vol_pp=-vol_pp))

    if stress_days > 0:
        scenarios.append(StressScenario(name=f"Time +{stress_days}d", days=stress_days))

    return scenarios


def _render_portfolio_risk(
    console: Console,
    exposure: PortfolioExposure,
    *,
    stress_spot_pct: list[float],
    stress_vol_pp: float,
    stress_days: int,
) -> None:
    from rich.table import Table

    def _fmt_num(val: float | None, *, digits: int = 2) -> str:
        if val is None:
            return "-"
        return f"{val:,.{digits}f}"

    def _fmt_money(val: float | None) -> str:
        if val is None:
            return "-"
        return f"${val:,.2f}"

    def _fmt_pct(val: float | None) -> str:
        if val is None:
            return "-"
        return f"{val:.1%}"

    table = Table(title="Portfolio Greeks (best-effort)")
    table.add_column("As-of")
    table.add_column("Delta (shares)", justify="right")
    table.add_column("Theta/day ($)", justify="right")
    table.add_column("Vega ($/IV)", justify="right")
    table.add_row(
        "-" if exposure.as_of is None else exposure.as_of.isoformat(),
        _fmt_num(exposure.total_delta_shares),
        _fmt_money(exposure.total_theta_dollars_per_day),
        _fmt_money(exposure.total_vega_dollars_per_iv),
    )
    console.print(table)

    if exposure.assumptions:
        console.print("Assumptions: " + "; ".join(exposure.assumptions))
    if exposure.warnings:
        console.print("[yellow]Warnings:[/yellow] " + "; ".join(exposure.warnings))

    scenarios = _build_stress_scenarios(
        stress_spot_pct=stress_spot_pct,
        stress_vol_pp=stress_vol_pp,
        stress_days=stress_days,
    )
    if not scenarios:
        return

    stress_results = run_stress(exposure, scenarios)
    stress_table = Table(title="Portfolio Stress (best-effort)")
    stress_table.add_column("Scenario")
    stress_table.add_column("PnL $", justify="right")
    stress_table.add_column("PnL %", justify="right")
    stress_table.add_column("Notes")

    for result in stress_results:
        notes = ", ".join(result.warnings) if result.warnings else "-"
        stress_table.add_row(
            result.name,
            _fmt_money(result.pnl),
            _fmt_pct(result.pnl_pct),
            notes,
        )
    console.print(stress_table)
