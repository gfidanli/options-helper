from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
import options_helper.commands.journal_eval_common as journal_eval_common
from options_helper.pipelines.journal_log_research_runtime import _collect_research_events as _collect_research_events_runtime
from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.confluence import ConfluenceScore, score_confluence
from options_helper.analysis.extension_scan import compute_current_extension_percentile
from options_helper.analysis.journal_eval import build_journal_report, render_journal_report_markdown
from options_helper.analysis.research import (
    OptionCandidate,
    TradeLevels,
    UnderlyingSetup,
    VolatilityContext,
    analyze_underlying,
    build_confluence_inputs,
)
from options_helper.commands.position_metrics import _position_metrics
from options_helper.data.candles import close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.earnings import safe_next_earnings_date
from options_helper.data.journal import SignalContext, SignalEvent
from options_helper.data.market_types import DataFetchError
from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row
from options_helper.data.providers.base import MarketDataProvider
from options_helper.data.technical_backtesting_config import (
    ConfigError as TechnicalConfigError,
    load_technical_backtesting_config,
)
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    import pandas as pd


app = typer.Typer(help="Signal journal + outcome tracking.")

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


_JOURNAL_LOG_PORTFOLIO_PATH_ARG = typer.Argument(..., help="Path to portfolio JSON.")
_JOURNAL_LOG_POSITIONS_OPT = typer.Option(True, "--positions/--no-positions", help="Log position signals.")
_JOURNAL_LOG_RESEARCH_OPT = typer.Option(False, "--research/--no-research", help="Log research recommendations.")
_JOURNAL_LOG_SCANNER_OPT = typer.Option(False, "--scanner/--no-scanner", help="Log scanner shortlist entries.")
_JOURNAL_LOG_AS_OF_OPT = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'.")
_JOURNAL_LOG_OFFLINE_OPT = typer.Option(
    True,
    "--offline/--online",
    help="Use offline snapshots + candle cache when available.",
)
_JOURNAL_LOG_OFFLINE_STRICT_OPT = typer.Option(
    False,
    "--offline-strict",
    help="Fail if any position is missing snapshot coverage (used with --offline).",
)
_JOURNAL_LOG_SNAPSHOTS_DIR_OPT = typer.Option(
    Path("data/options_snapshots"),
    "--snapshots-dir",
    help="Directory containing options snapshot folders (used with --offline).",
)
_JOURNAL_LOG_CACHE_DIR_OPT = typer.Option(
    Path("data/candles"),
    "--cache-dir",
    help="Directory for locally cached daily candles.",
)
_JOURNAL_LOG_JOURNAL_DIR_OPT = typer.Option(
    Path("data/journal"),
    "--journal-dir",
    help="Directory for journal signal events (writes signal_events.jsonl).",
)
_JOURNAL_LOG_PERIOD_OPT = typer.Option("2y", "--period", help="Underlying history period (online only).")
_JOURNAL_LOG_WATCHLISTS_PATH_OPT = typer.Option(
    Path("data/watchlists.json"),
    "--watchlists-path",
    help="Path to watchlists JSON store (used for research).",
)
_JOURNAL_LOG_RESEARCH_WATCHLIST_OPT = typer.Option(
    "watchlist",
    "--research-watchlist",
    help="Watchlist name to research (ignored when --research-symbol is provided).",
)
_JOURNAL_LOG_RESEARCH_SYMBOL_OPT = typer.Option(None, "--research-symbol", help="Run research for a single symbol.")
_JOURNAL_LOG_RESEARCH_TOP_OPT = typer.Option(10, "--research-top", min=1, max=500, help="Max research symbols to log after ranking.")
_JOURNAL_LOG_RESEARCH_PERIOD_OPT = typer.Option(
    "5y",
    "--research-period",
    help="Daily candle period to ensure cached for research (yfinance period format).",
)
_JOURNAL_LOG_RESEARCH_WINDOW_PCT_OPT = typer.Option(
    0.30,
    "--research-window-pct",
    min=0.0,
    max=2.0,
    help="Strike window around spot for research option selection.",
)
_JOURNAL_LOG_RESEARCH_SHORT_MIN_DTE_OPT = typer.Option(30, "--research-short-min-dte")
_JOURNAL_LOG_RESEARCH_SHORT_MAX_DTE_OPT = typer.Option(90, "--research-short-max-dte")
_JOURNAL_LOG_RESEARCH_LONG_MIN_DTE_OPT = typer.Option(365, "--research-long-min-dte")
_JOURNAL_LOG_RESEARCH_LONG_MAX_DTE_OPT = typer.Option(1500, "--research-long-max-dte")
_JOURNAL_LOG_RESEARCH_INCLUDE_BAD_QUOTES_OPT = typer.Option(
    False,
    "--research-include-bad-quotes",
    help="Include research candidates with bad quote quality (best-effort).",
)
_JOURNAL_LOG_DERIVED_DIR_OPT = typer.Option(
    Path("data/derived"),
    "--derived-dir",
    help="Directory for derived metric files (used for IV percentile context).",
)
_JOURNAL_LOG_SCANNER_RUN_DIR_OPT = typer.Option(
    Path("data/scanner/runs"),
    "--scanner-run-dir",
    help="Scanner runs directory (used to load shortlist.csv).",
)
_JOURNAL_LOG_SCANNER_RUN_ID_OPT = typer.Option(
    None,
    "--scanner-run-id",
    help="Specific scanner run id to load (defaults to latest).",
)
_JOURNAL_LOG_SCANNER_TOP_OPT = typer.Option(20, "--scanner-top", min=1, max=500, help="Max scanner shortlist symbols to log.")


@dataclass(frozen=True)
class _JournalLogArgs:
    portfolio_path: Path
    positions: bool
    research: bool
    scanner: bool
    as_of: str
    offline: bool
    offline_strict: bool
    snapshots_dir: Path
    cache_dir: Path
    journal_dir: Path
    period: str
    watchlists_path: Path
    research_watchlist: str
    research_symbol: str | None
    research_top: int
    research_period: str
    research_window_pct: float
    research_short_min_dte: int
    research_short_max_dte: int
    research_long_min_dte: int
    research_long_max_dte: int
    research_include_bad_quotes: bool
    derived_dir: Path
    scanner_run_dir: Path
    scanner_run_id: str | None
    scanner_top: int


@dataclass
class _JournalLogRuntime:
    portfolio: object
    console: Console
    provider: MarketDataProvider | None
    candle_store: object
    earnings_store: object
    journal_store: object
    events: list[SignalEvent] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {"position": 0, "research": 0, "scanner": 0})
    offline_missing: list[str] = field(default_factory=list)


@dataclass
class _PositionSymbolData:
    history_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    last_price_by_symbol: dict[str, float | None] = field(default_factory=dict)
    as_of_by_symbol: dict[str, date | None] = field(default_factory=dict)
    next_earnings_by_symbol: dict[str, date | None] = field(default_factory=dict)
    snapshot_day_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class _ResearchCache:
    history_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    setup_by_symbol: dict[str, UnderlyingSetup] = field(default_factory=dict)
    extension_pct_by_symbol: dict[str, float | None] = field(default_factory=dict)
    pre_confluence_by_symbol: dict[str, ConfluenceScore] = field(default_factory=dict)


def _history_as_of_date(history: pd.DataFrame) -> date | None:
    if history.empty or not isinstance(history.index, pd.DatetimeIndex):
        return None
    last_ts = history.index.max()
    if last_ts is None or pd.isna(last_ts):
        return None
    return last_ts.date()


def _validate_journal_contexts(
    portfolio: object,
    *,
    console: Console,
    positions: bool,
    research: bool,
    scanner: bool,
) -> bool:
    if not positions and not research and not scanner:
        console.print("No journal contexts selected.")
        raise typer.Exit(0)
    if positions and not portfolio.positions:
        if research or scanner:
            console.print("[yellow]Warning:[/yellow] No positions found; skipping position logging.")
            return False
        console.print("No positions.")
        raise typer.Exit(0)
    return positions


def _build_journal_runtime(
    args: _JournalLogArgs,
    *,
    portfolio: object,
    console: Console,
    positions_enabled: bool,
) -> _JournalLogRuntime:
    provider: MarketDataProvider | None = None
    if (positions_enabled and not args.offline) or args.research:
        provider = cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(args.cache_dir, provider=provider)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    journal_store = cli_deps.build_journal_store(args.journal_dir)
    return _JournalLogRuntime(
        portfolio=portfolio,
        console=console,
        provider=provider,
        candle_store=candle_store,
        earnings_store=earnings_store,
        journal_store=journal_store,
    )


def _resolve_snapshot_date(
    symbol: str,
    *,
    as_of: str,
    history: pd.DataFrame,
    snapshot_store: OptionsSnapshotStore | None,
    console: Console,
) -> date | None:
    snapshot_date = None
    if snapshot_store is not None:
        try:
            snapshot_date = snapshot_store.resolve_date(symbol, as_of)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] snapshot date resolve failed for {symbol}: {exc}")
    if snapshot_date is None:
        snapshot_date = _history_as_of_date(history)
    return snapshot_date


def _load_snapshot_day(
    symbol: str,
    *,
    snapshot_date: date | None,
    snapshot_store: OptionsSnapshotStore | None,
    console: Console,
) -> pd.DataFrame:
    if snapshot_store is None or snapshot_date is None:
        return pd.DataFrame()
    try:
        return snapshot_store.load_day(symbol, snapshot_date)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Warning:[/yellow] snapshot load failed for {symbol}: {exc}")
        return pd.DataFrame()


def _prepare_position_symbol_data(
    runtime: _JournalLogRuntime,
    *,
    args: _JournalLogArgs,
    snapshot_store: OptionsSnapshotStore | None,
) -> _PositionSymbolData:
    symbol_data = _PositionSymbolData()
    symbols = sorted({pos.symbol for pos in runtime.portfolio.positions})
    for symbol in symbols:
        history = pd.DataFrame()
        snapshot_date = None
        if args.offline:
            try:
                history = runtime.candle_store.load(symbol)
            except Exception as exc:  # noqa: BLE001
                runtime.console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {symbol}: {exc}")
            snapshot_date = _resolve_snapshot_date(
                symbol,
                as_of=args.as_of,
                history=history,
                snapshot_store=snapshot_store,
                console=runtime.console,
            )
            if snapshot_date is not None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                history = history.loc[history.index <= pd.Timestamp(snapshot_date)]
            symbol_data.snapshot_day_by_symbol[symbol] = _load_snapshot_day(
                symbol,
                snapshot_date=snapshot_date,
                snapshot_store=snapshot_store,
                console=runtime.console,
            )
        else:
            try:
                history = runtime.candle_store.get_daily_history(symbol, period=args.period)
            except Exception as exc:  # noqa: BLE001
                runtime.console.print(f"[red]Candle cache error:[/red] {symbol}: {exc}")
        symbol_data.history_by_symbol[symbol] = history
        symbol_data.as_of_by_symbol[symbol] = snapshot_date if args.offline else _history_as_of_date(history)
        if args.offline and snapshot_date is not None:
            symbol_data.last_price_by_symbol[symbol] = close_asof(history, snapshot_date)
        else:
            symbol_data.last_price_by_symbol[symbol] = last_close(history)
        symbol_data.next_earnings_by_symbol[symbol] = safe_next_earnings_date(runtime.earnings_store, symbol)
    return symbol_data


def _offline_snapshot_row(
    position: object,
    *,
    symbol_data: _PositionSymbolData,
    offline_missing: list[str],
) -> tuple[dict, date | None, str | None, list[str]]:
    snapshot_date = symbol_data.as_of_by_symbol.get(position.symbol)
    snapshot_day = symbol_data.snapshot_day_by_symbol.get(position.symbol, pd.DataFrame())
    row = None
    data_warnings: list[str] = []
    if snapshot_date is None:
        msg = f"{position.id}: missing offline as-of date for {position.symbol}"
        offline_missing.append(msg)
        data_warnings.append(msg)
    elif snapshot_day.empty:
        msg = f"{position.id}: missing snapshot day data for {position.symbol} (as-of {snapshot_date.isoformat()})"
        offline_missing.append(msg)
        data_warnings.append(msg)
    else:
        row = find_snapshot_row(
            snapshot_day,
            expiry=position.expiry,
            strike=position.strike,
            option_type=position.option_type,
        )
        if row is None:
            msg = (
                f"{position.id}: missing snapshot row for {position.symbol} {position.expiry.isoformat()} "
                f"{position.option_type} {position.strike:g} (as-of {snapshot_date.isoformat()})"
            )
            offline_missing.append(msg)
            data_warnings.append(msg)
    snapshot_row = row if row is not None else {}
    return snapshot_row, snapshot_date, _contract_symbol_from_row(row), data_warnings


def _append_position_event(
    runtime: _JournalLogRuntime,
    *,
    args: _JournalLogArgs,
    position: object,
    symbol_data: _PositionSymbolData,
) -> None:
    try:
        snapshot_row = None
        snapshot_date = None
        contract_symbol = None
        data_warnings: list[str] = []
        if args.offline:
            snapshot_row, snapshot_date, contract_symbol, data_warnings = _offline_snapshot_row(
                position,
                symbol_data=symbol_data,
                offline_missing=runtime.offline_missing,
            )
        metrics = _position_metrics(
            None if args.offline else runtime.provider,
            position,
            risk_profile=runtime.portfolio.risk_profile,
            underlying_history=symbol_data.history_by_symbol.get(position.symbol, pd.DataFrame()),
            underlying_last_price=symbol_data.last_price_by_symbol.get(position.symbol),
            as_of=symbol_data.as_of_by_symbol.get(position.symbol),
            next_earnings_date=symbol_data.next_earnings_by_symbol.get(position.symbol),
            snapshot_row=snapshot_row,
        )
        advice = advise(metrics, runtime.portfolio)
        payload = _build_position_journal_payload(metrics, advice, data_warnings=data_warnings)
        runtime.events.append(
            SignalEvent(
                date=metrics.as_of or date.today(),
                symbol=position.symbol,
                context=SignalContext.POSITION,
                payload=payload,
                snapshot_date=snapshot_date,
                contract_symbol=contract_symbol,
            )
        )
        runtime.counts["position"] += 1
    except DataFetchError as exc:
        runtime.console.print(f"[red]Data error:[/red] {exc}")
    except Exception as exc:  # noqa: BLE001
        runtime.console.print(f"[red]Unexpected error:[/red] {exc}")


def _collect_position_events(runtime: _JournalLogRuntime, *, args: _JournalLogArgs) -> None:
    snapshot_store = cli_deps.build_snapshot_store(args.snapshots_dir) if args.offline else None
    symbol_data = _prepare_position_symbol_data(runtime, args=args, snapshot_store=snapshot_store)
    for position in runtime.portfolio.positions:
        _append_position_event(runtime, args=args, position=position, symbol_data=symbol_data)


def _load_research_configs(console: Console) -> tuple[object | None, object | None]:
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
    return confluence_cfg, technicals_cfg


def _resolve_research_symbols(args: _JournalLogArgs, *, console: Console) -> list[str]:
    if args.research_symbol:
        return [args.research_symbol.strip().upper()]
    watchlists = load_watchlists(args.watchlists_path)
    symbols = watchlists.get(args.research_watchlist)
    if symbols:
        return symbols
    console.print(f"[yellow]Warning:[/yellow] research watchlist '{args.research_watchlist}' is empty or missing.")
    return []


def _build_research_cache(
    *,
    symbols: list[str],
    candle_store: object,
    period: str,
    risk_profile: object,
    technicals_cfg: object,
    confluence_cfg: object,
    console: Console,
) -> _ResearchCache:
    cache = _ResearchCache()
    for symbol in symbols:
        try:
            history = candle_store.get_daily_history(symbol, period=period)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] {symbol}: candle cache error: {exc}")
            history = pd.DataFrame()
        cache.history_by_symbol[symbol] = history
        setup = analyze_underlying(symbol, history=history, risk_profile=risk_profile)
        cache.setup_by_symbol[symbol] = setup
        ext_pct = None
        if technicals_cfg is not None and history is not None and not history.empty:
            try:
                ext_result = compute_current_extension_percentile(history, technicals_cfg)
                ext_pct = ext_result.percentile
            except Exception:  # noqa: BLE001
                ext_pct = None
        cache.extension_pct_by_symbol[symbol] = ext_pct
        inputs = build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None)
        cache.pre_confluence_by_symbol[symbol] = score_confluence(inputs, cfg=confluence_cfg)
    return cache


def _rank_research_symbols(
    symbols: list[str],
    pre_confluence: dict[str, ConfluenceScore],
    research_top: int,
) -> list[str]:
    ranked = list(symbols)
    if len(ranked) > 1:
        def _sort_key(symbol: str) -> tuple[float, float, str]:
            score = pre_confluence.get(symbol)
            coverage = score.coverage if score is not None else -1.0
            total = score.total if score is not None else -1.0
            return (-coverage, -total, symbol)

        ranked = sorted(ranked, key=_sort_key)
    if research_top and ranked:
        ranked = ranked[:research_top]
    return ranked


def _collect_scanner_events(runtime: _JournalLogRuntime, *, args: _JournalLogArgs) -> None:
    run_root = _latest_scanner_run_dir(args.scanner_run_dir, run_id=args.scanner_run_id)
    if run_root is None:
        runtime.console.print("[yellow]Warning:[/yellow] No scanner runs found; skipping scanner logging.")
        return
    rows = _read_scanner_shortlist(run_root / "shortlist.csv")
    if not rows:
        runtime.console.print("[yellow]Warning:[/yellow] Scanner shortlist.csv empty or missing.")
        return
    if args.scanner_top and len(rows) > args.scanner_top:
        rows = rows[:args.scanner_top]
    run_date = _scanner_run_date(run_root.name) or date.today()
    for rank, row in enumerate(rows, start=1):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        payload = {
            "rank": rank,
            "score": row.get("score"),
            "coverage": row.get("coverage"),
            "top_reasons": row.get("top_reasons"),
            "run_id": run_root.name,
            "run_path": str(run_root),
        }
        runtime.events.append(
            SignalEvent(
                date=run_date,
                symbol=symbol,
                context=SignalContext.SCANNER,
                payload=payload,
                snapshot_date=None,
                contract_symbol=None,
            )
        )
        runtime.counts["scanner"] += 1


def _finalize_journal_log(runtime: _JournalLogRuntime, *, offline_strict: bool) -> None:
    if not runtime.events:
        runtime.console.print("No journal events written.")
        raise typer.Exit(0)
    logged = runtime.journal_store.append_events(runtime.events)
    runtime.console.print(
        f"Logged {logged} signal(s) to {runtime.journal_store.path()} "
        f"(positions={runtime.counts['position']}, research={runtime.counts['research']}, scanner={runtime.counts['scanner']})"
    )
    if runtime.offline_missing:
        for msg in runtime.offline_missing:
            runtime.console.print(f"[yellow]Warning:[/yellow] {msg}")
        if offline_strict:
            raise typer.Exit(1)


def _run_journal_log(args: _JournalLogArgs) -> None:
    _ensure_pandas()
    portfolio = load_portfolio(args.portfolio_path)
    console = Console()
    positions_enabled = _validate_journal_contexts(
        portfolio,
        console=console,
        positions=args.positions,
        research=args.research,
        scanner=args.scanner,
    )
    runtime = _build_journal_runtime(args, portfolio=portfolio, console=console, positions_enabled=positions_enabled)
    if positions_enabled:
        _collect_position_events(runtime, args=args)
    if args.research:
        _collect_research_events_runtime(
            runtime,
            args=args,
            load_research_configs_fn=_load_research_configs,
            resolve_research_symbols_fn=_resolve_research_symbols,
            build_research_cache_fn=_build_research_cache,
            rank_research_symbols_fn=_rank_research_symbols,
            history_as_of_date_fn=_history_as_of_date,
            iso_date_fn=_iso_date,
            clean_float_fn=_clean_float,
            levels_payload_fn=_levels_payload,
            vol_context_payload_fn=_vol_context_payload,
            confluence_payload_fn=_confluence_payload,
            option_candidate_payload_fn=_option_candidate_payload,
            build_derived_store_fn=cli_deps.build_derived_store,
        )
    if args.scanner:
        _collect_scanner_events(runtime, args=args)
    _finalize_journal_log(runtime, offline_strict=args.offline_strict)


@app.command("log")
def journal_log(
    portfolio_path: Path = _JOURNAL_LOG_PORTFOLIO_PATH_ARG,
    positions: bool = _JOURNAL_LOG_POSITIONS_OPT,
    research: bool = _JOURNAL_LOG_RESEARCH_OPT,
    scanner: bool = _JOURNAL_LOG_SCANNER_OPT,
    as_of: str = _JOURNAL_LOG_AS_OF_OPT,
    offline: bool = _JOURNAL_LOG_OFFLINE_OPT,
    offline_strict: bool = _JOURNAL_LOG_OFFLINE_STRICT_OPT,
    snapshots_dir: Path = _JOURNAL_LOG_SNAPSHOTS_DIR_OPT,
    cache_dir: Path = _JOURNAL_LOG_CACHE_DIR_OPT,
    journal_dir: Path = _JOURNAL_LOG_JOURNAL_DIR_OPT,
    period: str = _JOURNAL_LOG_PERIOD_OPT,
    watchlists_path: Path = _JOURNAL_LOG_WATCHLISTS_PATH_OPT,
    research_watchlist: str = _JOURNAL_LOG_RESEARCH_WATCHLIST_OPT,
    research_symbol: str | None = _JOURNAL_LOG_RESEARCH_SYMBOL_OPT,
    research_top: int = _JOURNAL_LOG_RESEARCH_TOP_OPT,
    research_period: str = _JOURNAL_LOG_RESEARCH_PERIOD_OPT,
    research_window_pct: float = _JOURNAL_LOG_RESEARCH_WINDOW_PCT_OPT,
    research_short_min_dte: int = _JOURNAL_LOG_RESEARCH_SHORT_MIN_DTE_OPT,
    research_short_max_dte: int = _JOURNAL_LOG_RESEARCH_SHORT_MAX_DTE_OPT,
    research_long_min_dte: int = _JOURNAL_LOG_RESEARCH_LONG_MIN_DTE_OPT,
    research_long_max_dte: int = _JOURNAL_LOG_RESEARCH_LONG_MAX_DTE_OPT,
    research_include_bad_quotes: bool = _JOURNAL_LOG_RESEARCH_INCLUDE_BAD_QUOTES_OPT,
    derived_dir: Path = _JOURNAL_LOG_DERIVED_DIR_OPT,
    scanner_run_dir: Path = _JOURNAL_LOG_SCANNER_RUN_DIR_OPT,
    scanner_run_id: str | None = _JOURNAL_LOG_SCANNER_RUN_ID_OPT,
    scanner_top: int = _JOURNAL_LOG_SCANNER_TOP_OPT,
) -> None:
    """Append journal signal events (positions, research, scanner) best-effort."""
    _run_journal_log(_JournalLogArgs(**locals()))


@app.command("evaluate")
def journal_evaluate(
    journal_dir: Path = journal_eval_common._JOURNAL_EVALUATE_JOURNAL_DIR_OPT,
    cache_dir: Path = journal_eval_common._JOURNAL_EVALUATE_CACHE_DIR_OPT,
    snapshots_dir: Path = journal_eval_common._JOURNAL_EVALUATE_SNAPSHOTS_DIR_OPT,
    horizons: str = journal_eval_common._JOURNAL_EVALUATE_HORIZONS_OPT,
    window: int = journal_eval_common._JOURNAL_EVALUATE_WINDOW_OPT,
    as_of: str | None = journal_eval_common._JOURNAL_EVALUATE_AS_OF_OPT,
    out_dir: Path = journal_eval_common._JOURNAL_EVALUATE_OUT_DIR_OPT,
    top: int = journal_eval_common._JOURNAL_EVALUATE_TOP_OPT,
) -> None:
    """Evaluate journal outcomes across horizons (offline, deterministic)."""
    _ensure_pandas()
    console = Console()
    store = cli_deps.build_journal_store(journal_dir)
    events, as_of_date = journal_eval_common.load_filtered_events(
        store=store,
        console=console,
        as_of=as_of,
        window=window,
        parse_date=_parse_date,
    )
    horizon_vals = journal_eval_common.parse_horizon_values(horizons)
    symbols = {e.symbol for e in events if e.symbol}
    candle_store = cli_deps.build_candle_store(cache_dir)
    history_by_symbol = journal_eval_common.load_history_by_symbol(
        symbols=symbols,
        candle_store=candle_store,
        console=console,
    )
    snapshot_store = cli_deps.build_snapshot_store(snapshots_dir)
    snapshot_loader = journal_eval_common.make_snapshot_loader(snapshot_store)
    report = build_journal_report(
        events,
        history_by_symbol=history_by_symbol,
        horizons=horizon_vals,
        snapshot_loader=snapshot_loader,
        top_n=top,
    )
    journal_eval_common.persist_report(
        report=report,
        as_of_date=as_of_date,
        window=window,
        out_dir=out_dir,
        console=console,
        render_markdown=render_journal_report_markdown,
    )
