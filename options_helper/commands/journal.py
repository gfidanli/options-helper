from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.confluence import ConfluenceScore, score_confluence
from options_helper.analysis.extension_scan import compute_current_extension_percentile
from options_helper.analysis.journal_eval import build_journal_report, render_journal_report_markdown
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
from options_helper.models import OptionType
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


@app.command("log")
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
        provider = cli_deps.build_provider()

    candle_store = cli_deps.build_candle_store(cache_dir, provider=provider)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    journal_store = cli_deps.build_journal_store(journal_dir)

    events: list[SignalEvent] = []
    counts = {"position": 0, "research": 0, "scanner": 0}
    offline_missing: list[str] = []

    if positions:
        snapshot_store: OptionsSnapshotStore | None = None
        if offline:
            snapshot_store = cli_deps.build_snapshot_store(snapshots_dir)

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
        derived_store = cli_deps.build_derived_store(derived_dir)
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


@app.command("evaluate")
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
    store = cli_deps.build_journal_store(journal_dir)
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
    candle_store = cli_deps.build_candle_store(cache_dir)
    history_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            history_by_symbol[sym] = candle_store.load(sym)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {sym}: {exc}")
            history_by_symbol[sym] = pd.DataFrame()

    snapshot_store = cli_deps.build_snapshot_store(snapshots_dir)
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
