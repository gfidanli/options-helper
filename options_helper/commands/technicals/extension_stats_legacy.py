from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from options_helper.commands.technicals.extension_stats_enrichment_legacy import (
    enrich_extension_payload,
)
from options_helper.commands.technicals.extension_stats_markdown_legacy import (
    build_extension_stats_markdown,
)
from options_helper.commands.technicals.extension_stats_runtime_legacy import (
    build_extension_stats_runtime,
)
from options_helper.commands.technicals_common import setup_technicals_logging
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config


def _load_ohlc_df(
    *,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
):
    from options_helper.data.candles import CandleCacheError
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache, load_ohlc_from_path

    if ohlc_path:
        return load_ohlc_from_path(ohlc_path)
    if symbol:
        try:
            return load_ohlc_from_cache(
                symbol,
                cache_dir,
                backfill_if_missing=True,
                period="max",
                raise_on_backfill_error=True,
            )
        except CandleCacheError as exc:
            raise typer.BadParameter(f"Failed to backfill OHLC for {symbol}: {exc}") from exc
    raise typer.BadParameter("Provide --ohlc-path or --symbol/--cache-dir")


def _write_extension_artifacts(
    *,
    out: Path | None,
    sym_label: str,
    asof: str,
    payload: dict[str, object],
    md: str,
    write_json: bool,
    write_md: bool,
    console: Console,
) -> None:
    if out is None:
        return
    base = out / sym_label
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / f"{asof}.json"
    md_path = base / f"{asof}.md"
    if write_json:
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        console.print(f"Wrote JSON: {json_path}")
    if write_md:
        md_path.write_text(md, encoding="utf-8")
        console.print(f"Wrote Markdown: {md_path}")


def _print_markdown(*, md: str, console: Console) -> None:
    try:
        from rich.markdown import Markdown

        console.print(Markdown(md))
    except Exception:  # noqa: BLE001
        console.print(md)


def _build_runtime(
    *,
    df,
    cfg,
    symbol: str | None,
    tail_pct: float | None,
    percentile_window_years: int | None,
    divergence_window_days: int,
    divergence_min_extension_days: int,
    divergence_min_extension_percentile: float | None,
    divergence_max_extension_percentile: float | None,
    divergence_min_price_delta_pct: float,
    divergence_min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
):
    return build_extension_stats_runtime(
        df=df,
        cfg=cfg,
        symbol=symbol,
        tail_pct=tail_pct,
        percentile_window_years=percentile_window_years,
        divergence_window_days=divergence_window_days,
        divergence_min_extension_days=divergence_min_extension_days,
        divergence_min_extension_percentile=divergence_min_extension_percentile,
        divergence_max_extension_percentile=divergence_max_extension_percentile,
        divergence_min_price_delta_pct=divergence_min_price_delta_pct,
        divergence_min_rsi_delta=divergence_min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )


def _render_markdown(
    *,
    runtime,
    payload: dict[str, object],
    daily_tail_events: list[dict[str, object]],
    weekly_tail_events: list[dict[str, object]],
    rsi_overbought: float,
    rsi_oversold: float,
) -> str:
    from options_helper.technicals_backtesting.rsi_divergence import rsi_regime_tag

    return build_extension_stats_markdown(
        sym_label=runtime.sym_label,
        report_daily=runtime.report_daily,
        report_weekly=runtime.report_weekly,
        payload=payload,
        rsi_divergence_cfg=runtime.rsi_divergence_cfg,
        rsi_divergence_daily=runtime.rsi_divergence_daily,
        rsi_divergence_weekly=runtime.rsi_divergence_weekly,
        daily_tail_events=daily_tail_events,
        weekly_tail_events=weekly_tail_events,
        forward_days_daily=runtime.forward_days_daily,
        forward_days_weekly=runtime.forward_days_weekly,
        max_return_horizons_days=runtime.max_return_horizons_days,
        weekly_rsi_series=runtime.weekly_rsi_series,
        rsi_regime_tag=rsi_regime_tag,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )


def technicals_extension_stats(
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(Path("config/technical_backtesting.yaml"), "--config", help="Config path."),
    tail_pct: float | None = typer.Option(None, "--tail-pct", help="Symmetric tail threshold percentile (e.g. 5 => low<=5, high>=95). Overrides config tail_high_pct/tail_low_pct."),
    percentile_window_years: int | None = typer.Option(None, "--percentile-window-years", help="Rolling window (years) for extension percentiles + tail events. Default: auto (1y if <5y history, else 3y)."),
    out: Path | None = typer.Option(Path("data/reports/technicals/extension"), "--out", help="Output root for extension stats artifacts."),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write JSON artifact."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write Markdown artifact."),
    print_to_console: bool = typer.Option(False, "--print/--no-print", help="Print Markdown to console."),
    divergence_window_days: int = typer.Option(14, "--divergence-window-days", help="Lookback window (trading bars) for RSI divergence detection."),
    divergence_min_extension_days: int = typer.Option(5, "--divergence-min-extension-days", help="Minimum days in the window where extension percentile is elevated/depressed to qualify."),
    divergence_min_extension_percentile: float | None = typer.Option(None, "--divergence-min-extension-percentile", help="High extension percentile threshold for bearish divergence gating (default: tail_high_pct)."),
    divergence_max_extension_percentile: float | None = typer.Option(None, "--divergence-max-extension-percentile", help="Low extension percentile threshold for bullish divergence gating (default: tail_low_pct)."),
    divergence_min_price_delta_pct: float = typer.Option(0.0, "--divergence-min-price-delta-pct", help="Minimum % move between swing points (in the divergence direction)."),
    divergence_min_rsi_delta: float = typer.Option(0.0, "--divergence-min-rsi-delta", help="Minimum RSI difference between swing points."),
    rsi_overbought: float = typer.Option(70.0, "--rsi-overbought", help="RSI threshold for overbought tagging."),
    rsi_oversold: float = typer.Option(30.0, "--rsi-oversold", help="RSI threshold for oversold tagging."),
    require_rsi_extreme: bool = typer.Option(False, "--require-rsi-extreme/--allow-rsi-neutral", help="If set, only keep bearish divergences at overbought RSI and bullish divergences at oversold RSI."),
) -> None:
    """Compute extension percentile stats (tail events + rolling windows) from cached candles."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)
    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for extension stats.")

    runtime = _build_runtime(
        df=df,
        cfg=cfg,
        symbol=symbol,
        tail_pct=tail_pct,
        percentile_window_years=percentile_window_years,
        divergence_window_days=divergence_window_days,
        divergence_min_extension_days=divergence_min_extension_days,
        divergence_min_extension_percentile=divergence_min_extension_percentile,
        divergence_max_extension_percentile=divergence_max_extension_percentile,
        divergence_min_price_delta_pct=divergence_min_price_delta_pct,
        divergence_min_rsi_delta=divergence_min_rsi_delta,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        require_rsi_extreme=require_rsi_extreme,
    )
    if runtime.warmup_warning:
        console.print("[yellow]Warning:[/yellow] insufficient history for warmup; using full history.")

    payload, daily_tail_events, weekly_tail_events = enrich_extension_payload(runtime)
    md = _render_markdown(
        runtime=runtime,
        payload=payload,
        daily_tail_events=daily_tail_events,
        weekly_tail_events=weekly_tail_events,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
    )
    _write_extension_artifacts(
        out=out,
        sym_label=runtime.sym_label,
        asof=runtime.report_daily.asof,
        payload=payload,
        md=md,
        write_json=write_json,
        write_md=write_md,
        console=console,
    )
    if print_to_console:
        _print_markdown(md=md, console=console)


__all__ = ["technicals_extension_stats"]
