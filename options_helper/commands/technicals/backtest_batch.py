from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import typer

from options_helper.commands.technicals.backtest_batch_runtime import (
    run_technicals_backtest_batch_runtime,
)
from options_helper.commands.technicals_common import setup_technicals_logging
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.technicals_backtesting.backtest.batch_runner import normalize_batch_symbols

_STRATEGY_OPT = typer.Option(
    "MeanReversionIBS",
    "--strategy",
    help="Backtesting strategy id from technical_backtesting config.",
)
_SYMBOL_OPT = typer.Option(
    None,
    "--symbol",
    help="Single symbol input (for example: SPY).",
)
_TICKERS_OPT = typer.Option(
    None,
    "--tickers",
    help="Comma-separated symbol list (for example: SPY,QQQ,IWM).",
)
_CONFIG_PATH_OPT = typer.Option(
    Path("config/technical_backtesting.yaml"),
    "--config-path",
    help="Path to technical backtesting config YAML.",
)
_CACHE_DIR_OPT = typer.Option(
    Path("data/cache/candles"),
    "--cache-dir",
    help="Candle cache root used by batch runtime.",
)
_COMMISSION_OPT = typer.Option(
    None,
    "--commission",
    help="Commission override (CLI precedence over strategy/global config).",
)
_SLIPPAGE_BPS_OPT = typer.Option(
    None,
    "--slippage-bps",
    help="Slippage override in basis points (CLI precedence over strategy/global config).",
)
_REQUIRE_SMA_TREND_OPT = typer.Option(
    None,
    "--require-sma-trend/--no-require-sma-trend",
    help="Enable/disable the SMA trend overlay gate.",
)
_SMA_TREND_WINDOW_OPT = typer.Option(
    None,
    "--sma-trend-window",
    help="SMA window used by the SMA trend overlay gate.",
)
_REQUIRE_WEEKLY_TREND_OPT = typer.Option(
    None,
    "--require-weekly-trend/--no-require-weekly-trend",
    help="Enable/disable the weekly trend overlay gate.",
)
_REQUIRE_MA_DIRECTION_OPT = typer.Option(
    None,
    "--require-ma-direction/--no-require-ma-direction",
    help="Enable/disable the MA direction overlay gate.",
)
_MA_DIRECTION_WINDOW_OPT = typer.Option(
    None,
    "--ma-direction-window",
    help="MA window used by the MA direction overlay gate.",
)
_MA_DIRECTION_LOOKBACK_OPT = typer.Option(
    None,
    "--ma-direction-lookback",
    help="Lookback bars used for MA direction slope checks.",
)


def _resolve_symbols(*, symbol: str | None, tickers: str | None) -> tuple[str, ...]:
    raw_inputs = [token for token in (symbol, tickers) if token]
    if not raw_inputs:
        raise typer.BadParameter("Provide --symbol and/or --tickers.")
    try:
        return normalize_batch_symbols(raw_inputs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_overlay_overrides(
    *,
    require_sma_trend: bool | None,
    sma_trend_window: int | None,
    require_weekly_trend: bool | None,
    require_ma_direction: bool | None,
    ma_direction_window: int | None,
    ma_direction_lookback: int | None,
) -> dict[str, object]:
    candidates: dict[str, object | None] = {
        "require_sma_trend": require_sma_trend,
        "sma_trend_window": sma_trend_window,
        "require_weekly_trend": require_weekly_trend,
        "require_ma_direction": require_ma_direction,
        "ma_direction_window": ma_direction_window,
        "ma_direction_lookback": ma_direction_lookback,
    }
    return {
        key: value
        for key, value in candidates.items()
        if value is not None
    }


def _apply_strategy_default_overrides(
    *,
    cfg: Mapping[str, object],
    strategy: str,
    overrides: Mapping[str, object],
) -> dict[str, object]:
    merged_cfg = deepcopy(dict(cfg))
    if not overrides:
        return merged_cfg

    strategies = merged_cfg.get("strategies")
    if not isinstance(strategies, dict):
        raise typer.BadParameter("technical config must define a 'strategies' mapping.")
    strategy_cfg = strategies.get(strategy)
    if not isinstance(strategy_cfg, dict):
        raise typer.BadParameter(f"Unknown strategy: {strategy}")

    defaults = strategy_cfg.get("defaults")
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise typer.BadParameter(f"Strategy defaults must be a mapping for {strategy}.")

    strategy_cfg = dict(strategy_cfg)
    strategy_cfg["defaults"] = {**defaults, **dict(overrides)}
    strategies = dict(strategies)
    strategies[strategy] = strategy_cfg
    merged_cfg["strategies"] = strategies
    return merged_cfg


def technicals_backtest_batch(
    strategy: str = _STRATEGY_OPT,
    symbol: str | None = _SYMBOL_OPT,
    tickers: str | None = _TICKERS_OPT,
    config_path: Path = _CONFIG_PATH_OPT,
    cache_dir: Path = _CACHE_DIR_OPT,
    commission: float | None = _COMMISSION_OPT,
    slippage_bps: float | None = _SLIPPAGE_BPS_OPT,
    require_sma_trend: bool | None = _REQUIRE_SMA_TREND_OPT,
    sma_trend_window: int | None = _SMA_TREND_WINDOW_OPT,
    require_weekly_trend: bool | None = _REQUIRE_WEEKLY_TREND_OPT,
    require_ma_direction: bool | None = _REQUIRE_MA_DIRECTION_OPT,
    ma_direction_window: int | None = _MA_DIRECTION_WINDOW_OPT,
    ma_direction_lookback: int | None = _MA_DIRECTION_LOOKBACK_OPT,
) -> None:
    """Run one-symbol or batch technical backtests with optional overlay/cost overrides."""
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)
    symbols = _resolve_symbols(symbol=symbol, tickers=tickers)
    overlay_overrides = _build_overlay_overrides(
        require_sma_trend=require_sma_trend,
        sma_trend_window=sma_trend_window,
        require_weekly_trend=require_weekly_trend,
        require_ma_direction=require_ma_direction,
        ma_direction_window=ma_direction_window,
        ma_direction_lookback=ma_direction_lookback,
    )
    effective_cfg = _apply_strategy_default_overrides(
        cfg=cfg,
        strategy=strategy,
        overrides=overlay_overrides,
    )
    try:
        result = run_technicals_backtest_batch_runtime(
            symbols=symbols,
            strategy=strategy,
            cfg=effective_cfg,
            cache_dir=cache_dir,
            cli_commission=commission,
            cli_slippage_bps=slippage_bps,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        "Batch backtest complete: "
        f"symbols={len(result.symbols)} success={result.success_count} failed={result.failure_count}"
    )
    if result.failure_count > 0:
        failed_symbols = ",".join(outcome.symbol for outcome in result.outcomes if not outcome.ok)
        typer.echo(f"Failed symbols: {failed_symbols}")
    typer.echo("Informational output only; not financial advice.")


def register(app: typer.Typer) -> None:
    app.command("backtest-batch")(technicals_backtest_batch)


__all__ = ["register", "technicals_backtest_batch"]
