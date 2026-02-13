from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

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


def _stats_to_dict(stats: object | None) -> dict | None:
    import pandas as pd

    if stats is None:
        return None
    if isinstance(stats, pd.Series):
        return {k: v for k, v in stats.to_dict().items() if not str(k).startswith("_")}
    if isinstance(stats, dict):
        return {k: v for k, v in stats.items() if not str(k).startswith("_")}
    return {"value": stats}


def technicals_compute_indicators(
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    output: Path | None = typer.Option(None, "--output", help="Output CSV/parquet path."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Compute indicators from OHLC data and optionally persist to disk."""
    from options_helper.technicals_backtesting.pipeline import compute_features

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for indicator computation.")

    features = compute_features(df, cfg)
    console.print(f"Computed features: {len(features)} rows, {len(features.columns)} columns")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".parquet":
            features.to_parquet(output)
        else:
            features.to_csv(output)
        console.print(f"Wrote features to {output}")


def technicals_optimize(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Optimize strategy parameters for a single dataset."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for optimization.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [column for column in needed if column in features.columns]
    features = features.loc[:, [column for column in cols if column in features.columns]]

    warmup = warmup_bars(cfg)
    best_params, best_stats, heatmap = optimize_params(
        features,
        StrategyClass,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        opt_cfg["maximize"],
        opt_cfg["method"],
        opt_cfg.get("sambo", {}),
        opt_cfg.get("custom_score", {}),
        warmup_bars=warmup,
        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
    )

    console.print(f"Best params: {best_params}")
    ticker = symbol or "UNKNOWN"
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup,
    }
    optimize_meta = {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }
    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        params=best_params,
        train_stats=best_stats,
        walk_forward_result=None,
        optimize_meta=optimize_meta,
        data_meta=data_meta,
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


def technicals_walk_forward(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run walk-forward optimization and write artifacts."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for walk-forward.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    walk_cfg = cfg["walk_forward"]
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [column for column in needed if column in features.columns]
    features = features.loc[:, [column for column in cols if column in features.columns]]

    warmup = warmup_bars(cfg)
    result = walk_forward_optimize(
        features,
        StrategyClass,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        opt_cfg["maximize"],
        opt_cfg["method"],
        opt_cfg.get("sambo", {}),
        opt_cfg.get("custom_score", {}),
        walk_cfg,
        strat_cfg["defaults"],
        warmup_bars=warmup,
        min_train_bars=opt_cfg.get("min_train_bars", 0),
        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
    )

    ticker = symbol or "UNKNOWN"
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup,
    }
    optimize_meta = {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }
    folds_out = []
    for fold in result.folds:
        folds_out.append(
            {
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "validate_start": fold["validate_start"],
                "validate_end": fold["validate_end"],
                "best_params": fold["best_params"],
                "train_stats": _stats_to_dict(fold["train_stats"]),
                "validate_stats": _stats_to_dict(fold["validate_stats"]),
                "validate_score": fold["validate_score"],
            }
        )
    wf_dict = {
        "params": result.params,
        "folds": folds_out,
        "stability": result.stability,
        "used_defaults": result.used_defaults,
        "reason": result.reason,
    }
    heatmap = None
    if result.folds:
        best_fold = max(result.folds, key=lambda fold: fold.get("validate_score", float("-inf")))
        heatmap = best_fold.get("heatmap")

    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        params=result.params,
        train_stats=None,
        walk_forward_result=wf_dict,
        optimize_meta=optimize_meta,
        data_meta=data_meta,
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


def technicals_run_all(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated tickers."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run both strategies for a list of tickers."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache
    from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
    from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)

    symbols = [token.strip().upper() for token in tickers.split(",") if token.strip()]
    if not symbols:
        raise typer.BadParameter("Provide at least one ticker.")

    for symbol in symbols:
        try:
            df = load_ohlc_from_cache(symbol, cache_dir)
            if df.empty:
                console.print(f"[yellow]No data for {symbol} in cache.[/yellow]")
                continue
            features = compute_features(df, cfg)
            warmup = warmup_bars(cfg)
            for strategy, strat_cfg in cfg["strategies"].items():
                if not strat_cfg.get("enabled", False):
                    continue
                needed = required_feature_columns_for_strategy(strategy, strat_cfg)
                cols = ["Open", "High", "Low", "Close"]
                if "Volume" in features.columns:
                    cols.append("Volume")
                cols += [column for column in needed if column in features.columns]
                strat_features = features.loc[:, [column for column in cols if column in features.columns]]
                StrategyClass = get_strategy(strategy)
                opt_cfg = cfg["optimization"]
                if cfg["walk_forward"]["enabled"]:
                    result = walk_forward_optimize(
                        strat_features,
                        StrategyClass,
                        cfg["backtest"],
                        strat_cfg["search_space"],
                        strat_cfg["constraints"],
                        opt_cfg["maximize"],
                        opt_cfg["method"],
                        opt_cfg.get("sambo", {}),
                        opt_cfg.get("custom_score", {}),
                        cfg["walk_forward"],
                        strat_cfg["defaults"],
                        warmup_bars=warmup,
                        min_train_bars=opt_cfg.get("min_train_bars", 0),
                        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
                    )
                    folds_out = []
                    for fold in result.folds:
                        folds_out.append(
                            {
                                "train_start": fold["train_start"],
                                "train_end": fold["train_end"],
                                "validate_start": fold["validate_start"],
                                "validate_end": fold["validate_end"],
                                "best_params": fold["best_params"],
                                "train_stats": _stats_to_dict(fold["train_stats"]),
                                "validate_stats": _stats_to_dict(fold["validate_stats"]),
                                "validate_score": fold["validate_score"],
                            }
                        )
                    wf_dict = {
                        "params": result.params,
                        "folds": folds_out,
                        "stability": result.stability,
                        "used_defaults": result.used_defaults,
                        "reason": result.reason,
                    }
                    heatmap = None
                    if result.folds:
                        best_fold = max(result.folds, key=lambda fold: fold.get("validate_score", float("-inf")))
                        heatmap = best_fold.get("heatmap")
                    train_stats = None
                    params = result.params
                else:
                    best_params, train_stats, heatmap = optimize_params(
                        strat_features,
                        StrategyClass,
                        cfg["backtest"],
                        strat_cfg["search_space"],
                        strat_cfg["constraints"],
                        opt_cfg["maximize"],
                        opt_cfg["method"],
                        opt_cfg.get("sambo", {}),
                        opt_cfg.get("custom_score", {}),
                        warmup_bars=warmup,
                        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
                    )
                    wf_dict = None
                    params = best_params

                data_meta = {
                    "start": features.index.min(),
                    "end": features.index.max(),
                    "bars": len(features),
                    "warmup_bars": warmup,
                }
                optimize_meta = {
                    "method": opt_cfg["method"],
                    "maximize": opt_cfg["maximize"],
                    "constraints": strat_cfg["constraints"],
                }
                write_artifacts(
                    cfg,
                    ticker=symbol,
                    strategy=strategy,
                    params=params,
                    train_stats=train_stats,
                    walk_forward_result=wf_dict,
                    optimize_meta=optimize_meta,
                    data_meta=data_meta,
                    heatmap=heatmap,
                )
            console.print(f"[green]Completed[/green] {symbol}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{symbol} failed:[/red] {exc}")


def register(app: typer.Typer) -> None:
    app.command("compute-indicators")(technicals_compute_indicators)
    app.command("optimize")(technicals_optimize)
    app.command("walk-forward")(technicals_walk_forward)
    app.command("run-all")(technicals_run_all)


__all__ = [
    "register",
    "technicals_compute_indicators",
    "technicals_optimize",
    "technicals_walk_forward",
    "technicals_run_all",
]
