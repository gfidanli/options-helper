from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from options_helper.commands import technicals_legacy as legacy


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
    cfg = legacy.load_technical_backtesting_config(config_path)
    legacy.setup_technicals_logging(cfg)

    df = legacy._load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
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
    cfg = legacy.load_technical_backtesting_config(config_path)
    legacy.setup_technicals_logging(cfg)

    df = legacy._load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
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
    cfg = legacy.load_technical_backtesting_config(config_path)
    legacy.setup_technicals_logging(cfg)

    df = legacy._load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
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
                "train_stats": legacy._stats_to_dict(fold["train_stats"]),
                "validate_stats": legacy._stats_to_dict(fold["validate_stats"]),
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


technicals_run_all = legacy.technicals_run_all


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
