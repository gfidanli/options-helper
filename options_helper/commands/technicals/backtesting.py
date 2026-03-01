from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from options_helper.commands.technicals_common import setup_technicals_logging
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config


_INTERVAL_OPT = typer.Option(
    None,
    "--interval",
    help="Target candle interval token (for artifacts and intraday resampling).",
)
_INTRADAY_DIR_OPT = typer.Option(
    None,
    "--intraday-dir",
    help="Intraday partition root (enables intraday input mode).",
)
_INTRADAY_TIMEFRAME_OPT = typer.Option(
    "1Min",
    "--intraday-timeframe",
    help="Source intraday timeframe in partition store (1Min or 5Min).",
)
_INTRADAY_START_OPT = typer.Option(
    None,
    "--intraday-start",
    help="Intraday start session date (YYYY-MM-DD).",
)
_INTRADAY_END_OPT = typer.Option(
    None,
    "--intraday-end",
    help="Intraday end session date (YYYY-MM-DD).",
)


@dataclass(frozen=True)
class _LoadedOhlcData:
    frame: object
    interval: str
    intraday_coverage: dict | None = None


def _parse_intraday_day(value: str | None, *, option_name: str) -> date:
    token = str(value or "").strip()
    if not token:
        raise typer.BadParameter(f"{option_name} is required when using --intraday-dir")
    try:
        return date.fromisoformat(token)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be YYYY-MM-DD; got {value!r}") from exc


def _resolve_interval_token(*, interval: str | None, intraday_mode: bool, intraday_timeframe: str) -> str:
    token = str(interval or "").strip()
    if token:
        return token
    if intraday_mode:
        intraday_token = str(intraday_timeframe or "").strip()
        return intraday_token or "1Min"
    return "1d"


def _serialize_intraday_coverage(coverage: object) -> dict:
    def _to_iso(days: tuple[date, ...]) -> list[str]:
        return [day.isoformat() for day in days]

    return {
        "symbol": getattr(coverage, "symbol", None),
        "base_timeframe": getattr(coverage, "base_timeframe", None),
        "target_interval": getattr(coverage, "target_interval", None),
        "requested_days": _to_iso(getattr(coverage, "requested_days", tuple())),
        "loaded_days": _to_iso(getattr(coverage, "loaded_days", tuple())),
        "missing_days": _to_iso(getattr(coverage, "missing_days", tuple())),
        "empty_days": _to_iso(getattr(coverage, "empty_days", tuple())),
        "requested_day_count": int(getattr(coverage, "requested_day_count", 0)),
        "loaded_day_count": int(getattr(coverage, "loaded_day_count", 0)),
        "missing_day_count": int(getattr(coverage, "missing_day_count", 0)),
        "empty_day_count": int(getattr(coverage, "empty_day_count", 0)),
        "loaded_row_count": int(getattr(coverage, "loaded_row_count", 0)),
        "output_row_count": int(getattr(coverage, "output_row_count", 0)),
    }


def _load_ohlc_df(
    *,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
    interval: str | None = None,
    intraday_dir: Path | None = None,
    intraday_timeframe: str = "1Min",
    intraday_start: str | None = None,
    intraday_end: str | None = None,
) -> _LoadedOhlcData:
    from options_helper.data.candles import CandleCacheError
    from options_helper.data.technical_backtesting_intraday import load_intraday_candles
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache, load_ohlc_from_path

    intraday_mode = intraday_dir is not None
    resolved_interval = _resolve_interval_token(
        interval=interval,
        intraday_mode=intraday_mode,
        intraday_timeframe=intraday_timeframe,
    )
    if ohlc_path:
        return _LoadedOhlcData(frame=load_ohlc_from_path(ohlc_path), interval=resolved_interval)
    if intraday_mode:
        if not symbol:
            raise typer.BadParameter("--symbol is required when using --intraday-dir")
        start_day = _parse_intraday_day(intraday_start, option_name="--intraday-start")
        end_day = _parse_intraday_day(intraday_end, option_name="--intraday-end")
        if end_day < start_day:
            raise typer.BadParameter("--intraday-end must be on or after --intraday-start")
        try:
            loaded = load_intraday_candles(
                symbol=symbol,
                start_day=start_day,
                end_day=end_day,
                base_timeframe=intraday_timeframe,
                target_interval=resolved_interval,
                intraday_dir=intraday_dir,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        return _LoadedOhlcData(
            frame=loaded.candles,
            interval=loaded.coverage.target_interval,
            intraday_coverage=_serialize_intraday_coverage(loaded.coverage),
        )
    if symbol:
        try:
            return _LoadedOhlcData(
                frame=load_ohlc_from_cache(
                    symbol,
                    cache_dir,
                    backfill_if_missing=True,
                    period="max",
                    raise_on_backfill_error=True,
                ),
                interval=resolved_interval,
            )
        except CandleCacheError as exc:
            raise typer.BadParameter(f"Failed to backfill OHLC for {symbol}: {exc}") from exc
    raise typer.BadParameter("Provide --ohlc-path, --intraday-dir, or --symbol/--cache-dir")


def _stats_to_dict(stats: object | None) -> dict | None:
    import pandas as pd

    if stats is None:
        return None
    if isinstance(stats, pd.Series):
        return {k: v for k, v in stats.to_dict().items() if not str(k).startswith("_")}
    if isinstance(stats, dict):
        return {k: v for k, v in stats.items() if not str(k).startswith("_")}
    return {"value": stats}


def _select_strategy_feature_frame(
    *,
    features,
    needed_columns: list[str],
):
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [column for column in needed_columns if column in features.columns]
    return features.loc[:, [column for column in cols if column in features.columns]]


def _serialize_walk_forward_folds(folds: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for fold in folds:
        rows.append(
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
    return rows


def _walk_forward_artifacts_payload(result) -> tuple[dict, object | None]:
    folds_out = _serialize_walk_forward_folds(result.folds)
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
    return wf_dict, heatmap


def _artifact_data_meta(features, *, warmup_bars: int, intraday_coverage: dict | None = None) -> dict:
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup_bars,
    }
    if intraday_coverage:
        data_meta["intraday_coverage"] = intraday_coverage
    return data_meta


def _artifact_optimize_meta(opt_cfg: dict, strat_cfg: dict) -> dict:
    return {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }


def _run_strategy_for_symbol(
    *,
    cfg: dict,
    features,
    strategy: str,
    strat_cfg: dict,
    warmup: int,
    required_feature_columns_for_strategy,
    get_strategy,
    optimize_params,
    walk_forward_optimize,
) -> tuple[dict, object | None, dict | None, object | None]:
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    strat_features = _select_strategy_feature_frame(features=features, needed_columns=needed)
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
        wf_dict, heatmap = _walk_forward_artifacts_payload(result)
        return result.params, None, wf_dict, heatmap

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
    return best_params, train_stats, None, heatmap


def _run_all_strategies_for_symbol(
    *,
    symbol: str,
    cache_dir: Path,
    interval: str | None,
    intraday_dir: Path | None,
    intraday_timeframe: str,
    intraday_start: str | None,
    intraday_end: str | None,
    cfg: dict,
    console: Console,
    compute_features,
    warmup_bars,
    write_artifacts,
    required_feature_columns_for_strategy,
    get_strategy,
    optimize_params,
    walk_forward_optimize,
) -> None:
    loaded = _load_ohlc_df(
        ohlc_path=None,
        symbol=symbol,
        cache_dir=cache_dir,
        interval=interval,
        intraday_dir=intraday_dir,
        intraday_timeframe=intraday_timeframe,
        intraday_start=intraday_start,
        intraday_end=intraday_end,
    )
    df = loaded.frame
    if df.empty:
        console.print(f"[yellow]No OHLC data found for {symbol}.[/yellow]")
        return
    features = compute_features(df, cfg)
    warmup = warmup_bars(cfg)
    for strategy, strat_cfg in cfg["strategies"].items():
        if not strat_cfg.get("enabled", False):
            continue
        params, train_stats, wf_dict, heatmap = _run_strategy_for_symbol(
            cfg=cfg,
            features=features,
            strategy=strategy,
            strat_cfg=strat_cfg,
            warmup=warmup,
            required_feature_columns_for_strategy=required_feature_columns_for_strategy,
            get_strategy=get_strategy,
            optimize_params=optimize_params,
            walk_forward_optimize=walk_forward_optimize,
        )
        write_artifacts(
            cfg,
            ticker=symbol,
            strategy=strategy,
            interval=loaded.interval,
            params=params,
            train_stats=train_stats,
            walk_forward_result=wf_dict,
            optimize_meta=_artifact_optimize_meta(cfg["optimization"], strat_cfg),
            data_meta=_artifact_data_meta(
                features,
                warmup_bars=warmup,
                intraday_coverage=loaded.intraday_coverage,
            ),
            heatmap=heatmap,
        )
    console.print(f"[green]Completed[/green] {symbol}")


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

    loaded = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    df = loaded.frame
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


def _execute_technicals_optimize(
    *,
    strategy: str,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
    interval: str | None,
    intraday_dir: Path | None,
    intraday_timeframe: str,
    intraday_start: str | None,
    intraday_end: str | None,
    config_path: Path,
) -> None:
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)
    loaded = _load_ohlc_df(
        ohlc_path=ohlc_path,
        symbol=symbol,
        cache_dir=cache_dir,
        interval=interval,
        intraday_dir=intraday_dir,
        intraday_timeframe=intraday_timeframe,
        intraday_start=intraday_start,
        intraday_end=intraday_end,
    )
    df = loaded.frame
    if df.empty:
        raise typer.BadParameter("No OHLC data found for optimization.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    features = _select_strategy_feature_frame(features=features, needed_columns=needed)

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
    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        interval=loaded.interval,
        params=best_params,
        train_stats=best_stats,
        walk_forward_result=None,
        optimize_meta=_artifact_optimize_meta(opt_cfg, strat_cfg),
        data_meta=_artifact_data_meta(
            features,
            warmup_bars=warmup,
            intraday_coverage=loaded.intraday_coverage,
        ),
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


def technicals_optimize(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    interval: str | None = _INTERVAL_OPT,
    intraday_dir: Path | None = _INTRADAY_DIR_OPT,
    intraday_timeframe: str = _INTRADAY_TIMEFRAME_OPT,
    intraday_start: str | None = _INTRADAY_START_OPT,
    intraday_end: str | None = _INTRADAY_END_OPT,
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Optimize strategy parameters for a single dataset."""
    _execute_technicals_optimize(
        strategy=strategy,
        ohlc_path=ohlc_path,
        symbol=symbol,
        cache_dir=cache_dir,
        interval=interval,
        intraday_dir=intraday_dir,
        intraday_timeframe=intraday_timeframe,
        intraday_start=intraday_start,
        intraday_end=intraday_end,
        config_path=config_path,
    )


def _execute_technicals_walk_forward(
    *,
    strategy: str,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
    interval: str | None,
    intraday_dir: Path | None,
    intraday_timeframe: str,
    intraday_start: str | None,
    intraday_end: str | None,
    config_path: Path,
) -> None:
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
    from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    setup_technicals_logging(cfg)
    loaded = _load_ohlc_df(
        ohlc_path=ohlc_path,
        symbol=symbol,
        cache_dir=cache_dir,
        interval=interval,
        intraday_dir=intraday_dir,
        intraday_timeframe=intraday_timeframe,
        intraday_start=intraday_start,
        intraday_end=intraday_end,
    )
    df = loaded.frame
    if df.empty:
        raise typer.BadParameter("No OHLC data found for walk-forward.")
    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    walk_cfg = cfg["walk_forward"]
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    features = _select_strategy_feature_frame(features=features, needed_columns=needed)
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
    wf_dict, heatmap = _walk_forward_artifacts_payload(result)
    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        interval=loaded.interval,
        params=result.params,
        train_stats=None,
        walk_forward_result=wf_dict,
        optimize_meta=_artifact_optimize_meta(opt_cfg, strat_cfg),
        data_meta=_artifact_data_meta(
            features,
            warmup_bars=warmup,
            intraday_coverage=loaded.intraday_coverage,
        ),
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


def technicals_walk_forward(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    interval: str | None = _INTERVAL_OPT,
    intraday_dir: Path | None = _INTRADAY_DIR_OPT,
    intraday_timeframe: str = _INTRADAY_TIMEFRAME_OPT,
    intraday_start: str | None = _INTRADAY_START_OPT,
    intraday_end: str | None = _INTRADAY_END_OPT,
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run walk-forward optimization and write artifacts."""
    _execute_technicals_walk_forward(
        strategy=strategy,
        ohlc_path=ohlc_path,
        symbol=symbol,
        cache_dir=cache_dir,
        interval=interval,
        intraday_dir=intraday_dir,
        intraday_timeframe=intraday_timeframe,
        intraday_start=intraday_start,
        intraday_end=intraday_end,
        config_path=config_path,
    )


def technicals_run_all(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated tickers."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    interval: str | None = _INTERVAL_OPT,
    intraday_dir: Path | None = _INTRADAY_DIR_OPT,
    intraday_timeframe: str = _INTRADAY_TIMEFRAME_OPT,
    intraday_start: str | None = _INTRADAY_START_OPT,
    intraday_end: str | None = _INTRADAY_END_OPT,
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run both strategies for a list of tickers."""
    from options_helper.data.technical_backtesting_artifacts import write_artifacts
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
            _run_all_strategies_for_symbol(
                symbol=symbol,
                cache_dir=cache_dir,
                interval=interval,
                intraday_dir=intraday_dir,
                intraday_timeframe=intraday_timeframe,
                intraday_start=intraday_start,
                intraday_end=intraday_end,
                cfg=cfg,
                console=console,
                compute_features=compute_features,
                warmup_bars=warmup_bars,
                write_artifacts=write_artifacts,
                required_feature_columns_for_strategy=required_feature_columns_for_strategy,
                get_strategy=get_strategy,
                optimize_params=optimize_params,
                walk_forward_optimize=walk_forward_optimize,
            )
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
