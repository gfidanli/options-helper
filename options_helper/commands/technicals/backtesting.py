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


technicals_optimize = legacy.technicals_optimize
technicals_walk_forward = legacy.technicals_walk_forward
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
