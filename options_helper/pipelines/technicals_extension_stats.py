from __future__ import annotations

from pathlib import Path


def run_extension_stats_for_symbol(
    *,
    symbol: str | None,
    ohlc_path: Path | None,
    cache_dir: Path,
    config_path: Path,
    tail_pct: float | None,
    percentile_window_years: int | None,
    out: Path,
    write_json: bool,
    write_md: bool,
    print_to_console: bool,
    divergence_window_days: int,
    divergence_min_extension_days: int,
    divergence_min_extension_percentile: float | None,
    divergence_max_extension_percentile: float | None,
    divergence_min_price_delta_pct: float,
    divergence_min_rsi_delta: float,
    rsi_overbought: float,
    rsi_oversold: float,
    require_rsi_extreme: bool,
) -> None:
    """Shared orchestration seam for technical extension-stats generation."""
    from options_helper.commands.technicals import technicals_extension_stats

    technicals_extension_stats(
        symbol=symbol,
        ohlc_path=ohlc_path,
        cache_dir=cache_dir,
        config_path=config_path,
        tail_pct=tail_pct,
        percentile_window_years=percentile_window_years,
        out=out,
        write_json=write_json,
        write_md=write_md,
        print_to_console=print_to_console,
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


__all__ = ["run_extension_stats_for_symbol"]
