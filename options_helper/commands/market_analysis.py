from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

import options_helper.cli_deps as cli_deps
from options_helper.analysis.iv_context import classify_iv_rv
from options_helper.analysis.tail_risk import TailRiskConfig, TailRiskResult, compute_tail_risk
from options_helper.data.confluence_config import ConfigError, load_confluence_config
from options_helper.data.tail_risk_artifacts import write_tail_risk_artifacts
from options_helper.schemas.common import utc_now
from options_helper.schemas.tail_risk import (
    TailRiskArtifact,
    TailRiskConfigArtifact,
    TailRiskIVContext,
    TailRiskPercentileRow,
)

app = typer.Typer(help="Market analysis utilities (informational only; not financial advice).")


@app.command("tail-risk")
def tail_risk(
    symbol: str = typer.Option(..., "--symbol", help="Ticker symbol."),
    lookback_days: int = typer.Option(252 * 6, "--lookback-days", min=30, help="Lookback window in trading days."),
    horizon_days: int = typer.Option(60, "--horizon-days", min=1, help="Forecast horizon in trading days."),
    num_simulations: int = typer.Option(
        25_000,
        "--num-simulations",
        min=1_000,
        help="Number of Monte Carlo simulated paths.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible simulations."),
    var_confidence: float = typer.Option(
        0.95,
        "--var-confidence",
        min=0.001,
        max=0.999,
        help="VaR/CVaR confidence level (e.g., 0.95).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh/--no-refresh",
        help="Refresh candle cache before running analysis.",
    ),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for artifacts (writes under {out}/tail_risk/{SYMBOL}/).",
    ),
) -> None:
    """Run Monte Carlo tail risk from local candle history with IV context (offline-first)."""
    console = Console(width=200)
    sym = _normalize_symbol(symbol)

    output_fmt = str(format or "").strip().lower()
    if output_fmt not in {"console", "json"}:
        raise typer.BadParameter("Invalid --format (use console|json)", param_hint="--format")

    provider = cli_deps.build_provider() if refresh else None
    candle_store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    derived_store = cli_deps.build_derived_store(derived_dir)

    history = (
        candle_store.get_daily_history(sym, period="max")
        if refresh
        else candle_store.load(sym)
    )
    close = _select_close(history)
    if close.empty:
        console.print(
            f"No candle history available for {sym}. "
            f"Use `options-helper ingest candles --symbol {sym}` (or run with `--refresh`)."
        )
        raise typer.Exit(1)

    cfg = TailRiskConfig(
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        seed=seed,
        var_confidence=var_confidence,
    )
    result = compute_tail_risk(close, config=cfg)
    derived_latest = _latest_derived_row(derived_store.load(sym))
    iv_context = _build_iv_context(derived_latest)
    artifact = _build_tail_risk_artifact(
        symbol=sym,
        config=cfg,
        result=result,
        iv_context=iv_context,
    )

    if output_fmt == "json":
        console.print(artifact.model_dump_json(indent=2))
    else:
        _render_console(console, artifact)

    if out is not None:
        paths = write_tail_risk_artifacts(artifact, out_dir=out)
        console.print(f"Saved: {paths.json_path}")
        console.print(f"Saved: {paths.report_path}")


def _normalize_symbol(value: str) -> str:
    raw = str(value or "").strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def _select_close(history: pd.DataFrame | None) -> pd.Series:
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    if "Close" in history.columns:
        return pd.to_numeric(history["Close"], errors="coerce")
    if "close" in history.columns:
        return pd.to_numeric(history["close"], errors="coerce")
    return pd.Series(dtype="float64")


def _latest_derived_row(df: pd.DataFrame) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    temp = df.copy()
    if "date" in temp.columns:
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.sort_values(by="date", kind="stable")
    latest = temp.iloc[-1]
    return dict(latest.to_dict())


def _build_iv_context(derived_row: dict[str, Any] | None) -> TailRiskIVContext | None:
    if not derived_row:
        return None
    iv_rv = _as_float(derived_row.get("iv_rv_20d"))
    if iv_rv is None:
        return None

    low, high = _load_iv_thresholds()
    regime = classify_iv_rv(iv_rv, low=low, high=high)
    if regime is None:
        return None

    return TailRiskIVContext(
        label=regime.label,
        reason=regime.reason,
        iv_rv_20d=regime.iv_rv,
        low=regime.low,
        high=regime.high,
        atm_iv_near=_as_float(derived_row.get("atm_iv_near")),
        rv_20d=_as_float(derived_row.get("rv_20d")),
        atm_iv_near_percentile=_as_float(derived_row.get("atm_iv_near_percentile")),
        iv_term_slope=_as_float(derived_row.get("iv_term_slope")),
    )


def _load_iv_thresholds() -> tuple[float, float]:
    default_low = 0.8
    default_high = 1.2
    try:
        cfg = load_confluence_config()
    except (ConfigError, OSError, ValueError):
        return default_low, default_high

    iv_cfg = (cfg.get("iv_regime") or {}) if isinstance(cfg, dict) else {}
    try:
        low = float(iv_cfg.get("low", default_low))
        high = float(iv_cfg.get("high", default_high))
    except Exception:  # noqa: BLE001
        return default_low, default_high
    if low >= high:
        return default_low, default_high
    return low, high


def _build_tail_risk_artifact(
    *,
    symbol: str,
    config: TailRiskConfig,
    result: TailRiskResult,
    iv_context: TailRiskIVContext | None,
) -> TailRiskArtifact:
    end_rows: list[TailRiskPercentileRow] = []
    for percentile in sorted(result.end_price_percentiles):
        end_rows.append(
            TailRiskPercentileRow(
                percentile=float(percentile),
                price=_as_float(result.end_price_percentiles.get(percentile)),
                return_pct=_as_float(result.end_return_percentiles.get(percentile)),
            )
        )
    return TailRiskArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=result.as_of,
        symbol=symbol,
        disclaimer="Not financial advice.",
        config=TailRiskConfigArtifact(
            lookback_days=config.lookback_days,
            horizon_days=config.horizon_days,
            num_simulations=config.num_simulations,
            seed=config.seed,
            var_confidence=config.var_confidence,
            end_percentiles=[float(value) for value in config.end_percentiles],
            chart_percentiles=[float(value) for value in config.chart_percentiles],
            sample_paths=min(config.sample_paths, config.num_simulations),
            trading_days_per_year=config.trading_days_per_year,
        ),
        spot=result.spot,
        realized_vol_annual=result.realized_vol_annual,
        expected_return_annual=result.expected_return_annual,
        var_return=result.var_return,
        cvar_return=result.cvar_return,
        end_percentiles=end_rows,
        iv_context=iv_context,
        warnings=list(result.warnings),
    )


def _render_console(console: Console, artifact: TailRiskArtifact) -> None:
    console.print(f"[bold]{artifact.symbol}[/bold] tail risk as-of {artifact.as_of}")
    console.print(
        f"Spot={artifact.spot:.2f} | RV={artifact.realized_vol_annual:.2%} | "
        f"ExpRet={artifact.expected_return_annual:.2%} | "
        f"VaR({artifact.config.var_confidence:.0%})={artifact.var_return:.2%} | "
        f"CVaR({artifact.config.var_confidence:.0%})="
        + ("-" if artifact.cvar_return is None else f"{artifact.cvar_return:.2%}")
    )

    table = Table(title="Horizon End Percentiles")
    table.add_column("Pct", justify="right")
    table.add_column("End Price", justify="right")
    table.add_column("End Return", justify="right")
    for row in artifact.end_percentiles:
        table.add_row(
            f"{row.percentile:.0f}",
            "-" if row.price is None else f"{row.price:.2f}",
            "-" if row.return_pct is None else f"{row.return_pct:.2%}",
        )
    console.print(table)

    if artifact.iv_context is None:
        console.print(
            "IV context unavailable. Run `options-helper derived update --symbol "
            f"{artifact.symbol}` after snapshot ingestion."
        )
    else:
        iv = artifact.iv_context
        console.print(
            "IV context: "
            f"{iv.label} (IV/RV20={iv.iv_rv_20d:.2f}x; low={iv.low:.2f}, high={iv.high:.2f})"
        )
        console.print(iv.reason)
        if iv.atm_iv_near is not None or iv.rv_20d is not None:
            console.print(
                "ATM IV near="
                + ("-" if iv.atm_iv_near is None else f"{iv.atm_iv_near:.2%}")
                + ", RV20="
                + ("-" if iv.rv_20d is None else f"{iv.rv_20d:.2%}")
                + ", IV percentile="
                + (
                    "-"
                    if iv.atm_iv_near_percentile is None
                    else f"{iv.atm_iv_near_percentile:.0f}%"
                )
                + ", IV term slope="
                + ("-" if iv.iv_term_slope is None else f"{iv.iv_term_slope:+.3f}")
            )

    if artifact.warnings:
        console.print("Warnings: " + ", ".join(artifact.warnings))
    console.print(artifact.disclaimer)


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out

