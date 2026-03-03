from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from options_helper.analysis.price_regime import PriceRegimeTag, classify_price_regime
from options_helper.analysis.regime_tactic import map_regime_to_tactic
from options_helper.data.regime_tactic_artifacts import (
    build_regime_tactic_artifact_payload,
    write_regime_tactic_artifact_json,
)

if TYPE_CHECKING:
    import pandas as pd


class TradeDirectionOption(str, Enum):
    long = "long"
    short = "short"


_REQUIRED_OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def _normalize_symbol(symbol: str | None) -> str | None:
    token = str(symbol or "").strip().upper()
    if token:
        return token
    return None


def _asof_label(frame: pd.DataFrame) -> str:
    import pandas as pd

    if frame.empty:
        return "-"
    try:
        return pd.Timestamp(frame.index.max()).date().isoformat()
    except Exception:  # noqa: BLE001
        return str(frame.index[-1])


def _symbol_regime_for_tactic(symbol_regime: PriceRegimeTag) -> str:
    if symbol_regime == "choppy":
        return "sideways"
    return symbol_regime


def _validate_ohlc_frame(*, frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    import pandas as pd

    if frame is None or frame.empty:
        raise ValueError(f"{source_label} contains no OHLC rows.")

    alias_map = {str(col).lower(): str(col) for col in frame.columns}
    missing = [name for name in _REQUIRED_OHLC_COLUMNS if name.lower() not in alias_map]
    if missing:
        missing_labels = ", ".join(missing)
        raise ValueError(f"{source_label} is missing required OHLC columns: {missing_labels}.")

    out = frame[[alias_map[name.lower()] for name in _REQUIRED_OHLC_COLUMNS]].copy()
    out.columns = list(_REQUIRED_OHLC_COLUMNS)
    index = pd.to_datetime(out.index, errors="coerce")
    if isinstance(index, pd.Series):
        index = pd.DatetimeIndex(index.to_numpy())
    if not isinstance(index, pd.DatetimeIndex) or index.isna().any():
        raise ValueError(f"{source_label} contains unparseable timestamps in the index.")

    out.index = index
    duplicate_rows = int(out.index.duplicated(keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"{source_label} has duplicate timestamps ({duplicate_rows} rows).")
    if not out.index.is_monotonic_increasing:
        raise ValueError(f"{source_label} timestamps must be sorted ascending.")
    return out.apply(pd.to_numeric, errors="coerce")


def _load_ohlc_inputs(
    *,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
    path_option: str,
    symbol_option: str,
    source_label: str,
) -> pd.DataFrame:
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache, load_ohlc_from_path

    if ohlc_path is not None:
        try:
            frame = load_ohlc_from_path(ohlc_path)
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint=path_option) from exc
        try:
            return _validate_ohlc_frame(frame=frame, source_label=f"{source_label} OHLC ({ohlc_path})")
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint=path_option) from exc

    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        raise typer.BadParameter(
            f"Missing {source_label} OHLC source. Provide {path_option} or {symbol_option}.",
            param_hint=path_option,
        )

    frame = load_ohlc_from_cache(normalized_symbol, cache_dir, backfill_if_missing=False, period="max")
    if frame.empty:
        raise typer.BadParameter(
            (
                f"No cached {source_label} OHLC found for {normalized_symbol} in {cache_dir}. "
                f"Provide {path_option} for deterministic input."
            ),
            param_hint=path_option,
        )
    try:
        return _validate_ohlc_frame(frame=frame, source_label=f"Cached {source_label} OHLC for {normalized_symbol}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=path_option) from exc


def _print_recommendation(
    *,
    symbol_label: str,
    symbol_asof: str,
    symbol_regime: PriceRegimeTag,
    market_label: str,
    market_asof: str,
    market_regime: PriceRegimeTag,
    recommendation,
    direction: TradeDirectionOption,
) -> None:
    mapped_symbol_regime = _symbol_regime_for_tactic(symbol_regime)
    symbol_regime_text = (
        f"{symbol_regime} (mapped as {mapped_symbol_regime})"
        if mapped_symbol_regime != symbol_regime
        else symbol_regime
    )
    typer.echo(f"Symbol: {symbol_label} (as of {symbol_asof})")
    typer.echo(f"Market: {market_label} (as of {market_asof})")
    typer.echo(f"Direction: {direction.value}")
    typer.echo(f"Symbol regime: {symbol_regime_text}")
    typer.echo(f"Market regime: {market_regime}")
    typer.echo(f"Recommendation: {recommendation.tactic}")
    typer.echo(f"Support model: {recommendation.support_model}")
    if recommendation.rationale:
        typer.echo("Rationale:")
        for line in recommendation.rationale:
            typer.echo(f"- {line}")
    typer.echo("Informational output only; not financial advice.")


def technicals_regime_tactic(
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Underlying symbol (required when --ohlc-path is not provided).",
    ),
    ohlc_path: Path | None = typer.Option(
        None,
        "--ohlc-path",
        help="Optional CSV/Parquet OHLC path for the underlying symbol.",
    ),
    market_symbol: str = typer.Option(
        "SPY",
        "--market-symbol",
        help="Market proxy symbol used when --market-ohlc-path is not provided.",
    ),
    market_ohlc_path: Path | None = typer.Option(
        None,
        "--market-ohlc-path",
        help="Optional CSV/Parquet OHLC path for market proxy.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--cache-dir",
        help="Daily candle cache directory for symbol/market loads.",
    ),
    direction: TradeDirectionOption = typer.Option(
        TradeDirectionOption.long,
        "--direction",
        help="Trade direction context for tactic mapping.",
    ),
    out: Path = typer.Option(
        Path("data/reports/technicals/regime_tactic"),
        "--out",
        help="Output root for regime-tactic artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write JSON artifact."),
) -> None:
    """Recommend a long/short entry tactic from symbol + market price regimes."""
    normalized_symbol = _normalize_symbol(symbol)
    normalized_market_symbol = _normalize_symbol(market_symbol) or "SPY"

    symbol_ohlc = _load_ohlc_inputs(
        ohlc_path=ohlc_path,
        symbol=normalized_symbol,
        cache_dir=cache_dir,
        path_option="--ohlc-path",
        symbol_option="--symbol",
        source_label="symbol",
    )
    market_ohlc = _load_ohlc_inputs(
        ohlc_path=market_ohlc_path,
        symbol=normalized_market_symbol if market_ohlc_path is None else None,
        cache_dir=cache_dir,
        path_option="--market-ohlc-path",
        symbol_option="--market-symbol",
        source_label="market",
    )

    symbol_regime, symbol_diagnostics = classify_price_regime(symbol_ohlc)
    market_regime, market_diagnostics = classify_price_regime(market_ohlc)
    recommendation = map_regime_to_tactic(
        market_regime,
        _symbol_regime_for_tactic(symbol_regime),
        direction=direction.value,
    )
    symbol_asof = _asof_label(symbol_ohlc)
    market_asof = _asof_label(market_ohlc)

    _print_recommendation(
        symbol_label=normalized_symbol or "UNKNOWN",
        symbol_asof=symbol_asof,
        symbol_regime=symbol_regime,
        market_label=normalized_market_symbol,
        market_asof=market_asof,
        market_regime=market_regime,
        recommendation=recommendation,
        direction=direction,
    )

    if write_json:
        payload = build_regime_tactic_artifact_payload(
            asof_date=symbol_asof,
            symbol=normalized_symbol or "UNKNOWN",
            market_symbol=normalized_market_symbol,
            symbol_regime=symbol_regime,
            symbol_diagnostics=symbol_diagnostics,
            market_regime=market_regime,
            market_diagnostics=market_diagnostics,
            direction=direction.value,
            tactic=recommendation.tactic,
            support_model=recommendation.support_model,
            rationale=recommendation.rationale,
        )
        json_path = write_regime_tactic_artifact_json(payload, out=out)
        typer.echo(f"Wrote JSON: {json_path}")


def register(app: typer.Typer) -> None:
    app.command("regime-tactic")(technicals_regime_tactic)


__all__ = ["register", "technicals_regime_tactic"]
