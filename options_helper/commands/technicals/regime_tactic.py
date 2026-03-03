from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from options_helper.analysis.price_regime import PriceRegimeTag, classify_price_regime
from options_helper.analysis.regime_tactic import map_regime_to_tactic
from options_helper.commands.technicals.common import _load_ohlc_df

if TYPE_CHECKING:
    import pandas as pd


class TradeDirectionOption(str, Enum):
    long = "long"
    short = "short"


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


def _load_ohlc_inputs(
    *,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
    argument_name: str,
) -> pd.DataFrame:
    try:
        return _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint=argument_name) from exc


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
) -> None:
    """Recommend a long/short entry tactic from symbol + market price regimes."""
    normalized_symbol = _normalize_symbol(symbol)
    normalized_market_symbol = _normalize_symbol(market_symbol) or "SPY"

    symbol_ohlc = _load_ohlc_inputs(
        ohlc_path=ohlc_path,
        symbol=normalized_symbol,
        cache_dir=cache_dir,
        argument_name="--ohlc-path",
    )
    market_ohlc = _load_ohlc_inputs(
        ohlc_path=market_ohlc_path,
        symbol=normalized_market_symbol if market_ohlc_path is None else None,
        cache_dir=cache_dir,
        argument_name="--market-ohlc-path",
    )

    symbol_regime, _ = classify_price_regime(symbol_ohlc)
    market_regime, _ = classify_price_regime(market_ohlc)
    recommendation = map_regime_to_tactic(
        market_regime,
        _symbol_regime_for_tactic(symbol_regime),
        direction=direction.value,
    )
    _print_recommendation(
        symbol_label=normalized_symbol or "UNKNOWN",
        symbol_asof=_asof_label(symbol_ohlc),
        symbol_regime=symbol_regime,
        market_label=normalized_market_symbol,
        market_asof=_asof_label(market_ohlc),
        market_regime=market_regime,
        recommendation=recommendation,
        direction=direction,
    )


def register(app: typer.Typer) -> None:
    app.command("regime-tactic")(technicals_regime_tactic)


__all__ = ["register", "technicals_regime_tactic"]
