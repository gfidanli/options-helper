from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.analysis.osi import format_osi, parse_contract_symbol
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.data.candles import CandleStore, last_close
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.data.market_types import DataFetchError
from options_helper.data.providers import get_provider
from options_helper.data.providers.base import MarketDataProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotSymbolResult:
    symbol: str
    snapshot_date: date | None
    expiries: int
    status: str
    error: str | None


def _add_osi_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "contractSymbol" not in df.columns:
        return df

    def _parse(raw: object) -> tuple[str | None, str | None]:
        parsed = parse_contract_symbol(raw)
        if parsed is None:
            return (None, None)
        try:
            osi = format_osi(parsed)
        except ValueError:
            osi = None
        return (osi, parsed.underlying_norm or None)

    parsed = df["contractSymbol"].map(_parse)
    osi_values = parsed.map(lambda item: item[0])
    underlying_values = parsed.map(lambda item: item[1])

    out = df.copy()
    out["osi"] = osi_values
    out["underlying_norm"] = underlying_values
    return out


def snapshot_full_chain_for_symbols(
    symbols: list[str],
    *,
    cache_dir: Path,
    candle_cache_dir: Path,
    spot_period: str = "10d",
    max_expiries: int | None = None,
    risk_free_rate: float = 0.0,
    symbol_source: str = "scanner",
    watchlists: list[str] | None = None,
    provider: MarketDataProvider | None = None,
) -> list[SnapshotSymbolResult]:
    provider = provider or get_provider()
    store = OptionsSnapshotStore(cache_dir)
    candle_store = CandleStore(candle_cache_dir, provider=provider)
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )

    results: list[SnapshotSymbolResult] = []
    for symbol in symbols:
        sym = symbol.strip().upper()
        if not sym:
            continue
        try:
            history = candle_store.get_daily_history(sym, period=spot_period)
            spot = last_close(history)
            data_date: date | None = history.index.max().date() if not history.empty else None

            if spot is None or spot <= 0:
                try:
                    underlying = provider.get_underlying(sym, period=spot_period, interval="1d")
                    spot = underlying.last_price
                    if data_date is None and underlying.history is not None and not underlying.history.empty:
                        try:
                            data_date = underlying.history.index.max().date()
                        except Exception:  # noqa: BLE001
                            pass
                except DataFetchError as exc:
                    results.append(
                        SnapshotSymbolResult(
                            symbol=sym,
                            snapshot_date=data_date,
                            expiries=0,
                            status="no_spot",
                            error=str(exc),
                        )
                    )
                    continue

            if spot is None or spot <= 0:
                results.append(
                    SnapshotSymbolResult(
                        symbol=sym,
                        snapshot_date=data_date,
                        expiries=0,
                        status="no_spot",
                        error="Missing spot price",
                    )
                )
                continue

            snapshot_date = data_date or date.today()
            expiries = provider.list_option_expiries(sym)
            if not expiries:
                results.append(
                    SnapshotSymbolResult(
                        symbol=sym,
                        snapshot_date=snapshot_date,
                        expiries=0,
                        status="no_expiries",
                        error="No listed option expiries",
                    )
                )
                continue
            if max_expiries is not None:
                expiries = expiries[:max_expiries]

            meta = {
                "spot": spot,
                "spot_period": spot_period,
                "full_chain": True,
                "all_expiries": True,
                "risk_free_rate": risk_free_rate,
                "window_pct": None,
                "strike_min": None,
                "strike_max": None,
                "snapshot_date": snapshot_date.isoformat(),
                "symbol_source": symbol_source,
                "watchlists": watchlists or [],
                "provider": provider_name,
            }
            if provider_version:
                meta["provider_version"] = provider_version

            ok_expiries = 0
            total_contracts = 0
            missing_bid_ask = 0
            stale_quotes = 0
            spread_pcts: list[float] = []
            for exp in expiries:
                try:
                    raw = provider.get_options_chain_raw(sym, exp)
                except DataFetchError as exc:
                    logger.warning("%s %s: %s", sym, exp.isoformat(), exc)
                    continue

                meta_with_underlying = dict(meta)
                meta_with_underlying["underlying"] = raw.get("underlying", {})

                calls = pd.DataFrame(raw.get("calls", []))
                puts = pd.DataFrame(raw.get("puts", []))
                calls["optionType"] = "call"
                puts["optionType"] = "put"
                calls["expiry"] = exp.isoformat()
                puts["expiry"] = exp.isoformat()

                df = pd.concat([calls, puts], ignore_index=True)
                df = _add_osi_columns(df)
                df = add_black_scholes_greeks_to_chain(
                    df,
                    spot=spot,
                    expiry=exp,
                    as_of=snapshot_date,
                    r=risk_free_rate,
                )

                quality = compute_quote_quality(df, min_volume=0, min_open_interest=0, as_of=snapshot_date)
                total_contracts += len(df)
                if not quality.empty:
                    warnings = quality["quality_warnings"].tolist()
                    missing_bid_ask += sum("quote_missing_bid_ask" in w for w in warnings if isinstance(w, list))
                    stale_quotes += sum("quote_stale" in w for w in warnings if isinstance(w, list))
                    spread_series = pd.to_numeric(quality["spread_pct"], errors="coerce")
                    spread_series = spread_series.where(spread_series >= 0)
                    spread_pcts.extend(spread_series.dropna().tolist())

                store.save_expiry_snapshot(
                    sym,
                    snapshot_date,
                    expiry=exp,
                    snapshot=df,
                    meta=meta_with_underlying,
                )
                store.save_expiry_snapshot_raw(
                    sym,
                    snapshot_date,
                    expiry=exp,
                    raw=raw,
                    meta=meta_with_underlying,
                )
                ok_expiries += 1

            if total_contracts > 0:
                spread_median = float(np.nanmedian(spread_pcts)) if spread_pcts else None
                spread_worst = float(np.nanmax(spread_pcts)) if spread_pcts else None
                store._upsert_meta(
                    store._day_dir(sym, snapshot_date),
                    {
                        "quote_quality": {
                            "contracts": int(total_contracts),
                            "missing_bid_ask_count": int(missing_bid_ask),
                            "missing_bid_ask_pct": float(missing_bid_ask / total_contracts),
                            "spread_pct_median": spread_median,
                            "spread_pct_worst": spread_worst,
                            "stale_quotes": int(stale_quotes),
                            "stale_pct": float(stale_quotes / total_contracts),
                        }
                    },
                )

            results.append(
                SnapshotSymbolResult(
                    symbol=sym,
                    snapshot_date=snapshot_date,
                    expiries=ok_expiries,
                    status="ok",
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                SnapshotSymbolResult(
                    symbol=sym,
                    snapshot_date=None,
                    expiries=0,
                    status="error",
                    error=str(exc),
                )
            )

    return results
