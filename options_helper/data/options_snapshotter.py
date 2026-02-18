from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import inspect
import logging

import numpy as np
import pandas as pd

from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.analysis.osi import format_osi, parse_contract_symbol
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.data.candles import last_close
from options_helper.data.market_types import DataFetchError
from options_helper.data.providers import get_provider
from options_helper.data.providers.base import MarketDataProvider
from options_helper.data.store_factory import get_candle_store, get_options_snapshot_store


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotSymbolResult:
    symbol: str
    snapshot_date: date | None
    expiries: int
    status: str
    error: str | None


@dataclass(frozen=True)
class _SpotResolution:
    spot: float | None
    data_date: date | None
    error: str | None


@dataclass(frozen=True)
class _ExpirySnapshotCollection:
    chain_frames: list[pd.DataFrame]
    quality_frames: list[pd.DataFrame]
    raw_by_expiry: dict[date, dict]
    saved_expiries: list[date]
    underlying_payload: dict | None


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


def _supports_snapshot_date(provider: MarketDataProvider) -> bool:
    try:
        sig = inspect.signature(provider.get_options_chain_raw)
    except (TypeError, ValueError):
        return False
    if "snapshot_date" in sig.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def _fetch_chain_raw(
    provider: MarketDataProvider,
    symbol: str,
    expiry: date,
    snapshot_date: date,
) -> dict:
    if _supports_snapshot_date(provider):
        return provider.get_options_chain_raw(symbol, expiry, snapshot_date=snapshot_date)
    return provider.get_options_chain_raw(symbol, expiry)


def _provider_details(provider: MarketDataProvider) -> tuple[str, str | None, dict | None]:
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )
    provider_params = None
    if hasattr(provider, "provider_params"):
        try:
            provider_params = provider.provider_params
        except Exception:  # noqa: BLE001
            provider_params = None
    if not isinstance(provider_params, dict):
        provider_params = None
    return provider_name, provider_version, provider_params


def _resolve_spot(
    *,
    symbol: str,
    candle_store,
    provider: MarketDataProvider,
    spot_period: str,
) -> _SpotResolution:
    history = candle_store.get_daily_history(symbol, period=spot_period)
    spot = last_close(history)
    data_date: date | None = history.index.max().date() if not history.empty else None
    if spot is not None and spot > 0:
        return _SpotResolution(spot=spot, data_date=data_date, error=None)
    try:
        underlying = provider.get_underlying(symbol, period=spot_period, interval="1d")
        spot = underlying.last_price
        if data_date is None and underlying.history is not None and not underlying.history.empty:
            try:
                data_date = underlying.history.index.max().date()
            except Exception:  # noqa: BLE001
                pass
    except DataFetchError as exc:
        return _SpotResolution(spot=None, data_date=data_date, error=str(exc))
    if spot is None or spot <= 0:
        return _SpotResolution(spot=None, data_date=data_date, error="Missing spot price")
    return _SpotResolution(spot=spot, data_date=data_date, error=None)


def _build_snapshot_meta(
    *,
    spot: float,
    snapshot_date: date,
    spot_period: str,
    risk_free_rate: float,
    symbol_source: str,
    watchlists: list[str] | None,
    provider_name: str,
    provider_version: str | None,
    provider_params: dict | None,
) -> dict:
    meta: dict[str, object] = {
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
    if provider_params is not None:
        meta["provider_params"] = provider_params
    if provider_version:
        meta["provider_version"] = provider_version
    return meta


def _collect_expiry_snapshots(
    *,
    provider: MarketDataProvider,
    symbol: str,
    expiries: list[date],
    snapshot_date: date,
    spot: float,
    risk_free_rate: float,
) -> _ExpirySnapshotCollection:
    chain_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    raw_by_expiry: dict[date, dict] = {}
    saved_expiries: list[date] = []
    underlying_payload: dict | None = None
    for expiry in expiries:
        try:
            raw = _fetch_chain_raw(provider, symbol, expiry, snapshot_date)
        except DataFetchError as exc:
            logger.warning("%s %s: %s", symbol, expiry.isoformat(), exc)
            continue
        underlying = raw.get("underlying")
        if not isinstance(underlying, dict):
            underlying = {}
        if underlying_payload is None or underlying:
            underlying_payload = underlying
        calls = pd.DataFrame(raw.get("calls", []))
        puts = pd.DataFrame(raw.get("puts", []))
        calls["optionType"] = "call"
        puts["optionType"] = "put"
        calls["expiry"] = expiry.isoformat()
        puts["expiry"] = expiry.isoformat()
        frame = pd.concat([calls, puts], ignore_index=True)
        frame = _add_osi_columns(frame)
        frame = add_black_scholes_greeks_to_chain(
            frame,
            spot=spot,
            expiry=expiry,
            as_of=snapshot_date,
            r=risk_free_rate,
        )
        chain_frames.append(frame)
        quality_frames.append(frame)
        raw_by_expiry[expiry] = raw
        saved_expiries.append(expiry)
    return _ExpirySnapshotCollection(
        chain_frames=chain_frames,
        quality_frames=quality_frames,
        raw_by_expiry=raw_by_expiry,
        saved_expiries=saved_expiries,
        underlying_payload=underlying_payload,
    )


def _quote_quality_meta(quality_df: pd.DataFrame, *, snapshot_date: date) -> dict | None:
    total_contracts = int(len(quality_df))
    if total_contracts <= 0:
        return None
    quality = compute_quote_quality(quality_df, min_volume=0, min_open_interest=0, as_of=snapshot_date)
    missing_bid_ask = 0
    stale_quotes = 0
    spread_pcts: list[float] = []
    if not quality.empty:
        warnings = quality["quality_warnings"].tolist()
        missing_bid_ask = sum("quote_missing_bid_ask" in entry for entry in warnings if isinstance(entry, list))
        stale_quotes = sum("quote_stale" in entry for entry in warnings if isinstance(entry, list))
        spread_series = pd.to_numeric(quality["spread_pct"], errors="coerce")
        spread_series = spread_series.where(spread_series >= 0)
        spread_pcts.extend(spread_series.dropna().tolist())
    spread_median = float(np.nanmedian(spread_pcts)) if spread_pcts else None
    spread_worst = float(np.nanmax(spread_pcts)) if spread_pcts else None
    return {
        "contracts": total_contracts,
        "missing_bid_ask_count": int(missing_bid_ask),
        "missing_bid_ask_pct": float(missing_bid_ask / total_contracts),
        "spread_pct_median": spread_median,
        "spread_pct_worst": spread_worst,
        "stale_quotes": int(stale_quotes),
        "stale_pct": float(stale_quotes / total_contracts),
    }


def _snapshot_result(
    *,
    symbol: str,
    snapshot_date: date | None,
    expiries: int,
    status: str,
    error: str | None,
) -> SnapshotSymbolResult:
    return SnapshotSymbolResult(symbol=symbol, snapshot_date=snapshot_date, expiries=expiries, status=status, error=error)


def _snapshot_one_symbol(
    *,
    symbol: str,
    provider: MarketDataProvider,
    store,
    candle_store,
    spot_period: str,
    max_expiries: int | None, risk_free_rate: float,
    symbol_source: str, watchlists: list[str] | None,
    provider_name: str, provider_version: str | None, provider_params: dict | None,
) -> SnapshotSymbolResult:
    spot_resolution = _resolve_spot(symbol=symbol, candle_store=candle_store, provider=provider, spot_period=spot_period)
    if spot_resolution.error is not None or spot_resolution.spot is None:
        return _snapshot_result(
            symbol=symbol,
            snapshot_date=spot_resolution.data_date,
            expiries=0,
            status="no_spot",
            error=spot_resolution.error,
        )

    snapshot_date = spot_resolution.data_date or date.today()
    expiries = provider.list_option_expiries(symbol)
    if not expiries:
        return _snapshot_result(
            symbol=symbol,
            snapshot_date=snapshot_date,
            expiries=0,
            status="no_expiries",
            error="No listed option expiries",
        )
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    meta = _build_snapshot_meta(
        spot=spot_resolution.spot,
        snapshot_date=snapshot_date,
        spot_period=spot_period,
        risk_free_rate=risk_free_rate,
        symbol_source=symbol_source,
        watchlists=watchlists,
        provider_name=provider_name,
        provider_version=provider_version,
        provider_params=provider_params,
    )
    collection = _collect_expiry_snapshots(
        provider=provider,
        symbol=symbol,
        expiries=expiries,
        snapshot_date=snapshot_date,
        spot=spot_resolution.spot,
        risk_free_rate=risk_free_rate,
    )
    if not collection.saved_expiries:
        return _snapshot_result(symbol=symbol, snapshot_date=snapshot_date, expiries=0, status="ok", error=None)

    chain_df = pd.concat(collection.chain_frames, ignore_index=True) if collection.chain_frames else pd.DataFrame()
    quality_df = pd.concat(collection.quality_frames, ignore_index=True) if collection.quality_frames else pd.DataFrame()
    quote_quality = _quote_quality_meta(quality_df, snapshot_date=snapshot_date)
    if quote_quality is not None:
        meta["quote_quality"] = quote_quality
    if collection.underlying_payload is not None:
        meta["underlying"] = collection.underlying_payload
    store.save_day_snapshot(
        symbol,
        snapshot_date,
        chain=chain_df,
        expiries=collection.saved_expiries,
        raw_by_expiry=collection.raw_by_expiry or None,
        meta=meta,
    )
    return _snapshot_result(
        symbol=symbol,
        snapshot_date=snapshot_date,
        expiries=len(collection.saved_expiries),
        status="ok",
        error=None,
    )


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
    store = get_options_snapshot_store(cache_dir)
    candle_store = get_candle_store(candle_cache_dir, provider=provider)
    provider_name, provider_version, provider_params = _provider_details(provider)

    results: list[SnapshotSymbolResult] = []
    for symbol in symbols:
        sym = symbol.strip().upper()
        if not sym:
            continue
        try:
            results.append(
                _snapshot_one_symbol(
                    symbol=sym,
                    provider=provider,
                    store=store,
                    candle_store=candle_store,
                    spot_period=spot_period,
                    max_expiries=max_expiries,
                    risk_free_rate=risk_free_rate,
                    symbol_source=symbol_source,
                    watchlists=watchlists,
                    provider_name=provider_name,
                    provider_version=provider_version,
                    provider_params=provider_params,
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
