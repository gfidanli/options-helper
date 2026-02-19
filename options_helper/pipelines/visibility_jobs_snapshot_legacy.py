from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import options_helper.cli_deps as cli_deps
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.data.candles import last_close
from options_helper.data.market_types import DataFetchError
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


@dataclass(frozen=True)
class _SnapshotRuntime:
    quality_logger: Any | None
    store: Any
    provider: Any
    candle_store: Any
    provider_name: str
    provider_version: str | None
    use_watchlists: bool
    want_full_chain: bool
    want_all_expiries: bool
    symbols: list[str]
    watchlists_used: list[str]
    expiries_by_symbol: dict[str, set[date]]
    required_date: date | None
    effective_max_expiries: int | None
    messages: list[str]


@dataclass(frozen=True)
class _SymbolSnapshotFrames:
    chain_frames: list[pd.DataFrame]
    quality_frames: list[pd.DataFrame]
    saved_expiries: list[date]
    raw_by_expiry: dict[date, dict[str, object]]
    underlying_payload: dict[str, object] | None


@dataclass(frozen=True)
class _SymbolSnapshotContext:
    spot: float
    effective_snapshot_date: date
    strike_min: float
    strike_max: float
    meta: dict[str, Any]
    expiries: list[date]


def _persist_snapshot_quality(
    *,
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_snapshot_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    store: Any,
    snapshot_dates_by_symbol: dict[str, date],
    skip_reason: str | None = None,
) -> None:
    persist_quality_results_fn(
        quality_logger,
        run_snapshot_quality_checks_fn(
            snapshot_store=store,
            snapshot_dates_by_symbol=snapshot_dates_by_symbol,
            skip_reason=skip_reason,
        ),
    )


def _no_symbols_result(
    *,
    message: str,
    result_factory: Callable[..., Any],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_snapshot_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    store: Any,
) -> Any:
    _persist_snapshot_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_snapshot_quality_checks_fn=run_snapshot_quality_checks_fn,
        quality_logger=quality_logger,
        store=store,
        snapshot_dates_by_symbol={},
        skip_reason="no_symbols",
    )
    return result_factory(
        messages=[message],
        dates_used=[],
        symbols=[],
        no_symbols=True,
    )


def _provider_metadata(provider: Any) -> tuple[str, str | None]:
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )
    return provider_name, provider_version


def _resolve_snapshot_symbols(
    *,
    portfolio: Any,
    symbol_watchlist_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    watchlists_loader: Callable[[Path], Any],
    use_watchlists: bool,
    parameter_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_snapshot_quality_checks_fn: Callable[..., list[Any]],
    quality_logger: Any | None,
    store: Any,
) -> tuple[list[str], dict[str, set[date]], list[str], Any | None]:
    watchlists_used: list[str] = []
    expiries_by_symbol: dict[str, set[date]] = {}
    if use_watchlists:
        wl = watchlists_loader(symbol_watchlist_path)
        if all_watchlists:
            watchlists_used = sorted(wl.watchlists.keys())
            symbols = sorted({value for values in wl.watchlists.values() for value in values})
            if not symbols:
                return [], {}, watchlists_used, _no_symbols_result(
                    message=f"No watchlists in {symbol_watchlist_path}",
                    result_factory=result_factory,
                    persist_quality_results_fn=persist_quality_results_fn,
                    run_snapshot_quality_checks_fn=run_snapshot_quality_checks_fn,
                    quality_logger=quality_logger,
                    store=store,
                )
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise parameter_error_factory(
                        f"Watchlist '{name}' is empty or missing in {symbol_watchlist_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
            watchlists_used = sorted(set(watchlist))
    else:
        if not portfolio.positions:
            return [], {}, [], _no_symbols_result(
                message="No positions.",
                result_factory=result_factory,
                persist_quality_results_fn=persist_quality_results_fn,
                run_snapshot_quality_checks_fn=run_snapshot_quality_checks_fn,
                quality_logger=quality_logger,
                store=store,
            )
        for position in portfolio.positions:
            expiries_by_symbol.setdefault(position.symbol, set()).add(position.expiry)
        symbols = sorted(expiries_by_symbol.keys())
    return symbols, expiries_by_symbol, watchlists_used, None


def _parse_required_date(
    require_data_date: str | None,
    require_data_tz: str,
    *,
    parameter_error_factory: Callable[..., Exception],
) -> date | None:
    if require_data_date is None:
        return None
    spec = require_data_date.strip().lower()
    try:
        if spec in {"today", "now"}:
            return datetime.now(ZoneInfo(require_data_tz)).date()
        return date.fromisoformat(spec)
    except Exception as exc:  # noqa: BLE001
        raise parameter_error_factory(
            f"Invalid --require-data-date/--require-data-tz: {exc}",
            param_hint="--require-data-date",
        ) from exc


def _load_symbol_spot_data(
    *,
    symbol_value: str,
    candle_store: Any,
    provider: Any,
    spot_period: str,
) -> tuple[float | None, date | None]:
    history = candle_store.get_daily_history(symbol_value, period=spot_period)
    spot = last_close(history)
    data_date = history.index.max().date() if not history.empty else None
    if spot is None:
        try:
            underlying = provider.get_underlying(symbol_value, period=spot_period, interval="1d")
            spot = underlying.last_price
            if data_date is None and underlying.history is not None and not underlying.history.empty:
                try:
                    data_date = underlying.history.index.max().date()
                except Exception:  # noqa: BLE001
                    pass
        except DataFetchError:
            spot = None
    return spot, data_date


def _base_snapshot_meta(
    *,
    spot: float,
    spot_period: str,
    want_full_chain: bool,
    want_all_expiries: bool,
    risk_free_rate: float,
    window_pct: float,
    strike_min: float,
    strike_max: float,
    effective_snapshot_date: date,
    mode: str,
    watchlists_used: list[str],
    provider_name: str,
    provider_version: str | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "spot": spot,
        "spot_period": spot_period,
        "full_chain": want_full_chain,
        "all_expiries": want_all_expiries,
        "risk_free_rate": risk_free_rate,
        "window_pct": None if want_full_chain else window_pct,
        "strike_min": None if want_full_chain else strike_min,
        "strike_max": None if want_full_chain else strike_max,
        "snapshot_date": effective_snapshot_date.isoformat(),
        "symbol_source": mode,
        "watchlists": watchlists_used,
        "provider": provider_name,
    }
    if provider_version:
        meta["provider_version"] = provider_version
    return meta


def _resolve_symbol_expiries(
    *,
    symbol_value: str,
    use_watchlists: bool,
    want_all_expiries: bool,
    expiries_by_symbol: dict[str, set[date]],
    provider: Any,
    effective_max_expiries: int | None,
    messages: list[str],
) -> list[date]:
    if not use_watchlists and not want_all_expiries:
        return sorted(expiries_by_symbol.get(symbol_value, set()))
    expiries = provider.list_option_expiries(symbol_value)
    if not expiries:
        messages.append(
            f"[yellow]Warning:[/yellow] {symbol_value}: no listed option expiries; skipping snapshot."
        )
        return []
    if effective_max_expiries is not None:
        expiries = expiries[:effective_max_expiries]
    return expiries


def _full_chain_frame(
    *,
    provider: Any,
    symbol_value: str,
    expiry: date,
    spot: float,
    effective_snapshot_date: date,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    raw = provider.get_options_chain_raw(symbol_value, expiry)
    underlying = raw.get("underlying")
    if not isinstance(underlying, dict):
        underlying = {}
    calls = pd.DataFrame(raw.get("calls", []))
    puts = pd.DataFrame(raw.get("puts", []))
    calls["optionType"] = "call"
    puts["optionType"] = "put"
    calls["expiry"] = expiry.isoformat()
    puts["expiry"] = expiry.isoformat()
    frame = pd.concat([calls, puts], ignore_index=True)
    frame = add_black_scholes_greeks_to_chain(
        frame,
        spot=spot,
        expiry=expiry,
        as_of=effective_snapshot_date,
        r=risk_free_rate,
    )
    return frame, raw, underlying


def _windowed_chain_frame(
    *,
    provider: Any,
    symbol_value: str,
    expiry: date,
    spot: float,
    strike_min: float,
    strike_max: float,
    effective_snapshot_date: date,
    risk_free_rate: float,
) -> pd.DataFrame:
    chain = provider.get_options_chain(symbol_value, expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["optionType"] = "call"
    puts["optionType"] = "put"
    calls["expiry"] = expiry.isoformat()
    puts["expiry"] = expiry.isoformat()
    frame = pd.concat([calls, puts], ignore_index=True)
    if "strike" in frame.columns:
        frame = frame[(frame["strike"] >= strike_min) & (frame["strike"] <= strike_max)]
    return add_black_scholes_greeks_to_chain(
        frame,
        spot=spot,
        expiry=expiry,
        as_of=effective_snapshot_date,
        r=risk_free_rate,
    )


def _trim_windowed_snapshot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "contractSymbol",
        "optionType",
        "expiry",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility",
        "inTheMoney",
        "bs_price",
        "bs_delta",
        "bs_gamma",
        "bs_theta_per_day",
        "bs_vega",
    ]
    selected = [column for column in keep if column in frame.columns]
    return frame[selected]


def _collect_symbol_snapshot_frames(
    *,
    runtime: _SnapshotRuntime,
    symbol_value: str,
    expiries: list[date],
    spot: float,
    strike_min: float,
    strike_max: float,
    effective_snapshot_date: date,
    risk_free_rate: float,
) -> _SymbolSnapshotFrames:
    chain_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    saved_expiries: list[date] = []
    raw_by_expiry: dict[date, dict[str, object]] = {}
    underlying_payload: dict[str, object] | None = None
    for expiry in expiries:
        if runtime.want_full_chain:
            try:
                frame, raw, underlying = _full_chain_frame(
                    provider=runtime.provider,
                    symbol_value=symbol_value,
                    expiry=expiry,
                    spot=spot,
                    effective_snapshot_date=effective_snapshot_date,
                    risk_free_rate=risk_free_rate,
                )
            except DataFetchError as exc:
                runtime.messages.append(
                    f"[yellow]Warning:[/yellow] {symbol_value} {expiry.isoformat()}: {exc}; skipping snapshot."
                )
                continue
            chain_frames.append(frame)
            quality_frames.append(frame)
            raw_by_expiry[expiry] = raw
            if underlying_payload is None or underlying:
                underlying_payload = underlying
            saved_expiries.append(expiry)
            runtime.messages.append(f"{symbol_value} {expiry.isoformat()}: saved {len(frame)} contracts (full)")
            continue
        try:
            frame = _windowed_chain_frame(
                provider=runtime.provider,
                symbol_value=symbol_value,
                expiry=expiry,
                spot=spot,
                strike_min=strike_min,
                strike_max=strike_max,
                effective_snapshot_date=effective_snapshot_date,
                risk_free_rate=risk_free_rate,
            )
        except DataFetchError as exc:
            runtime.messages.append(
                f"[yellow]Warning:[/yellow] {symbol_value} {expiry.isoformat()}: {exc}; skipping snapshot."
            )
            continue
        quality_frames.append(frame)
        chain_frames.append(_trim_windowed_snapshot_columns(frame))
        saved_expiries.append(expiry)
        runtime.messages.append(f"{symbol_value} {expiry.isoformat()}: saved {len(frame)} contracts")
    return _SymbolSnapshotFrames(
        chain_frames=chain_frames,
        quality_frames=quality_frames,
        saved_expiries=saved_expiries,
        raw_by_expiry=raw_by_expiry,
        underlying_payload=underlying_payload,
    )


def _append_quote_quality_meta(
    *,
    meta: dict[str, Any],
    quality_df: pd.DataFrame,
    effective_snapshot_date: date,
) -> None:
    total_contracts = int(len(quality_df))
    if total_contracts <= 0:
        return
    quality = compute_quote_quality(
        quality_df,
        min_volume=0,
        min_open_interest=0,
        as_of=effective_snapshot_date,
    )
    missing_bid_ask = 0
    stale_quotes = 0
    spread_pcts: list[float] = []
    if not quality.empty:
        warnings = quality["quality_warnings"].tolist()
        missing_bid_ask = sum("quote_missing_bid_ask" in value for value in warnings if isinstance(value, list))
        stale_quotes = sum("quote_stale" in value for value in warnings if isinstance(value, list))
        spread_series = pd.to_numeric(quality["spread_pct"], errors="coerce")
        spread_series = spread_series.where(spread_series >= 0)
        spread_pcts.extend(spread_series.dropna().tolist())
    spread_median = float(np.nanmedian(spread_pcts)) if spread_pcts else None
    spread_worst = float(np.nanmax(spread_pcts)) if spread_pcts else None
    meta["quote_quality"] = {
        "contracts": total_contracts,
        "missing_bid_ask_count": int(missing_bid_ask),
        "missing_bid_ask_pct": float(missing_bid_ask / total_contracts),
        "spread_pct_median": spread_median,
        "spread_pct_worst": spread_worst,
        "stale_quotes": int(stale_quotes),
        "stale_pct": float(stale_quotes / total_contracts),
    }


def _prepare_symbol_snapshot_context(
    *,
    runtime: _SnapshotRuntime,
    symbol_value: str,
    spot_period: str,
    window_pct: float,
    risk_free_rate: float,
) -> _SymbolSnapshotContext | None:
    spot, data_date = _load_symbol_spot_data(
        symbol_value=symbol_value,
        candle_store=runtime.candle_store,
        provider=runtime.provider,
        spot_period=spot_period,
    )
    if spot is None or spot <= 0:
        runtime.messages.append(
            f"[yellow]Warning:[/yellow] {symbol_value}: missing spot price; skipping snapshot."
        )
        return None
    if runtime.required_date is not None and data_date != runtime.required_date:
        got = "-" if data_date is None else data_date.isoformat()
        runtime.messages.append(
            f"[yellow]Warning:[/yellow] {symbol_value}: candle date {got} != required {runtime.required_date.isoformat()}; "
            "skipping snapshot to avoid mis-dated overwrite."
        )
        return None

    effective_snapshot_date = data_date or date.today()
    strike_min = spot * (1.0 - window_pct)
    strike_max = spot * (1.0 + window_pct)
    mode = "watchlists" if runtime.use_watchlists else "portfolio"
    meta = _base_snapshot_meta(
        spot=spot,
        spot_period=spot_period,
        want_full_chain=runtime.want_full_chain,
        want_all_expiries=runtime.want_all_expiries,
        risk_free_rate=risk_free_rate,
        window_pct=window_pct,
        strike_min=strike_min,
        strike_max=strike_max,
        effective_snapshot_date=effective_snapshot_date,
        mode=mode,
        watchlists_used=runtime.watchlists_used,
        provider_name=runtime.provider_name,
        provider_version=runtime.provider_version,
    )
    expiries = _resolve_symbol_expiries(
        symbol_value=symbol_value,
        use_watchlists=runtime.use_watchlists,
        want_all_expiries=runtime.want_all_expiries,
        expiries_by_symbol=runtime.expiries_by_symbol,
        provider=runtime.provider,
        effective_max_expiries=runtime.effective_max_expiries,
        messages=runtime.messages,
    )
    if not expiries:
        return None

    return _SymbolSnapshotContext(
        spot=float(spot),
        effective_snapshot_date=effective_snapshot_date,
        strike_min=strike_min,
        strike_max=strike_max,
        meta=meta,
        expiries=expiries,
    )


def _save_symbol_snapshot(
    *,
    runtime: _SnapshotRuntime,
    symbol_value: str,
    context: _SymbolSnapshotContext,
    frames: _SymbolSnapshotFrames,
) -> None:
    chain_df = pd.concat(frames.chain_frames, ignore_index=True) if frames.chain_frames else pd.DataFrame()
    quality_df = pd.concat(frames.quality_frames, ignore_index=True) if frames.quality_frames else pd.DataFrame()
    _append_quote_quality_meta(
        meta=context.meta,
        quality_df=quality_df,
        effective_snapshot_date=context.effective_snapshot_date,
    )
    if runtime.want_full_chain and frames.underlying_payload is not None:
        context.meta["underlying"] = frames.underlying_payload
    runtime.store.save_day_snapshot(
        symbol_value,
        context.effective_snapshot_date,
        chain=chain_df,
        expiries=frames.saved_expiries,
        raw_by_expiry=frames.raw_by_expiry if frames.raw_by_expiry else None,
        meta=context.meta,
    )


def _process_symbol_snapshot(
    *,
    runtime: _SnapshotRuntime,
    symbol_value: str,
    spot_period: str,
    window_pct: float,
    risk_free_rate: float,
) -> date | None:
    context = _prepare_symbol_snapshot_context(
        runtime=runtime,
        symbol_value=symbol_value,
        spot_period=spot_period,
        window_pct=window_pct,
        risk_free_rate=risk_free_rate,
    )
    if context is None:
        return None

    frames = _collect_symbol_snapshot_frames(
        runtime=runtime,
        symbol_value=symbol_value,
        expiries=context.expiries,
        spot=context.spot,
        strike_min=context.strike_min,
        strike_max=context.strike_max,
        effective_snapshot_date=context.effective_snapshot_date,
        risk_free_rate=risk_free_rate,
    )
    if not frames.saved_expiries:
        return None

    _save_symbol_snapshot(
        runtime=runtime,
        symbol_value=symbol_value,
        context=context,
        frames=frames,
    )
    return context.effective_snapshot_date


def _prepare_snapshot_runtime(params: dict[str, Any]) -> tuple[_SnapshotRuntime | None, Any | None]:
    quality_logger = params["resolve_quality_run_logger_fn"](params["run_logger"])
    portfolio = params["portfolio_loader"](params["portfolio_path"])
    store = params["active_snapshot_store_fn"](params["snapshot_store_builder"](params["cache_dir"]))
    provider = params["provider_builder"]()
    candle_store = params["candle_store_builder"](params["candle_cache_dir"], provider=provider)
    provider_name, provider_version = _provider_metadata(provider)

    use_watchlists = bool(params["watchlist"]) or bool(params["all_watchlists"])
    symbols, expiries_by_symbol, watchlists_used, early_result = _resolve_snapshot_symbols(
        portfolio=portfolio,
        symbol_watchlist_path=params["watchlists_path"],
        watchlist=params["watchlist"],
        all_watchlists=params["all_watchlists"],
        watchlists_loader=params["watchlists_loader"],
        use_watchlists=use_watchlists,
        parameter_error_factory=params["parameter_error_factory"],
        result_factory=params["result_factory"],
        persist_quality_results_fn=params["persist_quality_results_fn"],
        run_snapshot_quality_checks_fn=params["run_snapshot_quality_checks_fn"],
        quality_logger=quality_logger,
        store=store,
    )
    if early_result is not None:
        return None, early_result
    required_date = _parse_required_date(
        params["require_data_date"],
        params["require_data_tz"],
        parameter_error_factory=params["parameter_error_factory"],
    )
    mode = "watchlists" if use_watchlists else "portfolio"
    messages = [
        f"Snapshotting options chains for {len(symbols)} symbol(s) "
        f"({mode}, {'full-chain' if params['full_chain'] else 'windowed'})..."
    ]
    effective_max_expiries = params["max_expiries"]
    if use_watchlists and not params["all_expiries"] and effective_max_expiries is None:
        effective_max_expiries = 2
    return (
        _SnapshotRuntime(
            quality_logger=quality_logger,
            store=store,
            provider=provider,
            candle_store=candle_store,
            provider_name=provider_name,
            provider_version=provider_version,
            use_watchlists=use_watchlists,
            want_full_chain=bool(params["full_chain"]),
            want_all_expiries=bool(params["all_expiries"]),
            symbols=symbols,
            watchlists_used=watchlists_used,
            expiries_by_symbol=expiries_by_symbol,
            required_date=required_date,
            effective_max_expiries=effective_max_expiries,
            messages=messages,
        ),
        None,
    )


def run_snapshot_options_job_impl(
    *,
    portfolio_path: Path,
    cache_dir: Path,
    candle_cache_dir: Path,
    window_pct: float,
    spot_period: str,
    require_data_date: str | None,
    require_data_tz: str,
    watchlists_path: Path,
    watchlist: list[str],
    all_watchlists: bool,
    all_expiries: bool,
    full_chain: bool,
    max_expiries: int | None,
    risk_free_rate: float,
    provider_builder: Callable[[], Any] = cli_deps.build_provider,
    snapshot_store_builder: Callable[[Path], Any] = cli_deps.build_snapshot_store,
    candle_store_builder: Callable[..., Any] = cli_deps.build_candle_store,
    portfolio_loader: Callable[[Path], Any] = load_portfolio,
    watchlists_loader: Callable[[Path], Any] = load_watchlists,
    run_logger: Any | None = None,
    resolve_quality_run_logger_fn: Callable[[Any | None], Any | None],
    persist_quality_results_fn: Callable[[Any | None, list[Any]], None],
    run_snapshot_quality_checks_fn: Callable[..., list[Any]],
    active_snapshot_store_fn: Callable[[Any], Any],
    parameter_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
) -> Any:
    params = dict(locals())
    runtime, early_result = _prepare_snapshot_runtime(params)
    if early_result is not None:
        return early_result
    if runtime is None:
        raise RuntimeError("snapshot runtime unavailable")

    dates_used: set[date] = set()
    snapshot_dates_by_symbol: dict[str, date] = {}
    for symbol_value in runtime.symbols:
        snapshot_date = _process_symbol_snapshot(
            runtime=runtime,
            symbol_value=symbol_value,
            spot_period=spot_period,
            window_pct=window_pct,
            risk_free_rate=risk_free_rate,
        )
        if snapshot_date is None:
            continue
        dates_used.add(snapshot_date)
        snapshot_dates_by_symbol[symbol_value.upper()] = snapshot_date

    if dates_used:
        days = ", ".join(sorted(value.isoformat() for value in dates_used))
        runtime.messages.append(f"Snapshot complete. Data date(s): {days}.")

    _persist_snapshot_quality(
        persist_quality_results_fn=persist_quality_results_fn,
        run_snapshot_quality_checks_fn=run_snapshot_quality_checks_fn,
        quality_logger=runtime.quality_logger,
        store=runtime.store,
        snapshot_dates_by_symbol=snapshot_dates_by_symbol,
        skip_reason="no_snapshots_saved" if not snapshot_dates_by_symbol else None,
    )
    return result_factory(
        messages=runtime.messages,
        dates_used=sorted(dates_used),
        symbols=runtime.symbols,
        no_symbols=False,
    )
