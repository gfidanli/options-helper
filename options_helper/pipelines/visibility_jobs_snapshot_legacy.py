from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import options_helper.cli_deps as cli_deps
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.data.candles import last_close
from options_helper.data.market_types import DataFetchError
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


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
    import numpy as np
    import pandas as pd

    quality_logger = resolve_quality_run_logger_fn(run_logger)
    portfolio = portfolio_loader(portfolio_path)
    store = active_snapshot_store_fn(snapshot_store_builder(cache_dir))
    provider = provider_builder()
    candle_store = candle_store_builder(candle_cache_dir, provider=provider)
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )

    want_full_chain = full_chain
    want_all_expiries = all_expiries
    use_watchlists = bool(watchlist) or all_watchlists

    watchlists_used: list[str] = []
    expiries_by_symbol: dict[str, set[date]] = {}
    symbols: list[str]
    messages: list[str] = []

    if use_watchlists:
        wl = watchlists_loader(watchlists_path)
        if all_watchlists:
            watchlists_used = sorted(wl.watchlists.keys())
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                persist_quality_results_fn(
                    quality_logger,
                    run_snapshot_quality_checks_fn(
                        snapshot_store=store,
                        snapshot_dates_by_symbol={},
                        skip_reason="no_symbols",
                    ),
                )
                return result_factory(
                    messages=[f"No watchlists in {watchlists_path}"],
                    dates_used=[],
                    symbols=[],
                    no_symbols=True,
                )
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise parameter_error_factory(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
            watchlists_used = sorted(set(watchlist))
    else:
        if not portfolio.positions:
            persist_quality_results_fn(
                quality_logger,
                run_snapshot_quality_checks_fn(
                    snapshot_store=store,
                    snapshot_dates_by_symbol={},
                    skip_reason="no_symbols",
                ),
            )
            return result_factory(
                messages=["No positions."],
                dates_used=[],
                symbols=[],
                no_symbols=True,
            )
        for p in portfolio.positions:
            expiries_by_symbol.setdefault(p.symbol, set()).add(p.expiry)
        symbols = sorted(expiries_by_symbol.keys())

    dates_used: set[date] = set()
    snapshot_dates_by_symbol: dict[str, date] = {}

    required_date: date | None = None
    if require_data_date is not None:
        spec = require_data_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_data_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise parameter_error_factory(
                f"Invalid --require-data-date/--require-data-tz: {exc}",
                param_hint="--require-data-date",
            ) from exc

    mode = "watchlists" if use_watchlists else "portfolio"
    messages.append(
        f"Snapshotting options chains for {len(symbols)} symbol(s) "
        f"({mode}, {'full-chain' if want_full_chain else 'windowed'})..."
    )

    effective_max_expiries = max_expiries
    if use_watchlists and not want_all_expiries and effective_max_expiries is None:
        effective_max_expiries = 2

    for symbol_value in symbols:
        history = candle_store.get_daily_history(symbol_value, period=spot_period)
        spot = last_close(history)
        data_date: date | None = history.index.max().date() if not history.empty else None
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

        if spot is None or spot <= 0:
            messages.append(f"[yellow]Warning:[/yellow] {symbol_value}: missing spot price; skipping snapshot.")
            continue

        if required_date is not None and data_date != required_date:
            got = "-" if data_date is None else data_date.isoformat()
            messages.append(
                f"[yellow]Warning:[/yellow] {symbol_value}: candle date {got} != required {required_date.isoformat()}; "
                "skipping snapshot to avoid mis-dated overwrite."
            )
            continue

        effective_snapshot_date = data_date or date.today()
        dates_used.add(effective_snapshot_date)

        strike_min = spot * (1.0 - window_pct)
        strike_max = spot * (1.0 + window_pct)

        meta = {
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

        expiries: list[date]
        if not use_watchlists and not want_all_expiries:
            expiries = sorted(expiries_by_symbol.get(symbol_value, set()))
        else:
            expiries = provider.list_option_expiries(symbol_value)
            if not expiries:
                messages.append(
                    f"[yellow]Warning:[/yellow] {symbol_value}: no listed option expiries; skipping snapshot."
                )
                continue
            if effective_max_expiries is not None:
                expiries = expiries[:effective_max_expiries]

        chain_frames: list[pd.DataFrame] = []
        quality_frames: list[pd.DataFrame] = []
        saved_expiries: list[date] = []
        raw_by_expiry: dict[date, dict[str, object]] = {}
        underlying_payload: dict[str, object] | None = None

        for exp in expiries:
            if want_full_chain:
                try:
                    raw = provider.get_options_chain_raw(symbol_value, exp)
                except DataFetchError as exc:
                    messages.append(
                        f"[yellow]Warning:[/yellow] {symbol_value} {exp.isoformat()}: {exc}; skipping snapshot."
                    )
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
                calls["expiry"] = exp.isoformat()
                puts["expiry"] = exp.isoformat()

                df = pd.concat([calls, puts], ignore_index=True)
                df = add_black_scholes_greeks_to_chain(
                    df,
                    spot=spot,
                    expiry=exp,
                    as_of=effective_snapshot_date,
                    r=risk_free_rate,
                )

                chain_frames.append(df)
                quality_frames.append(df)
                raw_by_expiry[exp] = raw
                saved_expiries.append(exp)
                messages.append(f"{symbol_value} {exp.isoformat()}: saved {len(df)} contracts (full)")
                continue

            try:
                chain = provider.get_options_chain(symbol_value, exp)
            except DataFetchError as exc:
                messages.append(
                    f"[yellow]Warning:[/yellow] {symbol_value} {exp.isoformat()}: {exc}; skipping snapshot."
                )
                continue

            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["optionType"] = "call"
            puts["optionType"] = "put"
            calls["expiry"] = exp.isoformat()
            puts["expiry"] = exp.isoformat()

            df = pd.concat([calls, puts], ignore_index=True)
            if not want_full_chain and "strike" in df.columns:
                df = df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]

            df = add_black_scholes_greeks_to_chain(
                df,
                spot=spot,
                expiry=exp,
                as_of=effective_snapshot_date,
                r=risk_free_rate,
            )

            quality_frames.append(df)

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
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

            chain_frames.append(df)
            saved_expiries.append(exp)
            messages.append(f"{symbol_value} {exp.isoformat()}: saved {len(df)} contracts")

        if not saved_expiries:
            continue

        chain_df = pd.concat(chain_frames, ignore_index=True) if chain_frames else pd.DataFrame()
        quality_df = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()

        total_contracts = int(len(quality_df))
        if total_contracts > 0:
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
                q_warn = quality["quality_warnings"].tolist()
                missing_bid_ask = sum("quote_missing_bid_ask" in w for w in q_warn if isinstance(w, list))
                stale_quotes = sum("quote_stale" in w for w in q_warn if isinstance(w, list))
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

        if want_full_chain and underlying_payload is not None:
            meta["underlying"] = underlying_payload

        store.save_day_snapshot(
            symbol_value,
            effective_snapshot_date,
            chain=chain_df,
            expiries=saved_expiries,
            raw_by_expiry=raw_by_expiry if raw_by_expiry else None,
            meta=meta,
        )
        snapshot_dates_by_symbol[symbol_value.upper()] = effective_snapshot_date

    if dates_used:
        days = ", ".join(sorted({d.isoformat() for d in dates_used}))
        messages.append(f"Snapshot complete. Data date(s): {days}.")

    persist_quality_results_fn(
        quality_logger,
        run_snapshot_quality_checks_fn(
            snapshot_store=store,
            snapshot_dates_by_symbol=snapshot_dates_by_symbol,
            skip_reason="no_snapshots_saved" if not snapshot_dates_by_symbol else None,
        ),
    )

    return result_factory(
        messages=messages,
        dates_used=sorted(dates_used),
        symbols=symbols,
        no_symbols=False,
    )
