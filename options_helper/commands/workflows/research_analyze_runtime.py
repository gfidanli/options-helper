from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import typer
from rich.console import Console

from options_helper.commands import workflows_legacy as legacy


@dataclass
class _AnalyzeSymbolData:
    history_by_symbol: dict[str, Any]
    last_price_by_symbol: dict[str, float | None]
    as_of_by_symbol: dict[str, Any]
    next_earnings_by_symbol: dict[str, Any]
    snapshot_day_by_symbol: dict[str, Any]
    chain_cache: dict[tuple[str, Any], object]


@dataclass
class _AnalyzeResults:
    single_metrics: list[Any]
    all_metrics: list[Any]
    multi_leg_summaries: list[Any]
    advice_by_id: dict[str, Any]
    offline_missing: list[str]


@dataclass
class _MultiLegState:
    leg_metrics: list[Any]
    net_mark_total: float
    net_mark_ready: bool
    dte_vals: list[int]
    low_oi: bool
    low_vol: bool
    bad_spread: bool
    quote_flags: bool


def _validate_analyze_interval(interval: str) -> None:
    if interval != "1d":
        raise typer.BadParameter("Only --interval 1d is supported for now (cache uses daily candles).")


def _load_online_history(*, candle_store: Any, sym: str, period: str, console: Console, pd: Any) -> Any:
    try:
        return candle_store.get_daily_history(sym, period=period)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Candle cache error:[/red] {sym}: {exc}")
        return pd.DataFrame()


def _load_offline_history_and_snapshot(
    *,
    candle_store: Any,
    snapshot_store: Any,
    sym: str,
    as_of: str,
    console: Console,
    pd: Any,
) -> tuple[Any, Any, Any]:
    try:
        history = candle_store.load(sym)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {sym}: {exc}")
        history = pd.DataFrame()

    try:
        snapshot_date = snapshot_store.resolve_date(sym, as_of)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Warning:[/yellow] snapshot date resolve failed for {sym}: {exc}")
        snapshot_date = None

    if snapshot_date is None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
        last_ts = history.index.max()
        if last_ts is not None and not pd.isna(last_ts):
            snapshot_date = last_ts.date()

    if snapshot_date is not None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
        history = history.loc[history.index <= pd.Timestamp(snapshot_date)]

    snap_df = pd.DataFrame()
    if snapshot_date is not None:
        try:
            snap_df = snapshot_store.load_day(sym, snapshot_date)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] snapshot load failed for {sym}: {exc}")
            snap_df = pd.DataFrame()
    return history, snapshot_date, snap_df


def _resolve_online_as_of(history: Any, pd: Any) -> Any:
    as_of_date = None
    if not history.empty and isinstance(history.index, pd.DatetimeIndex):
        last_ts = history.index.max()
        if last_ts is not None and not pd.isna(last_ts):
            as_of_date = last_ts.date()
    return as_of_date


def _build_analyze_symbol_data(
    *,
    symbols: set[str],
    offline: bool,
    as_of: str,
    period: str,
    candle_store: Any,
    snapshot_store: Any,
    earnings_store: Any,
    workflows_pkg: Any,
    console: Console,
    pd: Any,
) -> _AnalyzeSymbolData:
    symbol_data = _AnalyzeSymbolData(
        history_by_symbol={},
        last_price_by_symbol={},
        as_of_by_symbol={},
        next_earnings_by_symbol={},
        snapshot_day_by_symbol={},
        chain_cache={},
    )
    for sym in sorted(symbols):
        if offline:
            history, snapshot_date, snap_df = _load_offline_history_and_snapshot(
                candle_store=candle_store,
                snapshot_store=snapshot_store,
                sym=sym,
                as_of=as_of,
                console=console,
                pd=pd,
            )
            symbol_data.snapshot_day_by_symbol[sym] = snap_df
            symbol_data.as_of_by_symbol[sym] = snapshot_date
            symbol_data.last_price_by_symbol[sym] = (
                legacy.close_asof(history, snapshot_date) if snapshot_date is not None else legacy.last_close(history)
            )
        else:
            history = _load_online_history(candle_store=candle_store, sym=sym, period=period, console=console, pd=pd)
            symbol_data.as_of_by_symbol[sym] = _resolve_online_as_of(history, pd)
            symbol_data.last_price_by_symbol[sym] = legacy.last_close(history)
        symbol_data.history_by_symbol[sym] = history
        symbol_data.next_earnings_by_symbol[sym] = workflows_pkg.safe_next_earnings_date(earnings_store, sym)
    return symbol_data


def _resolve_offline_snapshot_row(
    *,
    position_id: str,
    symbol: str,
    expiry: date,
    strike: float,
    option_type: str,
    as_of_by_symbol: dict[str, Any],
    snapshot_day_by_symbol: dict[str, Any],
    offline_missing: list[str],
    warn_missing_as_of: bool,
    warn_missing_day: bool,
    pd: Any,
) -> tuple[dict[str, Any], bool, bool]:
    snap_date = as_of_by_symbol.get(symbol)
    df_snap = snapshot_day_by_symbol.get(symbol, pd.DataFrame())
    if snap_date is None:
        if not warn_missing_as_of:
            offline_missing.append(f"{position_id}: missing offline as-of date for {symbol}")
            warn_missing_as_of = True
        return {}, warn_missing_as_of, warn_missing_day
    if df_snap.empty:
        if not warn_missing_day:
            offline_missing.append(
                f"{position_id}: missing snapshot day data for {symbol} (as-of {snap_date.isoformat()})"
            )
            warn_missing_day = True
        return {}, warn_missing_as_of, warn_missing_day
    row = legacy.find_snapshot_row(
        df_snap,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    )
    if row is None:
        offline_missing.append(
            f"{position_id}: missing snapshot row for {symbol} {expiry.isoformat()} "
            f"{option_type} {strike:g} (as-of {snap_date.isoformat()})"
        )
        return {}, warn_missing_as_of, warn_missing_day
    return row, warn_missing_as_of, warn_missing_day


def _resolve_online_snapshot_row(
    *,
    provider: Any,
    chain_cache: dict[tuple[str, Any], object],
    symbol: str,
    expiry: date,
    option_type: str,
    strike: float,
) -> dict[str, Any]:
    if provider is None:
        return {}
    key = (symbol, expiry)
    chain = chain_cache.get(key)
    if chain is None:
        chain = provider.get_options_chain(symbol, expiry)
        chain_cache[key] = chain
    df_chain = chain.calls if option_type == "call" else chain.puts
    row = legacy.contract_row_by_strike(df_chain, strike)
    return row if row is not None else {}


def _update_multileg_state(*, state: _MultiLegState, metrics: Any, signed_contracts: float, risk_profile: Any) -> None:
    state.leg_metrics.append(metrics)
    if metrics.mark is None:
        state.net_mark_ready = False
    else:
        state.net_mark_total += metrics.mark * signed_contracts * 100.0
    if metrics.dte is not None:
        state.dte_vals.append(metrics.dte)
    if metrics.open_interest is not None and metrics.open_interest < risk_profile.min_open_interest:
        state.low_oi = True
    if metrics.volume is not None and metrics.volume < risk_profile.min_volume:
        state.low_vol = True
    if metrics.execution_quality == "bad":
        state.bad_spread = True
    if metrics.quality_warnings:
        state.quote_flags = True


def _build_multileg_summary(*, position: Any, state: _MultiLegState) -> Any:
    warnings: list[str] = []
    net_mark = state.net_mark_total if state.net_mark_ready else None
    if net_mark is None:
        warnings.append("missing_leg_marks")
    if position.net_debit is None:
        warnings.append("missing_net_debit")
    if state.low_oi:
        warnings.append("low_open_interest_leg")
    if state.low_vol:
        warnings.append("low_volume_leg")
    if state.bad_spread:
        warnings.append("bad_spread_leg")
    if state.quote_flags:
        warnings.append("quote_quality_leg")

    net_pnl_abs = net_pnl_pct = None
    if net_mark is not None and position.net_debit is not None:
        net_pnl_abs = net_mark - position.net_debit
        if position.net_debit > 0:
            net_pnl_pct = net_pnl_abs / position.net_debit

    return legacy.MultiLegSummary(
        position=position,
        leg_metrics=state.leg_metrics,
        net_mark=net_mark,
        net_pnl_abs=net_pnl_abs,
        net_pnl_pct=net_pnl_pct,
        dte_min=min(state.dte_vals) if state.dte_vals else None,
        dte_max=max(state.dte_vals) if state.dte_vals else None,
        warnings=warnings,
    )


def _process_multileg_position(
    *,
    position: Any,
    portfolio: Any,
    workflows_pkg: Any,
    symbol_data: _AnalyzeSymbolData,
    provider: Any,
    offline: bool,
    offline_missing: list[str],
    pd: Any,
) -> tuple[Any, list[Any]]:
    state = _MultiLegState([], 0.0, True, [], False, False, False, False)
    missing_as_of_warned = False
    missing_day_warned = False
    for idx, leg in enumerate(position.legs, start=1):
        if offline:
            snapshot_row, missing_as_of_warned, missing_day_warned = _resolve_offline_snapshot_row(
                position_id=position.id,
                symbol=position.symbol,
                expiry=leg.expiry,
                strike=leg.strike,
                option_type=leg.option_type,
                as_of_by_symbol=symbol_data.as_of_by_symbol,
                snapshot_day_by_symbol=symbol_data.snapshot_day_by_symbol,
                offline_missing=offline_missing,
                warn_missing_as_of=missing_as_of_warned,
                warn_missing_day=missing_day_warned,
                pd=pd,
            )
        else:
            snapshot_row = _resolve_online_snapshot_row(
                provider=provider,
                chain_cache=symbol_data.chain_cache,
                symbol=position.symbol,
                expiry=leg.expiry,
                option_type=leg.option_type,
                strike=leg.strike,
            )
        leg_position = legacy.Position(
            id=f"{position.id}:leg{idx}",
            symbol=position.symbol,
            option_type=leg.option_type,
            expiry=leg.expiry,
            strike=leg.strike,
            contracts=leg.contracts,
            cost_basis=0.0,
            opened_at=position.opened_at,
        )
        metrics = workflows_pkg._position_metrics(
            provider,
            leg_position,
            risk_profile=portfolio.risk_profile,
            underlying_history=symbol_data.history_by_symbol.get(position.symbol, pd.DataFrame()),
            underlying_last_price=symbol_data.last_price_by_symbol.get(position.symbol),
            as_of=symbol_data.as_of_by_symbol.get(position.symbol),
            next_earnings_date=symbol_data.next_earnings_by_symbol.get(position.symbol),
            snapshot_row=snapshot_row,
            include_pnl=False,
            contract_sign=1 if leg.side == "long" else -1,
        )
        _update_multileg_state(
            state=state,
            metrics=metrics,
            signed_contracts=leg.signed_contracts,
            risk_profile=portfolio.risk_profile,
        )
    return _build_multileg_summary(position=position, state=state), state.leg_metrics


def _process_single_position(
    *,
    position: Any,
    portfolio: Any,
    workflows_pkg: Any,
    symbol_data: _AnalyzeSymbolData,
    provider: Any,
    offline: bool,
    offline_missing: list[str],
    pd: Any,
) -> tuple[Any, Any]:
    snapshot_row = None
    if offline:
        snapshot_row, _, _ = _resolve_offline_snapshot_row(
            position_id=position.id,
            symbol=position.symbol,
            expiry=position.expiry,
            strike=position.strike,
            option_type=position.option_type,
            as_of_by_symbol=symbol_data.as_of_by_symbol,
            snapshot_day_by_symbol=symbol_data.snapshot_day_by_symbol,
            offline_missing=offline_missing,
            warn_missing_as_of=False,
            warn_missing_day=False,
            pd=pd,
        )
    metrics = workflows_pkg._position_metrics(
        provider,
        position,
        risk_profile=portfolio.risk_profile,
        underlying_history=symbol_data.history_by_symbol.get(position.symbol, pd.DataFrame()),
        underlying_last_price=symbol_data.last_price_by_symbol.get(position.symbol),
        as_of=symbol_data.as_of_by_symbol.get(position.symbol),
        next_earnings_date=symbol_data.next_earnings_by_symbol.get(position.symbol),
        snapshot_row=snapshot_row,
    )
    return metrics, legacy.advise(metrics, portfolio)
