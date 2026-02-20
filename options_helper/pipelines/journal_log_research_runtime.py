from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from options_helper.analysis.confluence import ConfluenceScore, score_confluence
from options_helper.analysis.research import (
    Direction,
    OptionCandidate,
    TradeLevels,
    UnderlyingSetup,
    VolatilityContext,
    analyze_underlying,
    build_confluence_inputs,
    choose_expiry,
    compute_volatility_context,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.data.earnings import safe_next_earnings_date
from options_helper.data.market_types import DataFetchError
from options_helper.models import OptionType

@dataclass
class _ResearchSelection:
    warnings: list[str] = field(default_factory=list)
    short_pick: OptionCandidate | None = None
    long_pick: OptionCandidate | None = None
    vol_context: VolatilityContext | None = None
    confluence_score: ConfluenceScore | None = None

def _choose_long_expiry_fallback(
    expiry_strings: list[str],
    *,
    expiry_as_of: date,
    min_dte: int,
) -> date | None:
    parsed: list[tuple[int, date]] = []
    for raw in expiry_strings:
        try:
            expiry = date.fromisoformat(raw)
        except ValueError:
            continue
        parsed.append(((expiry - expiry_as_of).days, expiry))
    parsed = [entry for entry in parsed if entry[0] >= min_dte]
    if not parsed:
        return None
    _, long_expiry = max(parsed, key=lambda entry: entry[0])
    return long_expiry


def _choose_research_expiries(
    expiry_strings: list[str],
    *,
    expiry_as_of: date,
    args: Any,
) -> tuple[date | None, date | None]:
    short_expiry = choose_expiry(
        expiry_strings,
        min_dte=args.research_short_min_dte,
        max_dte=args.research_short_max_dte,
        target_dte=60,
        today=expiry_as_of,
    )
    long_expiry = choose_expiry(
        expiry_strings,
        min_dte=args.research_long_min_dte,
        max_dte=args.research_long_max_dte,
        target_dte=540,
        today=expiry_as_of,
    )
    if long_expiry is None:
        long_expiry = _choose_long_expiry_fallback(
            expiry_strings,
            expiry_as_of=expiry_as_of,
            min_dte=args.research_long_min_dte,
        )
    return short_expiry, long_expiry


def _select_candidate_for_expiry(
    *,
    provider: Any,
    symbol: str,
    expiry: date,
    option_type: OptionType,
    setup: UnderlyingSetup,
    history: pd.DataFrame,
    derived_history: pd.DataFrame,
    vol_context: VolatilityContext | None,
    target_delta: float,
    risk_profile: Any,
    as_of: date,
    next_earnings_date: date | None,
    args: Any,
) -> tuple[OptionCandidate | None, VolatilityContext | None]:
    chain = provider.get_options_chain(symbol, expiry)
    if vol_context is None:
        vol_context = compute_volatility_context(
            history=history,
            spot=setup.spot,
            calls=chain.calls,
            puts=chain.puts,
            derived_history=derived_history,
        )
    option_df = chain.calls if option_type == "call" else chain.puts
    candidate = select_option_candidate(
        option_df,
        symbol=symbol,
        option_type=option_type,
        expiry=expiry,
        spot=setup.spot,
        target_delta=target_delta,
        window_pct=args.research_window_pct,
        min_open_interest=risk_profile.min_open_interest,
        min_volume=risk_profile.min_volume,
        as_of=as_of,
        next_earnings_date=next_earnings_date,
        earnings_warn_days=risk_profile.earnings_warn_days,
        earnings_avoid_days=risk_profile.earnings_avoid_days,
        include_bad_quotes=args.research_include_bad_quotes,
    )
    return candidate, vol_context


def _pick_directional_candidates(
    runtime: Any,
    *,
    args: Any,
    symbol: str,
    setup: UnderlyingSetup,
    history: pd.DataFrame,
    short_expiry: date | None,
    long_expiry: date | None,
    risk_profile: Any,
    next_earnings_date: date | None,
    expiry_as_of: date,
    derived_store: Any,
) -> tuple[list[str], OptionCandidate | None, OptionCandidate | None, VolatilityContext | None]:
    option_type: OptionType = "call" if setup.direction == Direction.BULLISH else "put"
    warnings: list[str] = []
    short_pick = None
    long_pick = None
    vol_context = None
    derived_history = derived_store.load(symbol)
    if short_expiry is not None and runtime.provider is not None:
        target_delta = 0.40 if option_type == "call" else -0.40
        short_pick, vol_context = _select_candidate_for_expiry(
            provider=runtime.provider,
            symbol=symbol,
            expiry=short_expiry,
            option_type=option_type,
            setup=setup,
            history=history,
            derived_history=derived_history,
            vol_context=vol_context,
            target_delta=target_delta,
            risk_profile=risk_profile,
            as_of=expiry_as_of,
            next_earnings_date=next_earnings_date,
            args=args,
        )
    else:
        warnings.append("no_short_expiry")
    if long_expiry is not None and runtime.provider is not None:
        target_delta = 0.70 if option_type == "call" else -0.70
        long_pick, vol_context = _select_candidate_for_expiry(
            provider=runtime.provider,
            symbol=symbol,
            expiry=long_expiry,
            option_type=option_type,
            setup=setup,
            history=history,
            derived_history=derived_history,
            vol_context=vol_context,
            target_delta=target_delta,
            risk_profile=risk_profile,
            as_of=expiry_as_of,
            next_earnings_date=next_earnings_date,
            args=args,
        )
    else:
        warnings.append("no_long_expiry")
    return warnings, short_pick, long_pick, vol_context


def _evaluate_research_symbol(
    runtime: Any,
    *,
    args: Any,
    symbol: str,
    setup: UnderlyingSetup,
    history: pd.DataFrame,
    ext_pct: float | None,
    as_of_date: date | None,
    risk_profile: Any,
    next_earnings_date: date | None,
    confluence_cfg: Any,
    derived_store: Any,
) -> _ResearchSelection:
    selection = _ResearchSelection()
    if setup.spot is None:
        selection.warnings.append("no_spot_price")
        return selection
    expiry_strings = [exp.isoformat() for exp in runtime.provider.list_option_expiries(symbol)] if runtime.provider else []
    if not expiry_strings:
        selection.warnings.append("no_listed_expiries")
        return selection
    expiry_as_of = as_of_date or date.today()
    short_expiry, long_expiry = _choose_research_expiries(expiry_strings, expiry_as_of=expiry_as_of, args=args)
    if setup.direction == Direction.NEUTRAL:
        selection.confluence_score = score_confluence(
            build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None),
            cfg=confluence_cfg,
        )
        return selection
    warnings, short_pick, long_pick, vol_context = _pick_directional_candidates(
        runtime,
        args=args,
        symbol=symbol,
        setup=setup,
        history=history,
        short_expiry=short_expiry,
        long_expiry=long_expiry,
        risk_profile=risk_profile,
        next_earnings_date=next_earnings_date,
        expiry_as_of=expiry_as_of,
        derived_store=derived_store,
    )
    selection.warnings.extend(warnings)
    selection.short_pick = short_pick
    selection.long_pick = long_pick
    selection.vol_context = vol_context
    selection.confluence_score = score_confluence(
        build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=vol_context),
        cfg=confluence_cfg,
    )
    return selection


def _build_research_payload(
    *,
    as_of_date: date | None,
    setup: UnderlyingSetup,
    ext_pct: float | None,
    levels: TradeLevels,
    selection: _ResearchSelection,
    confluence_score: ConfluenceScore | None,
    iso_date_fn: Callable[[date | None], str | None],
    clean_float_fn: Callable[[float | None], float | None],
    levels_payload_fn: Callable[[TradeLevels], dict | None],
    vol_context_payload_fn: Callable[[VolatilityContext | None], dict | None],
    confluence_payload_fn: Callable[[ConfluenceScore | None], dict | None],
    option_candidate_payload_fn: Callable[[OptionCandidate | None], dict | None],
) -> dict[str, object]:
    return {
        "as_of": iso_date_fn(as_of_date),
        "setup": {
            "direction": setup.direction.value,
            "spot": clean_float_fn(setup.spot),
            "reasons": list(setup.reasons),
            "daily_rsi": clean_float_fn(setup.daily_rsi),
            "daily_stoch_rsi": clean_float_fn(setup.daily_stoch_rsi),
            "weekly_rsi": clean_float_fn(setup.weekly_rsi),
            "weekly_breakout": setup.weekly_breakout,
        },
        "extension_percentile": clean_float_fn(ext_pct),
        "levels": levels_payload_fn(levels),
        "vol_context": vol_context_payload_fn(selection.vol_context),
        "confluence": confluence_payload_fn(confluence_score),
        "short_candidate": option_candidate_payload_fn(selection.short_pick),
        "long_candidate": option_candidate_payload_fn(selection.long_pick),
        "warnings": selection.warnings,
    }


def _build_research_event(
    runtime: Any,
    *,
    args: Any,
    symbol: str,
    cache: Any,
    risk_profile: Any,
    confluence_cfg: Any,
    derived_store: Any,
    history_as_of_date_fn: Callable[[pd.DataFrame], date | None],
    iso_date_fn: Callable[[date | None], str | None],
    clean_float_fn: Callable[[float | None], float | None],
    levels_payload_fn: Callable[[TradeLevels], dict | None],
    vol_context_payload_fn: Callable[[VolatilityContext | None], dict | None],
    confluence_payload_fn: Callable[[ConfluenceScore | None], dict | None],
    option_candidate_payload_fn: Callable[[OptionCandidate | None], dict | None],
) -> Any:
    history = cache.history_by_symbol.get(symbol, pd.DataFrame())
    setup = cache.setup_by_symbol.get(symbol) or analyze_underlying(symbol, history=history, risk_profile=risk_profile)
    ext_pct = cache.extension_pct_by_symbol.get(symbol)
    as_of_date = history_as_of_date_fn(history)
    next_earnings_date = safe_next_earnings_date(runtime.earnings_store, symbol)
    levels = suggest_trade_levels(setup, history=history, risk_profile=risk_profile)
    selection = _evaluate_research_symbol(
        runtime,
        args=args,
        symbol=symbol,
        setup=setup,
        history=history,
        ext_pct=ext_pct,
        as_of_date=as_of_date,
        risk_profile=risk_profile,
        next_earnings_date=next_earnings_date,
        confluence_cfg=confluence_cfg,
        derived_store=derived_store,
    )
    confluence_score = selection.confluence_score or cache.pre_confluence_by_symbol.get(symbol)
    payload = _build_research_payload(
        as_of_date=as_of_date,
        setup=setup,
        ext_pct=ext_pct,
        levels=levels,
        selection=selection,
        confluence_score=confluence_score,
        iso_date_fn=iso_date_fn,
        clean_float_fn=clean_float_fn,
        levels_payload_fn=levels_payload_fn,
        vol_context_payload_fn=vol_context_payload_fn,
        confluence_payload_fn=confluence_payload_fn,
        option_candidate_payload_fn=option_candidate_payload_fn,
    )
    from options_helper.data.journal import SignalContext, SignalEvent

    return SignalEvent(
        date=as_of_date or date.today(),
        symbol=symbol,
        context=SignalContext.RESEARCH,
        payload=payload,
        snapshot_date=None,
        contract_symbol=None,
    )


def _collect_research_events(
    runtime: Any,
    *,
    args: Any,
    load_research_configs_fn: Callable[[Any], tuple[Any, Any]],
    resolve_research_symbols_fn: Callable[[Any, Any], list[str]],
    build_research_cache_fn: Callable[..., Any],
    rank_research_symbols_fn: Callable[[list[str], dict[str, ConfluenceScore], int], list[str]],
    history_as_of_date_fn: Callable[[pd.DataFrame], date | None],
    iso_date_fn: Callable[[date | None], str | None],
    clean_float_fn: Callable[[float | None], float | None],
    levels_payload_fn: Callable[[TradeLevels], dict | None],
    vol_context_payload_fn: Callable[[VolatilityContext | None], dict | None],
    confluence_payload_fn: Callable[[ConfluenceScore | None], dict | None],
    option_candidate_payload_fn: Callable[[OptionCandidate | None], dict | None],
    build_derived_store_fn: Callable[[Path], Any],
) -> None:
    risk_profile = runtime.portfolio.risk_profile
    derived_store = build_derived_store_fn(args.derived_dir)
    confluence_cfg, technicals_cfg = load_research_configs_fn(runtime.console)
    symbols = resolve_research_symbols_fn(args, console=runtime.console)
    cache = build_research_cache_fn(
        symbols=symbols,
        candle_store=runtime.candle_store,
        period=args.research_period,
        risk_profile=risk_profile,
        technicals_cfg=technicals_cfg,
        confluence_cfg=confluence_cfg,
        console=runtime.console,
    )
    symbols = rank_research_symbols_fn(symbols, cache.pre_confluence_by_symbol, args.research_top)
    for symbol in symbols:
        try:
            runtime.events.append(
                _build_research_event(
                    runtime,
                    args=args,
                    symbol=symbol,
                    cache=cache,
                    risk_profile=risk_profile,
                    confluence_cfg=confluence_cfg,
                    derived_store=derived_store,
                    history_as_of_date_fn=history_as_of_date_fn,
                    iso_date_fn=iso_date_fn,
                    clean_float_fn=clean_float_fn,
                    levels_payload_fn=levels_payload_fn,
                    vol_context_payload_fn=vol_context_payload_fn,
                    confluence_payload_fn=confluence_payload_fn,
                    option_candidate_payload_fn=option_candidate_payload_fn,
                )
            )
            runtime.counts["research"] += 1
        except DataFetchError as exc:
            runtime.console.print(f"[red]Research data error:[/red] {symbol}: {exc}")
        except Exception as exc:  # noqa: BLE001
            runtime.console.print(f"[red]Research error:[/red] {symbol}: {exc}")
