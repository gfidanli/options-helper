from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.derived_metrics import DerivedRow
from options_helper.commands.common import _spot_from_meta
from options_helper.reporting_briefing import BriefingSymbolSection
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot, compute_technical_snapshot

from .visibility_jobs_briefing_models_legacy import _SymbolResult, _SymbolState
from .visibility_jobs_briefing_symbol_utils_legacy import (
    _collect_event_warnings,
    _compute_symbol_confluence,
    _load_compare_and_flow,
)

if TYPE_CHECKING:
    from options_helper.models import Position


def _load_symbol_snapshot_data(
    *,
    store: Any,
    sym: str,
    as_of: str,
    risk_profile: Any,
    positions_for_symbol: list[Position],
    next_earnings_date: date | None,
    execution_error_factory: Callable[..., Exception],
) -> tuple[tuple[date, pd.DataFrame], float, Any, list[str]]:
    to_date = store.resolve_date(sym, as_of)
    warnings = _collect_event_warnings(
        today=to_date,
        next_earnings_date=next_earnings_date,
        risk_profile=risk_profile,
        positions_for_symbol=positions_for_symbol,
    )
    df_to = store.load_day(sym, to_date)
    meta_to = store.load_meta(sym, to_date)
    quote_quality = meta_to.get("quote_quality") if isinstance(meta_to, dict) else None
    spot_to = _spot_from_meta(meta_to)
    if spot_to is None:
        raise execution_error_factory("missing spot price in meta.json (run snapshot-options first)")
    return (to_date, df_to), float(spot_to), quote_quality, warnings


def _load_symbol_technicals(
    *,
    sym: str,
    candle_store: Any,
    technicals_cfg: dict[str, Any] | None,
    technicals_cfg_error: str | None,
) -> tuple[TechnicalSnapshot | None, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if technicals_cfg is None:
        if technicals_cfg_error:
            warnings.append(f"technicals unavailable: {technicals_cfg_error}")
        return None, pd.DataFrame(), warnings

    candles = candle_store.load(sym)
    if candles.empty:
        warnings.append("technicals unavailable: missing candle cache (run refresh-candles)")
        return None, candles, warnings

    technicals = compute_technical_snapshot(candles, technicals_cfg)
    if technicals is None:
        warnings.append("technicals unavailable: insufficient candle history / warmup")
    return technicals, candles, warnings


def _maybe_update_symbol_derived(
    *,
    update_derived: bool,
    sym: str,
    chain: Any,
    candles: pd.DataFrame,
    derived_store: Any,
) -> tuple[bool, Any, list[str]]:
    if not update_derived:
        return False, None, []

    row = DerivedRow.from_chain_report(chain, candles=candles, derived_history=derived_store.load(sym))
    try:
        derived_store.upsert(sym, row)
    except Exception as exc:  # noqa: BLE001
        return False, None, [f"derived update failed: {exc}"]
    return True, row, []


def _compute_symbol_core(
    *,
    day_entry: tuple[date, pd.DataFrame],
    spot_to: float,
    sym: str,
    top: int,
    update_derived: bool,
    candles: pd.DataFrame,
    derived_store: Any,
    compare_enabled: bool,
    compare_norm: str,
    store: Any,
) -> tuple[Any, bool, Any, Any | None, pd.DataFrame | None, list[str]]:
    warnings: list[str] = []
    chain = compute_chain_report(
        day_entry[1],
        symbol=sym,
        as_of=day_entry[0],
        spot=spot_to,
        expiries_mode="near",
        top=10,
        best_effort=True,
    )
    derived_updated, derived_row, derived_warnings = _maybe_update_symbol_derived(
        update_derived=update_derived,
        sym=sym,
        chain=chain,
        candles=candles,
        derived_store=derived_store,
    )
    warnings.extend(derived_warnings)
    compare_report, flow_net, compare_warnings = _load_compare_and_flow(
        compare_enabled=compare_enabled,
        compare_norm=compare_norm,
        store=store,
        sym=sym,
        to_date=day_entry[0],
        df_to=day_entry[1],
        spot_to=spot_to,
        top=top,
    )
    warnings.extend(compare_warnings)
    return chain, derived_updated, derived_row, compare_report, flow_net, warnings


def _build_symbol_state(
    *,
    day_entry: tuple[date, pd.DataFrame],
    candles: pd.DataFrame,
    chain: Any,
    compare_report: Any,
    flow_net: pd.DataFrame | None,
    technicals: TechnicalSnapshot | None,
    confluence_score: Any,
    quote_quality: Any,
    derived_updated: bool,
    derived_row: Any,
    warnings: list[str],
) -> _SymbolState:
    return _SymbolState(
        day_entry=day_entry,
        candles=candles,
        chain=chain,
        compare_report=compare_report,
        flow_net=flow_net,
        technicals=technicals,
        confluence_score=confluence_score,
        quote_quality=quote_quality,
        derived_updated=derived_updated,
        derived_row=derived_row,
        warnings=warnings,
        errors=[],
    )


def _failed_symbol_state(*, warnings: list[str], error: str) -> _SymbolState:
    return _SymbolState(
        day_entry=None,
        candles=pd.DataFrame(),
        chain=None,
        compare_report=None,
        flow_net=None,
        technicals=None,
        confluence_score=None,
        quote_quality=None,
        derived_updated=False,
        derived_row=None,
        warnings=warnings,
        errors=[error],
    )


def _compute_symbol_state(
    *,
    sym: str,
    as_of: str,
    compare_enabled: bool,
    compare_norm: str,
    top: int,
    store: Any,
    candle_store: Any,
    derived_store: Any,
    next_earnings_date: date | None,
    technicals_cfg: dict[str, Any] | None,
    technicals_cfg_error: str | None,
    confluence_cfg: Any,
    risk_profile: Any,
    positions_for_symbol: list[Position],
    update_derived: bool,
    execution_error_factory: Callable[..., Exception],
) -> _SymbolState:
    warnings: list[str] = []
    try:
        day_entry, spot_to, quote_quality, event_warnings = _load_symbol_snapshot_data(
            store=store,
            sym=sym,
            as_of=as_of,
            risk_profile=risk_profile,
            positions_for_symbol=positions_for_symbol,
            next_earnings_date=next_earnings_date,
            execution_error_factory=execution_error_factory,
        )
        warnings.extend(event_warnings)
        technicals, candles, tech_warnings = _load_symbol_technicals(
            sym=sym,
            candle_store=candle_store,
            technicals_cfg=technicals_cfg,
            technicals_cfg_error=technicals_cfg_error,
        )
        warnings.extend(tech_warnings)
        chain, derived_updated, derived_row, compare_report, flow_net, core_warnings = _compute_symbol_core(
            day_entry=day_entry,
            spot_to=spot_to,
            sym=sym,
            top=top,
            update_derived=update_derived,
            candles=candles,
            derived_store=derived_store,
            compare_enabled=compare_enabled,
            compare_norm=compare_norm,
            store=store,
        )
        warnings.extend(core_warnings)
        confluence_score, confluence_warning = _compute_symbol_confluence(
            technicals=technicals,
            flow_net=flow_net,
            derived_row=derived_row,
            confluence_cfg=confluence_cfg,
        )
        if confluence_warning:
            warnings.append(confluence_warning)
        return _build_symbol_state(
            day_entry=day_entry,
            candles=candles,
            chain=chain,
            compare_report=compare_report,
            flow_net=flow_net,
            technicals=technicals,
            confluence_score=confluence_score,
            quote_quality=quote_quality,
            derived_updated=derived_updated,
            derived_row=derived_row,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_symbol_state(warnings=warnings, error=str(exc))


def _build_symbol_section(
    *,
    sym: str,
    as_of: str,
    compare_enabled: bool,
    compare_norm: str,
    top: int,
    store: Any,
    candle_store: Any,
    derived_store: Any,
    earnings_store: Any,
    technicals_cfg: dict[str, Any] | None,
    technicals_cfg_error: str | None,
    confluence_cfg: Any,
    risk_profile: Any,
    positions_for_symbol: list[Position],
    update_derived: bool,
    safe_next_earnings_date_fn: Callable[..., date | None],
    execution_error_factory: Callable[..., Exception],
) -> _SymbolResult:
    next_earnings_date = safe_next_earnings_date_fn(earnings_store, sym)
    state = _compute_symbol_state(
        sym=sym,
        as_of=as_of,
        compare_enabled=compare_enabled,
        compare_norm=compare_norm,
        top=top,
        store=store,
        candle_store=candle_store,
        derived_store=derived_store,
        next_earnings_date=next_earnings_date,
        technicals_cfg=technicals_cfg,
        technicals_cfg_error=technicals_cfg_error,
        confluence_cfg=confluence_cfg,
        risk_profile=risk_profile,
        positions_for_symbol=positions_for_symbol,
        update_derived=update_derived,
        execution_error_factory=execution_error_factory,
    )
    as_of_label = "-" if state.day_entry is None else state.day_entry[0].isoformat()
    return _SymbolResult(
        section=BriefingSymbolSection(
            symbol=sym,
            as_of=as_of_label,
            chain=state.chain,
            compare=state.compare_report,
            flow_net=state.flow_net,
            technicals=state.technicals,
            confluence=state.confluence_score,
            errors=state.errors,
            warnings=state.warnings,
            quote_quality=state.quote_quality,
            derived_updated=state.derived_updated,
            derived=state.derived_row,
            next_earnings_date=next_earnings_date,
        ),
        day_entry=state.day_entry,
        candles=state.candles,
        next_earnings_date=next_earnings_date,
    )
