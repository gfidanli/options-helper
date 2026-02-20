from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd
from rich.console import RenderableType

from options_helper.analysis.portfolio_risk import compute_portfolio_exposure, run_stress
from options_helper.commands.common import _build_stress_scenarios
from options_helper.commands.position_metrics import _extract_float, _mark_price, _position_metrics
from options_helper.data.candles import close_asof, last_close
from options_helper.reporting_briefing import render_portfolio_table_markdown

from .visibility_jobs_briefing_models_legacy import _PortfolioOutputs

if TYPE_CHECKING:
    from options_helper.models import Position


def _resolve_position_snapshot(
    *,
    position: Position,
    day_cache: dict[str, tuple[date, pd.DataFrame]],
) -> tuple[date | None, Any, float | None, float | None]:
    to_date, df_to = day_cache.get(position.symbol.upper(), (None, pd.DataFrame()))
    if df_to.empty:
        return to_date, None, None, None

    subset = df_to.copy()
    if "expiry" in subset.columns:
        subset = subset[subset["expiry"].astype(str) == position.expiry.isoformat()]
    if "optionType" in subset.columns:
        subset = subset[subset["optionType"].astype(str).str.lower() == position.option_type]
    if "strike" in subset.columns:
        subset = subset.assign(_strike=pd.to_numeric(subset["strike"], errors="coerce"))
        subset = subset[(subset["_strike"] - float(position.strike)).abs() < 1e-9]
    if subset.empty:
        return to_date, None, None, None

    snapshot_row = subset.iloc[0]
    bid = _extract_float(snapshot_row, "bid")
    ask = _extract_float(snapshot_row, "ask")
    mark = _mark_price(bid=bid, ask=ask, last=_extract_float(snapshot_row, "lastPrice"))
    spread_pct = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        if mid > 0:
            spread_pct = (ask - bid) / mid
    return to_date, snapshot_row, mark, spread_pct


def _ensure_symbol_history(*, sym: str, candles_by_symbol: dict[str, pd.DataFrame], candle_store: Any) -> pd.DataFrame:
    history = candles_by_symbol.get(sym)
    if history is not None and not history.empty:
        return history

    try:
        history = candle_store.load(sym)
    except Exception:  # noqa: BLE001
        history = pd.DataFrame()
    candles_by_symbol[sym] = history
    return history


def _collect_position_metrics(
    *,
    position: Position,
    risk_profile: Any,
    to_date: date | None,
    snapshot_row: Any,
    history: pd.DataFrame,
    next_earnings_date: date | None,
    metrics: list[Any],
    renderables: list[RenderableType],
) -> None:
    try:
        last_price = close_asof(history, to_date) if to_date is not None else last_close(history)
        metrics.append(
            _position_metrics(
                None,
                position,
                risk_profile=risk_profile,
                underlying_history=history,
                underlying_last_price=last_price,
                as_of=to_date,
                next_earnings_date=next_earnings_date,
                snapshot_row=snapshot_row if snapshot_row is not None else {},
            )
        )
    except Exception as exc:  # noqa: BLE001
        renderables.append(f"[yellow]Warning:[/yellow] portfolio exposure skipped for {position.id}: {exc}")


def _position_row_values(
    *,
    position: Position,
    sym: str,
    to_date: date | None,
    mark: float | None,
    spread_pct: float | None,
) -> tuple[dict[str, str], dict[str, object], float]:
    pnl_abs = (mark - position.cost_basis) * 100.0 * position.contracts if mark is not None else None
    pnl_pct = ((mark / position.cost_basis) - 1.0) if mark is not None and position.cost_basis > 0 else None
    row = {
        "id": position.id,
        "symbol": sym,
        "type": position.option_type,
        "expiry": position.expiry.isoformat(),
        "strike": f"{position.strike:g}",
        "ct": str(position.contracts),
        "cost": f"{position.cost_basis:.2f}",
        "mark": "-" if mark is None else f"{mark:.2f}",
        "pnl_$": "-" if pnl_abs is None else f"{pnl_abs:+.0f}",
        "pnl_%": "-" if pnl_pct is None else f"{pnl_pct * 100.0:+.1f}%",
        "spr_%": "-" if spread_pct is None else f"{spread_pct * 100.0:.1f}%",
        "as_of": "-" if to_date is None else to_date.isoformat(),
    }
    payload = {
        "id": position.id,
        "symbol": sym,
        "option_type": position.option_type,
        "expiry": position.expiry.isoformat(),
        "strike": float(position.strike),
        "contracts": int(position.contracts),
        "cost_basis": float(position.cost_basis),
        "mark": None if mark is None else float(mark),
        "pnl": None if pnl_abs is None else float(pnl_abs),
        "pnl_pct": None if pnl_pct is None else float(pnl_pct),
        "spr_pct": None if spread_pct is None else float(spread_pct),
        "as_of": None if to_date is None else to_date.isoformat(),
    }
    rank = float(pnl_pct) if pnl_pct is not None else float("-inf")
    return row, payload, rank


def _render_portfolio_table(rows_with_pnl: list[tuple[float, dict[str, str]]]) -> str | None:
    if not rows_with_pnl:
        return None

    rows_sorted = [row for _, row in sorted(rows_with_pnl, key=lambda item: item[0], reverse=True)]
    include_spread = any(row.get("spr_%") not in (None, "-") for row in rows_sorted)
    return render_portfolio_table_markdown(rows_sorted, include_spread=include_spread)


def _collect_portfolio_outputs(
    *,
    portfolio_positions: list[Position],
    risk_profile: Any,
    day_cache: dict[str, tuple[date, pd.DataFrame]],
    candles_by_symbol: dict[str, pd.DataFrame],
    next_earnings_by_symbol: dict[str, date | None],
    candle_store: Any,
    renderables: list[RenderableType],
) -> _PortfolioOutputs:
    rows_payload: list[dict[str, object]] = []
    rows_with_pnl: list[tuple[float, dict[str, str]]] = []
    metrics: list[Any] = []

    for position in portfolio_positions:
        sym = position.symbol.upper()
        to_date, snapshot_row, mark, spread_pct = _resolve_position_snapshot(position=position, day_cache=day_cache)
        history = _ensure_symbol_history(sym=sym, candles_by_symbol=candles_by_symbol, candle_store=candle_store)
        _collect_position_metrics(
            position=position,
            risk_profile=risk_profile,
            to_date=to_date,
            snapshot_row=snapshot_row,
            history=history,
            next_earnings_date=next_earnings_by_symbol.get(sym),
            metrics=metrics,
            renderables=renderables,
        )
        row, payload, rank = _position_row_values(
            position=position,
            sym=sym,
            to_date=to_date,
            mark=mark,
            spread_pct=spread_pct,
        )
        rows_payload.append(payload)
        rows_with_pnl.append((rank, row))

    return _PortfolioOutputs(
        rows_payload=rows_payload,
        table_markdown=_render_portfolio_table(rows_with_pnl),
        metrics=metrics,
    )


def _portfolio_risk_outputs(portfolio_outputs: _PortfolioOutputs) -> tuple[Any, Any]:
    if not portfolio_outputs.metrics:
        return None, None

    exposure = compute_portfolio_exposure(portfolio_outputs.metrics)
    stress = run_stress(exposure, _build_stress_scenarios(stress_spot_pct=[], stress_vol_pp=5.0, stress_days=7))
    return exposure, stress
