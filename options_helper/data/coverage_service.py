from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from options_helper.analysis.coverage import (
    compute_candle_coverage,
    compute_contract_oi_coverage,
    compute_option_bars_coverage,
    compute_snapshot_coverage,
    to_dict,
)
from options_helper.analysis.coverage_repair import build_repair_suggestions
from options_helper.data.coverage_duckdb import load_symbol_coverage_frames


def build_symbol_coverage(
    symbol: str,
    *,
    days: int,
    duckdb_path: Path | None = None,
) -> dict[str, Any]:
    lookback_days = max(1, int(days))
    frames = load_symbol_coverage_frames(symbol, duckdb_path=duckdb_path)
    as_of = _derive_as_of_date(frames.candles, frames.snapshot_headers, frames.contract_snapshots, frames.option_bars_meta)

    candle_cov = compute_candle_coverage(frames.candles, lookback_days=lookback_days)
    snapshot_cov = compute_snapshot_coverage(frames.snapshot_headers, lookback_days=lookback_days)
    contract_cov = compute_contract_oi_coverage(
        frames.contracts,
        frames.contract_snapshots,
        lookback_days=lookback_days,
        as_of=as_of,
    )
    option_bars_cov = compute_option_bars_coverage(
        frames.option_bars_meta,
        lookback_days=lookback_days,
        as_of=as_of,
    )

    candle_data = to_dict(candle_cov)
    snapshot_data = to_dict(snapshot_cov)
    contracts_data = to_dict(contract_cov)
    option_bars_data = to_dict(option_bars_cov)

    suggestions = build_repair_suggestions(
        symbol=frames.symbol,
        days=lookback_days,
        candles=candle_data,
        snapshots=snapshot_data,
        contracts_oi=contracts_data,
        option_bars=option_bars_data,
    )

    return {
        "symbol": frames.symbol,
        "days": lookback_days,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_path": str(frames.database_path),
        "database_exists": bool(frames.database_exists),
        "as_of_date": as_of.isoformat() if as_of is not None else None,
        "notes": list(frames.notes),
        "candles": candle_data,
        "snapshots": snapshot_data,
        "contracts_oi": contracts_data,
        "option_bars": option_bars_data,
        "repair_suggestions": suggestions,
    }


def _derive_as_of_date(
    candles: pd.DataFrame,
    snapshot_headers: pd.DataFrame,
    contract_snapshots: pd.DataFrame,
    option_bars_meta: pd.DataFrame,
) -> date | None:
    candidates: list[date] = []

    if candles is not None and not candles.empty and "ts" in candles.columns:
        parsed = pd.to_datetime(candles["ts"], errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            candidates.append(parsed.max().date())

    if snapshot_headers is not None and not snapshot_headers.empty and "snapshot_date" in snapshot_headers.columns:
        parsed = pd.to_datetime(snapshot_headers["snapshot_date"], errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            candidates.append(parsed.max().date())

    if contract_snapshots is not None and not contract_snapshots.empty and "as_of_date" in contract_snapshots.columns:
        parsed = pd.to_datetime(contract_snapshots["as_of_date"], errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            candidates.append(parsed.max().date())

    if option_bars_meta is not None and not option_bars_meta.empty and "end_ts" in option_bars_meta.columns:
        parsed = pd.to_datetime(option_bars_meta["end_ts"], errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            candidates.append(parsed.max().date())

    if not candidates:
        return None
    return max(candidates)
