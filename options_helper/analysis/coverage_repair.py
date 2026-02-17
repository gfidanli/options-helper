from __future__ import annotations

import math
from typing import Any


def _append_suggestion(
    suggestions: list[dict[str, Any]],
    *,
    priority: int,
    category: str,
    reason: str,
    command: str,
    note: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "priority": priority,
        "category": category,
        "reason": reason,
        "command": command,
    }
    if note:
        row["note"] = note
    suggestions.append(row)


def _snapshot_headers_sparse(snapshots: dict[str, Any]) -> bool:
    snapshot_days_present = int(snapshots.get("days_present_lookback") or 0)
    snapshot_expected = int(snapshots.get("expected_business_days") or 0)
    return snapshot_days_present == 0 or (
        snapshot_expected > 0 and (snapshot_days_present / snapshot_expected) < 0.8
    )


def _append_snapshot_header_suggestions(suggestions: list[dict[str, Any]], *, symbol: str) -> None:
    _append_suggestion(
        suggestions,
        priority=4,
        category="snapshot_headers",
        reason="options snapshot coverage is sparse",
        command=f"./.venv/bin/options-helper watchlists add monitor {symbol}",
    )
    _append_suggestion(
        suggestions,
        priority=5,
        category="snapshot_headers",
        reason="capture full-chain snapshots for watchlist symbols",
        command=(
            "./.venv/bin/options-helper snapshot-options portfolio.json "
            "--watchlist monitor --all-expiries --full-chain"
        ),
        note=(
            "Alpaca option-chain snapshots are live-only; expired historical dates are not fully backfillable "
            "from this endpoint."
        ),
    )


def build_repair_suggestions(
    *,
    symbol: str,
    days: int,
    candles: dict[str, Any],
    snapshots: dict[str, Any],
    contracts_oi: dict[str, Any],
    option_bars: dict[str, Any],
) -> list[dict[str, Any]]:
    sym = _normalize_symbol(symbol)
    lookback_years = max(1, int(math.ceil(max(1, int(days)) / 252.0)))

    suggestions: list[dict[str, Any]] = []

    candle_rows = int(candles.get("rows_total") or 0)
    candle_gaps = int(candles.get("missing_business_days") or 0)
    candle_missing_cells = int(candles.get("missing_value_cells") or 0)
    if candle_rows == 0 or candle_gaps > 0 or candle_missing_cells > 0:
        reason = "missing candle rows" if candle_rows == 0 else "candle gaps or nulls detected"
        _append_suggestion(
            suggestions,
            priority=1,
            category="candles",
            reason=reason,
            command=f"./.venv/bin/options-helper ingest candles --symbol {sym}",
        )

    contracts_total = int(contracts_oi.get("contracts_total") or 0)
    contracts_with_snapshots = int(contracts_oi.get("contracts_with_snapshots") or 0)
    snapshot_days_missing = int(contracts_oi.get("snapshot_days_missing") or 0)
    contracts_with_oi = int(contracts_oi.get("contracts_with_oi") or 0)
    if contracts_total > 0 and (
        contracts_with_snapshots < contracts_total or snapshot_days_missing > 0 or contracts_with_oi < contracts_total
    ):
        _append_suggestion(
            suggestions,
            priority=2,
            category="contracts",
            reason="daily contract/OI snapshots are incomplete",
            command=(
                f"./.venv/bin/options-helper ingest options-bars --symbol {sym} --contracts-only "
                "--contracts-status all --contracts-exp-start 2000-01-01"
            ),
            note="Open interest history is point-in-time; missing historical days usually cannot be backfilled.",
        )

    bars_contracts = int(option_bars.get("contracts_total") or 0)
    bars_covered = int(option_bars.get("contracts_covering_lookback_end") or 0)
    bars_with_rows = int(option_bars.get("contracts_with_rows") or 0)
    if bars_contracts == 0 or bars_covered < max(1, bars_contracts) or bars_with_rows < max(1, bars_contracts):
        _append_suggestion(
            suggestions,
            priority=3,
            category="option_bars",
            reason="option bars coverage is incomplete",
            command=(
                f"./.venv/bin/options-helper ingest options-bars --symbol {sym} "
                f"--contracts-status all --lookback-years {lookback_years} --resume"
            ),
        )
    if _snapshot_headers_sparse(snapshots):
        _append_snapshot_header_suggestions(suggestions, symbol=sym)

    suggestions.sort(key=lambda row: int(row.get("priority") or 999))
    return suggestions


def _normalize_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return "SPY"
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in {".", "-", "_"})
    return cleaned or "SPY"
