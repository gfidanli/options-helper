from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path
from options_helper.data.strategy_modeling_io import (
    AdjustedDataFallbackMode,
    StrategyModelingIntradayPreflightResult,
    build_required_intraday_sessions,
    list_strategy_modeling_universe,
    load_daily_ohlc_history,
    normalize_symbol,
    preflight_intraday_coverage,
)

_DEFAULT_INTRADAY_DIR = Path("data/intraday")
_ALLOWED_FALLBACK_MODES: set[str] = {"warn_and_skip_symbol", "use_unadjusted_ohlc"}


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


def list_strategy_modeling_symbols(*, database_path: str | Path | None = None) -> tuple[list[str], list[str]]:
    payload = _list_strategy_modeling_symbols_cached(database_path=str(resolve_duckdb_path(database_path)))
    return list(payload["symbols"]), list(payload["notes"])


@st.cache_data(ttl=60, show_spinner=False)
def _list_strategy_modeling_symbols_cached(*, database_path: str) -> dict[str, Any]:
    try:
        result = list_strategy_modeling_universe(database_path=database_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "symbols": [],
            "notes": [str(exc)],
            "database_exists": Path(database_path).exists(),
        }

    return {
        "symbols": list(result.symbols),
        "notes": list(result.notes),
        "database_exists": bool(result.database_exists),
    }


def load_strategy_modeling_data_payload(
    *,
    symbols: Sequence[str] | str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    database_path: str | Path | None = None,
    intraday_dir: str | Path = _DEFAULT_INTRADAY_DIR,
    intraday_timeframe: str = "1Min",
    require_intraday_bars: bool = True,
    adjusted_data_fallback_mode: AdjustedDataFallbackMode = "warn_and_skip_symbol",
) -> dict[str, Any]:
    resolved_db = resolve_duckdb_path(database_path)
    resolved_intraday = Path(intraday_dir)

    try:
        return _load_strategy_modeling_payload_cached(
            database_path=str(resolved_db),
            intraday_dir=str(resolved_intraday),
            symbols=_normalize_symbol_filters(symbols),
            start_date_iso=start_date.isoformat() if start_date is not None else None,
            end_date_iso=end_date.isoformat() if end_date is not None else None,
            intraday_timeframe=str(intraday_timeframe or "").strip() or "1Min",
            require_intraday_bars=bool(require_intraday_bars),
            adjusted_data_fallback_mode=str(adjusted_data_fallback_mode),
        )
    except Exception as exc:  # noqa: BLE001
        payload = _init_payload(
            database_path=resolved_db,
            intraday_dir=resolved_intraday,
            start_date=start_date,
            end_date=end_date,
            intraday_timeframe=intraday_timeframe,
            require_intraday_bars=require_intraday_bars,
            adjusted_data_fallback_mode=str(adjusted_data_fallback_mode),
        )
        payload["errors"].append(str(exc))
        payload["status"] = "error"
        payload["blocking"] = build_blocking_status_payload(
            preflight_payload=payload["intraday_preflight"],
            errors=payload["errors"],
            require_intraday_bars=bool(require_intraday_bars),
        )
        return payload


@st.cache_data(ttl=60, show_spinner=False)
def _load_strategy_modeling_payload_cached(
    *,
    database_path: str,
    intraday_dir: str,
    symbols: tuple[str, ...],
    start_date_iso: str | None,
    end_date_iso: str | None,
    intraday_timeframe: str,
    require_intraday_bars: bool,
    adjusted_data_fallback_mode: str,
) -> dict[str, Any]:
    start_date = _parse_date_or_none(start_date_iso)
    end_date = _parse_date_or_none(end_date_iso)
    payload = _init_payload(
        database_path=Path(database_path),
        intraday_dir=Path(intraday_dir),
        start_date=start_date,
        end_date=end_date,
        intraday_timeframe=intraday_timeframe,
        require_intraday_bars=require_intraday_bars,
        adjusted_data_fallback_mode=adjusted_data_fallback_mode,
    )

    validation_errors = validate_filter_combination(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        intraday_timeframe=intraday_timeframe,
        adjusted_data_fallback_mode=adjusted_data_fallback_mode,
    )
    if validation_errors:
        payload["errors"].extend(validation_errors)
        payload["status"] = "error"
        payload["blocking"] = build_blocking_status_payload(
            preflight_payload=payload["intraday_preflight"],
            errors=payload["errors"],
            require_intraday_bars=require_intraday_bars,
        )
        return payload

    universe = list_strategy_modeling_universe(database_path=database_path)
    payload["notes"].extend(universe.notes)
    payload["universe_symbols"] = list(universe.symbols)

    if not universe.symbols:
        if universe.notes:
            payload["errors"].append(universe.notes[0])
        else:
            payload["errors"].append("No symbols available in universe.")
        payload["status"] = "error"
        payload["blocking"] = build_blocking_status_payload(
            preflight_payload=payload["intraday_preflight"],
            errors=payload["errors"],
            require_intraday_bars=require_intraday_bars,
        )
        return payload

    requested_symbols, symbol_errors, symbol_notes = _resolve_requested_symbols(
        symbols=symbols,
        universe_symbols=tuple(universe.symbols),
    )
    payload["notes"].extend(symbol_notes)
    if symbol_errors:
        payload["errors"].extend(symbol_errors)
        payload["status"] = "error"
        payload["blocking"] = build_blocking_status_payload(
            preflight_payload=payload["intraday_preflight"],
            errors=payload["errors"],
            require_intraday_bars=require_intraday_bars,
        )
        return payload
    payload["requested_symbols"] = list(requested_symbols)

    try:
        daily = load_daily_ohlc_history(
            requested_symbols,
            database_path=database_path,
            adjusted_data_fallback_mode=adjusted_data_fallback_mode,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:  # noqa: BLE001
        payload["errors"].append(str(exc))
        payload["status"] = "error"
        payload["blocking"] = build_blocking_status_payload(
            preflight_payload=payload["intraday_preflight"],
            errors=payload["errors"],
            require_intraday_bars=require_intraday_bars,
        )
        return payload

    payload["notes"].extend(daily.notes)
    payload["modeled_symbols"] = sorted(daily.candles_by_symbol)
    payload["skipped_symbols"] = sorted(daily.skipped_symbols)
    payload["missing_symbols"] = sorted(daily.missing_symbols)
    payload["source_by_symbol"] = dict(daily.source_by_symbol)
    payload["daily_rows_by_symbol"] = {
        symbol: int(len(frame.index)) for symbol, frame in sorted(daily.candles_by_symbol.items())
    }

    required_sessions = build_required_intraday_sessions(
        daily.candles_by_symbol,
        start_date=start_date,
        end_date=end_date,
    )
    payload["required_sessions_by_symbol"] = serialize_required_sessions(required_sessions)

    preflight = preflight_intraday_coverage(
        required_sessions,
        intraday_dir=Path(intraday_dir),
        timeframe=intraday_timeframe,
        require_intraday_bars=require_intraday_bars,
    )
    payload["intraday_preflight"] = intraday_preflight_to_payload(preflight)

    if not payload["modeled_symbols"]:
        payload["errors"].append("No daily candles available after applying current filters.")
        payload["status"] = "error"

    payload["has_data"] = bool(payload["modeled_symbols"]) and not bool(payload["errors"])
    payload["blocking"] = build_blocking_status_payload(
        preflight_payload=payload["intraday_preflight"],
        errors=payload["errors"],
        require_intraday_bars=require_intraday_bars,
    )
    if payload["errors"]:
        payload["status"] = "error"
    return payload


def validate_filter_combination(
    *,
    symbols: Sequence[str],
    start_date: date | None,
    end_date: date | None,
    intraday_timeframe: str,
    adjusted_data_fallback_mode: str,
) -> list[str]:
    errors: list[str] = []

    if start_date is not None and end_date is not None and start_date > end_date:
        errors.append(
            f"Invalid date range: start_date ({start_date.isoformat()}) must be on or before "
            f"end_date ({end_date.isoformat()})."
        )
    if not str(intraday_timeframe or "").strip():
        errors.append("intraday_timeframe must be non-empty.")
    if str(adjusted_data_fallback_mode or "") not in _ALLOWED_FALLBACK_MODES:
        allowed = ", ".join(sorted(_ALLOWED_FALLBACK_MODES))
        errors.append(
            f"Invalid adjusted_data_fallback_mode={adjusted_data_fallback_mode!r}; expected one of: {allowed}"
        )
    if symbols and all(not normalize_symbol(item) for item in symbols):
        errors.append("Symbol filter is present but did not contain any valid ticker values.")

    return errors


def serialize_required_sessions(required_sessions_by_symbol: Mapping[str, Sequence[date]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for symbol in sorted(required_sessions_by_symbol):
        days = sorted({day for day in required_sessions_by_symbol[symbol]})
        out[symbol] = [day.isoformat() for day in days]
    return out


def intraday_preflight_to_payload(preflight: StrategyModelingIntradayPreflightResult) -> dict[str, Any]:
    coverage_by_symbol: dict[str, dict[str, Any]] = {}
    for symbol in sorted(preflight.coverage_by_symbol):
        coverage = preflight.coverage_by_symbol[symbol]
        coverage_by_symbol[symbol] = {
            "symbol": coverage.symbol,
            "required_days": [day.isoformat() for day in coverage.required_days],
            "covered_days": [day.isoformat() for day in coverage.covered_days],
            "missing_days": [day.isoformat() for day in coverage.missing_days],
            "is_complete": coverage.is_complete,
        }

    return {
        "require_intraday_bars": bool(preflight.require_intraday_bars),
        "is_blocked": bool(preflight.is_blocked),
        "blocked_symbols": list(preflight.blocked_symbols),
        "notes": list(preflight.notes),
        "coverage_by_symbol": coverage_by_symbol,
    }


def preflight_coverage_rows(preflight_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    coverage_payload = preflight_payload.get("coverage_by_symbol", {})
    if not isinstance(coverage_payload, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for symbol in sorted(str(key) for key in coverage_payload):
        row_payload = coverage_payload.get(symbol, {})
        if not isinstance(row_payload, Mapping):
            continue
        required_days = _string_list(row_payload.get("required_days"))
        covered_days = _string_list(row_payload.get("covered_days"))
        missing_days = _string_list(row_payload.get("missing_days"))
        rows.append(
            {
                "symbol": symbol,
                "required_count": len(required_days),
                "covered_count": len(covered_days),
                "missing_count": len(missing_days),
                "missing_days": missing_days,
                "is_complete": bool(row_payload.get("is_complete", not missing_days)),
            }
        )
    return rows


def build_blocking_status_payload(
    *,
    preflight_payload: Mapping[str, Any],
    errors: Sequence[str],
    require_intraday_bars: bool,
) -> dict[str, Any]:
    coverage_rows = preflight_coverage_rows(preflight_payload)
    blocked_symbols = _string_list(preflight_payload.get("blocked_symbols"))
    missing_sessions_total = sum(int(row["missing_count"]) for row in coverage_rows)
    blocked_by_preflight = bool(require_intraday_bars) and bool(blocked_symbols)
    blocked_by_errors = bool(errors)

    reason: str | None = None
    if blocked_by_errors:
        reason = "requirements_not_met"
    elif blocked_by_preflight:
        reason = "intraday_coverage_missing"

    return {
        "is_blocked": blocked_by_errors or blocked_by_preflight,
        "reason": reason,
        "require_intraday_bars": bool(require_intraday_bars),
        "blocked_symbols": blocked_symbols,
        "missing_sessions_total": missing_sessions_total,
        "coverage_rows": coverage_rows,
        "error_count": len(errors),
    }


def _resolve_requested_symbols(
    *,
    symbols: Sequence[str],
    universe_symbols: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    requested = list(symbols)
    if not requested:
        return list(universe_symbols), [], []

    universe_set = {normalize_symbol(value) for value in universe_symbols}
    selected = [symbol for symbol in requested if symbol in universe_set]
    missing = [symbol for symbol in requested if symbol not in universe_set]

    errors: list[str] = []
    notes: list[str] = []
    if missing:
        notes.append("Ignored symbol filters not present in universe: " + ", ".join(sorted(missing)))
    if not selected:
        errors.append("Symbol filters excluded all available symbols in the current universe.")
    return selected, errors, notes


def _normalize_symbol_filters(symbols: Sequence[str] | str | None) -> tuple[str, ...]:
    if symbols is None:
        return ()

    raw_values: Sequence[object]
    if isinstance(symbols, str):
        raw_values = tuple(part.strip() for part in symbols.split(","))
    else:
        raw_values = tuple(symbols)

    out: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        normalized = normalize_symbol(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _parse_date_or_none(value: str | None) -> date | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return date.fromisoformat(cleaned)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _init_payload(
    *,
    database_path: Path,
    intraday_dir: Path,
    start_date: date | None,
    end_date: date | None,
    intraday_timeframe: str,
    require_intraday_bars: bool,
    adjusted_data_fallback_mode: str,
) -> dict[str, Any]:
    preflight_payload = {
        "require_intraday_bars": bool(require_intraday_bars),
        "is_blocked": False,
        "blocked_symbols": [],
        "notes": [],
        "coverage_by_symbol": {},
    }

    return {
        "status": "ok",
        "errors": [],
        "notes": [],
        "database_path": str(database_path),
        "intraday_dir": str(intraday_dir),
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "intraday_timeframe": str(intraday_timeframe),
        "require_intraday_bars": bool(require_intraday_bars),
        "adjusted_data_fallback_mode": adjusted_data_fallback_mode,
        "universe_symbols": [],
        "requested_symbols": [],
        "modeled_symbols": [],
        "skipped_symbols": [],
        "missing_symbols": [],
        "source_by_symbol": {},
        "daily_rows_by_symbol": {},
        "required_sessions_by_symbol": {},
        "intraday_preflight": preflight_payload,
        "blocking": build_blocking_status_payload(
            preflight_payload=preflight_payload,
            errors=[],
            require_intraday_bars=bool(require_intraday_bars),
        ),
        "has_data": False,
    }


__all__ = [
    "build_blocking_status_payload",
    "intraday_preflight_to_payload",
    "list_strategy_modeling_symbols",
    "load_strategy_modeling_data_payload",
    "normalize_symbol",
    "preflight_coverage_rows",
    "resolve_duckdb_path",
    "serialize_required_sessions",
    "validate_filter_combination",
]
