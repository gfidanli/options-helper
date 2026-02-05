from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

from options_helper.analysis.osi import normalize_underlying
from options_helper.watchlists import load_watchlists


DEFAULT_WATCHLISTS = ("positions", "monitor")


@dataclass(frozen=True)
class SymbolSelection:
    symbols: list[str]
    watchlists_used: list[str]
    warnings: list[str]


def normalize_symbols(values: Iterable[str]) -> list[str]:
    normalized: set[str] = set()
    for raw in values or []:
        if raw is None:
            continue
        for part in str(raw).split(","):
            sym = normalize_underlying(part)
            if sym:
                normalized.add(sym)
    return sorted(normalized)


def resolve_symbols(
    *,
    watchlists_path: Path,
    watchlists: list[str] | None,
    symbols: list[str] | None,
    default_watchlists: Sequence[str] = DEFAULT_WATCHLISTS,
) -> SymbolSelection:
    if symbols:
        return SymbolSelection(normalize_symbols(symbols), [], [])

    names = list(watchlists) if watchlists else list(default_watchlists)
    wl = load_watchlists(watchlists_path)
    warnings: list[str] = []
    selected: set[str] = set()

    for name in names:
        syms = wl.get(name)
        if not syms:
            warnings.append(f"Watchlist '{name}' is empty or missing in {watchlists_path}")
            continue
        for sym in syms:
            norm = normalize_underlying(sym)
            if norm:
                selected.add(norm)

    return SymbolSelection(sorted(selected), names, warnings)


def parse_date(value: str, *, label: str) -> date:
    raw = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid {label} date: {value} (use YYYY-MM-DD)")


def shift_years(value: date, years: int) -> date:
    if years == 0:
        return value
    year = value.year + years
    month = value.month
    day = value.day
    last_day = calendar.monthrange(year, month)[1]
    if day > last_day:
        day = last_day
    return date(year, month, day)
