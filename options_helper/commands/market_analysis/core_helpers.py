from __future__ import annotations

from datetime import datetime

import pandas as pd
import typer


def _normalize_symbol(value: str) -> str:
    raw = str(value or "").strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def _normalize_output_format(value: str) -> str:
    output_fmt = str(value or "").strip().lower()
    if output_fmt not in {"console", "json"}:
        raise typer.BadParameter("Invalid --format (use console|json)", param_hint="--format")
    return output_fmt


def _to_python_datetime(value: pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "-"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


__all__ = [
    "_normalize_symbol",
    "_normalize_output_format",
    "_to_python_datetime",
    "_dedupe",
    "_fmt_num",
    "_fmt_int",
    "_fmt_pct",
    "_as_float",
]
