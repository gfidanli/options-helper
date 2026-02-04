from __future__ import annotations

from options_helper.analysis.osi import normalize_underlying


def to_alpaca_symbol(repo_symbol: str) -> str:
    normalized = normalize_underlying(repo_symbol)
    if not normalized:
        return ""
    if "-" in normalized and "." not in normalized:
        return normalized.replace("-", ".")
    return normalized


def to_repo_symbol(alpaca_symbol: str) -> str:
    return normalize_underlying(alpaca_symbol)
