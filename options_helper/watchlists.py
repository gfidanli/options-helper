from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from options_helper.models import Portfolio


class Watchlists(BaseModel):
    watchlists: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "Watchlists":
        normalized: dict[str, list[str]] = {}
        for name, symbols in (self.watchlists or {}).items():
            clean = {s.strip().upper() for s in (symbols or []) if s and s.strip()}
            normalized[name] = sorted(clean)
        self.watchlists = normalized
        return self

    def get(self, name: str) -> list[str]:
        return list(self.watchlists.get(name, []))

    def set(self, name: str, symbols: list[str]) -> None:
        clean = {s.strip().upper() for s in symbols if s and s.strip()}
        self.watchlists[name] = sorted(clean)

    def add(self, name: str, symbol: str) -> None:
        sym = symbol.strip().upper()
        if not sym:
            return
        existing = set(self.watchlists.get(name, []))
        existing.add(sym)
        self.watchlists[name] = sorted(existing)

    def remove(self, name: str, symbol: str) -> None:
        sym = symbol.strip().upper()
        existing = set(self.watchlists.get(name, []))
        existing.discard(sym)
        self.watchlists[name] = sorted(existing)


def load_watchlists(path: Path) -> Watchlists:
    if not path.exists():
        return Watchlists()
    raw = path.read_text(encoding="utf-8")
    try:
        return Watchlists.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid watchlists JSON at {path}") from exc


def save_watchlists(path: Path, watchlists: Watchlists) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(watchlists.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")


def build_default_watchlists(*, portfolio: Portfolio, extra_watchlist_symbols: list[str] | None = None) -> Watchlists:
    positions_symbols = sorted({p.symbol.upper() for p in portfolio.positions})
    wl = Watchlists()
    wl.set("positions", positions_symbols)
    wl.set("watchlist", extra_watchlist_symbols or ["IREN"])
    return wl

