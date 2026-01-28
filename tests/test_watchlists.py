from __future__ import annotations

from datetime import date
from pathlib import Path

from options_helper.models import Portfolio, Position
from options_helper.watchlists import Watchlists, build_default_watchlists, load_watchlists, save_watchlists


def test_watchlists_normalizes_symbols() -> None:
    wl = Watchlists(watchlists={"a": ["  iRen ", "IREN", ""]})
    assert wl.watchlists["a"] == ["IREN"]


def test_build_default_watchlists_from_portfolio() -> None:
    p = Portfolio(
        cash=0.0,
        positions=[
            Position(
                id="x",
                symbol="uroy",
                option_type="call",
                expiry=date(2026, 4, 17),
                strike=5.0,
                contracts=1,
                cost_basis=1.0,
            ),
            Position(
                id="y",
                symbol="SFIX",
                option_type="call",
                expiry=date(2027, 1, 15),
                strike=5.0,
                contracts=1,
                cost_basis=1.0,
            ),
        ],
    )
    wl = build_default_watchlists(portfolio=p, extra_watchlist_symbols=["IREN"])
    assert wl.get("positions") == ["SFIX", "UROY"]
    assert wl.get("watchlist") == ["IREN"]


def test_watchlists_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "watchlists.json"
    wl = Watchlists(watchlists={"watchlist": ["IREN", "CVX"]})
    save_watchlists(path, wl)
    loaded = load_watchlists(path)
    assert loaded.watchlists == {"watchlist": ["CVX", "IREN"]}

