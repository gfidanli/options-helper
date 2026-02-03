from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def _write_chain_day(
    cache_dir: Path,
    *,
    symbol: str,
    day: str,
    spot: float,
    oi_shift: int = 0,
) -> None:
    day_dir = cache_dir / symbol / day
    day_dir.mkdir(parents=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": spot}), encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "contractSymbol": f"{symbol}_C_110",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 110.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "openInterest": 900 + oi_shift,
                "volume": 10,
                "impliedVolatility": 0.30,
                "bs_delta": 0.25,
                "bs_gamma": 0.015,
            },
            {
                "contractSymbol": f"{symbol}_P_90",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 90.0,
                "bid": 0.9,
                "ask": 1.1,
                "lastPrice": 1.0,
                "openInterest": 1200 - oi_shift,
                "volume": 10,
                "impliedVolatility": 0.35,
                "bs_delta": -0.25,
                "bs_gamma": 0.012,
            },
        ]
    )
    df.to_csv(day_dir / "2026-02-20.csv", index=False)


def test_briefing_writes_daily_md_and_updates_derived(tmp_path: Path, monkeypatch) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                    "earnings_warn_days": 21,
                    "earnings_avoid_days": 0,
                },
                "positions": [
                    {
                        "id": "aaa-2026-02-20-110c",
                        "symbol": "AAA",
                        "option_type": "call",
                        "expiry": "2026-02-20",
                        "strike": 110.0,
                        "contracts": 1,
                        "cost_basis": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text('{"watchlists":{"monitor":["BBB"]}}', encoding="utf-8")

    cache_dir = tmp_path / "snapshots"
    _write_chain_day(cache_dir, symbol="AAA", day="2026-01-01", spot=100.0, oi_shift=0)
    _write_chain_day(cache_dir, symbol="AAA", day="2026-01-02", spot=100.0, oi_shift=100)

    # BBB only has one day; should appear with a compare warning but still produce chain.
    _write_chain_day(cache_dir, symbol="BBB", day="2026-01-02", spot=50.0, oi_shift=0)

    out_dir = tmp_path / "reports"
    derived_dir = tmp_path / "derived"

    def _stub_next_earnings_date(store, symbol):  # type: ignore[no-untyped-def]
        if symbol.upper() == "AAA":
            return date(2026, 1, 10)
        return None

    monkeypatch.setattr("options_helper.cli.safe_next_earnings_date", _stub_next_earnings_date)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "briefing",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--watchlist",
            "monitor",
            "--as-of",
            "2026-01-02",
            "--compare",
            "-1",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_dir),
            "--derived-dir",
            str(derived_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    md_path = out_dir / "2026-01-02.md"
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")

    assert "# Daily briefing (2026-01-02)" in content
    assert "Spr%" in content
    assert "## AAA (2026-01-02)" in content
    assert "Next earnings: 2026-01-10 (in 8 day(s))" in content
    assert "earnings_within_21d" in content
    assert "expiry_crosses_earnings" in content
    assert "### Chain" in content
    assert "### Compare" in content
    assert "### Flow (net, aggregated by strike)" in content
    assert "- Derived: updated" in content

    assert "## BBB (2026-01-02)" in content
    assert "compare unavailable" in content

    assert (derived_dir / "AAA.csv").exists()
    assert (derived_dir / "BBB.csv").exists()
