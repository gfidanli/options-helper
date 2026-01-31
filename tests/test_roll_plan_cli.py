from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_roll_plan_cli_ranks_candidates_and_applies_liquidity_gates(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "base_currency": "USD",
                "cash": 0.0,
                "risk_profile": {"min_open_interest": 100, "min_volume": 10},
                "positions": [
                    {
                        "id": "aaa-2026-02-20-100c",
                        "symbol": "AAA",
                        "option_type": "call",
                        "expiry": "2026-02-20",
                        "strike": 100.0,
                        "contracts": 1,
                        "cost_basis": 2.00,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "snapshots"
    day_dir = cache_dir / "AAA" / "2026-01-02"
    day_dir.mkdir(parents=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")

    # Current contract (near-dated)
    pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_20260220_100",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "bid": 2.0,
                "ask": 2.2,
                "lastPrice": 2.1,
                "openInterest": 500,
                "volume": 20,
                "impliedVolatility": 0.25,
                "bs_delta": 0.50,
                "bs_theta_per_day": -0.0200,
            }
        ]
    ).to_csv(day_dir / "2026-02-20.csv", index=False)

    # Candidates near a 12-month horizon (target ~365 DTE from 2026-01-02).
    pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_20261218_100",
                "optionType": "call",
                "expiry": "2026-12-18",
                "strike": 100.0,
                "bid": 4.9,
                "ask": 5.1,
                "lastPrice": 5.0,
                "openInterest": 150,
                "volume": 12,
                "impliedVolatility": 0.30,
                "bs_delta": 0.60,
                "bs_theta_per_day": -0.0100,
            }
        ]
    ).to_csv(day_dir / "2026-12-18.csv", index=False)

    pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_20270115_100",
                "optionType": "call",
                "expiry": "2027-01-15",
                "strike": 100.0,
                "bid": 5.4,
                "ask": 5.6,
                "lastPrice": 5.5,
                "openInterest": 160,
                "volume": 11,
                "impliedVolatility": 0.31,
                "bs_delta": 0.62,
                "bs_theta_per_day": -0.0090,
            }
        ]
    ).to_csv(day_dir / "2027-01-15.csv", index=False)

    # Illiquid / wide-spread candidate should be gated out.
    pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_20270219_100",
                "optionType": "call",
                "expiry": "2027-02-19",
                "strike": 100.0,
                "bid": 0.10,
                "ask": 1.00,
                "lastPrice": 0.50,
                "openInterest": 1,
                "volume": 0,
                "impliedVolatility": 0.35,
                "bs_delta": 0.60,
                "bs_theta_per_day": -0.0080,
            }
        ]
    ).to_csv(day_dir / "2027-02-19.csv", index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "roll-plan",
            str(portfolio_path),
            "--id",
            "aaa-2026-02-20-100c",
            "--as-of",
            "2026-01-02",
            "--cache-dir",
            str(cache_dir),
            "--intent",
            "max-upside",
            "--horizon-months",
            "12",
            "--top",
            "5",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA roll plan as-of 2026-01-02" in res.output

    # 2027-01-15 is closer to a 12-month horizon than 2026-12-18, so it should rank ahead.
    assert res.output.index("2027-01-15") < res.output.index("2026-12-18")

    # Illiquid expiry should not appear in ranked candidates.
    assert "2027-02-19" not in res.output

