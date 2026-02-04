from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.schemas.flow import FlowArtifact


def test_flow_writes_schema_valid_json(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    cache_dir = tmp_path / "snapshots"
    day1 = cache_dir / "AAA" / "2026-01-01"
    day2 = cache_dir / "AAA" / "2026-01-02"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)

    df_prev = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260219C00010000",
                "optionType": "call",
                "expiry": "2026-02-19",
                "strike": 10.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "volume": 100,
                "openInterest": 100,
            }
        ]
    )
    df_today = df_prev.copy()
    df_today.loc[0, "openInterest"] = 160

    df_prev.to_csv(day1 / "2026-02-19.csv", index=False)
    df_today.to_csv(day2 / "2026-02-19.csv", index=False)

    out_dir = tmp_path / "reports"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "flow",
            str(portfolio_path),
            "--cache-dir",
            str(cache_dir),
            "--symbol",
            "AAA",
            "--window",
            "1",
            "--group-by",
            "contract",
            "--top",
            "1",
            "--out",
            str(out_dir),
            "--strict",
        ],
    )
    assert res.exit_code == 0, res.output

    out_path = out_dir / "flow" / "AAA" / "2026-01-01_to_2026-01-02_w1_contract.json"
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    FlowArtifact.model_validate(payload)
    assert payload["symbol"] == "AAA"
    assert payload["as_of"] == "2026-01-02"
    assert payload["window"] == 1
