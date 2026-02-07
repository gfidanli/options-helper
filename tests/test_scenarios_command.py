from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_scenarios_command_writes_artifacts_under_portfolio_date(tmp_path: Path, monkeypatch) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0.0,
                "positions": [
                    {
                        "id": "call-1",
                        "symbol": "AAA",
                        "option_type": "call",
                        "expiry": "2026-03-20",
                        "strike": 100.0,
                        "contracts": 1,
                        "cost_basis": 2.0,
                    },
                    {
                        "id": "put-1",
                        "symbol": "AAA",
                        "option_type": "put",
                        "expiry": "2026-03-20",
                        "strike": 90.0,
                        "contracts": 1,
                        "cost_basis": 1.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    snapshots_dir = tmp_path / "snapshots"
    day_dir = snapshots_dir / "AAA" / "2026-02-07"
    day_dir.mkdir(parents=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260320C00100000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 100.0,
                "bid": 2.0,
                "ask": 2.2,
                "lastPrice": 2.1,
                "impliedVolatility": 0.25,
            }
        ]
    ).to_csv(day_dir / "2026-03-20.csv", index=False)

    def _boom_provider(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("build_provider should not be called by scenarios command")

    monkeypatch.setattr("options_helper.cli_deps.build_provider", _boom_provider)

    out_dir = tmp_path / "reports"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "scenarios",
            str(portfolio_path),
            "--as-of",
            "latest",
            "--cache-dir",
            str(snapshots_dir),
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Position Scenarios" in res.output
    assert "2026-02-07" in res.output
    assert "missing_snapshot_row" in res.output

    scenario_dir = out_dir / "scenarios" / "2026-02-07"
    files = sorted(scenario_dir.glob("*.json"))
    assert len(files) == 2

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    by_type = {payload["summary"]["option_type"]: payload for payload in payloads}
    assert set(by_type) == {"call", "put"}

    call_payload = by_type["call"]
    assert call_payload["as_of"] == "2026-02-07"
    assert call_payload["summary"]["warnings"] == []
    assert len(call_payload["grid"]) > 0

    put_payload = by_type["put"]
    put_warnings = set(put_payload["summary"]["warnings"])
    assert "missing_snapshot_row" in put_warnings
    assert "missing_iv" in put_warnings
    assert "missing_mark" in put_warnings
    assert put_payload["grid"] == []


def test_scenarios_command_degrades_gracefully_when_snapshot_day_missing(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0.0,
                "positions": [
                    {
                        "id": "bbb-call",
                        "symbol": "BBB",
                        "option_type": "call",
                        "expiry": "2026-03-20",
                        "strike": 50.0,
                        "contracts": 1,
                        "cost_basis": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "reports"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "scenarios",
            str(portfolio_path),
            "--as-of",
            "2026-02-01",
            "--cache-dir",
            str(tmp_path / "no-snapshots"),
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "missing_snapshot_day" in res.output
    assert "missing_snapshot_row" in res.output

    scenario_dir = out_dir / "scenarios" / "2026-02-01"
    files = sorted(scenario_dir.glob("*.json"))
    assert len(files) == 1

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["as_of"] == "2026-02-01"
    warnings = set(payload["summary"]["warnings"])
    assert "missing_snapshot_day" in warnings
    assert "missing_snapshot_row" in warnings
    assert "missing_spot" in warnings
    assert "missing_iv" in warnings
    assert "missing_mark" in warnings
    assert payload["grid"] == []
