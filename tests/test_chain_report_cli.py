from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from options_helper.cli import app


def test_chain_report_json_from_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    day_dir = cache_dir / "AAA" / "2026-01-02"
    day_dir.mkdir(parents=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")

    df = pd.DataFrame(
        [
            # Calls
            {
                "contractSymbol": "AAA_C_100",
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
                "bs_gamma": 0.020,
            },
            {
                "contractSymbol": "AAA_C_110",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 110.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "openInterest": 900,
                "volume": 15,
                "impliedVolatility": 0.30,
                "bs_delta": 0.25,
                "bs_gamma": 0.015,
            },
            # Puts
            {
                "contractSymbol": "AAA_P_100",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "bid": 1.3,
                "ask": 1.5,
                "lastPrice": 1.4,
                "openInterest": 600,
                "volume": 18,
                "impliedVolatility": 0.25,
                "bs_delta": -0.50,
                "bs_gamma": 0.020,
            },
            {
                "contractSymbol": "AAA_P_90",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 90.0,
                "bid": 0.9,
                "ask": 1.1,
                "lastPrice": 1.0,
                "openInterest": 1200,
                "volume": 12,
                "impliedVolatility": 0.35,
                "bs_delta": -0.25,
                "bs_gamma": 0.012,
            },
        ]
    )
    df.to_csv(day_dir / "2026-02-20.csv", index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "chain-report",
            "--symbol",
            "AAA",
            "--as-of",
            "2026-01-02",
            "--cache-dir",
            str(cache_dir),
            "--format",
            "json",
            "--expiries",
            "all",
            "--top",
            "3",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)

    assert payload["schema_version"] == 1
    assert "generated_at" in payload
    assert payload["symbol"] == "AAA"
    assert payload["as_of"] == "2026-01-02"
    assert payload["spot"] == 100.0

    # Totals
    assert payload["totals"]["calls_oi"] == 1400.0
    assert payload["totals"]["puts_oi"] == 1800.0
    assert payload["totals"]["pc_oi_ratio"] == pytest.approx(1800.0 / 1400.0)

    # Expected move: ATM straddle at strike 100 (mid prices).
    exp = payload["expiries"][0]
    assert exp["atm_strike"] == 100.0
    assert exp["call_mark_atm"] == pytest.approx(2.1)
    assert exp["put_mark_atm"] == pytest.approx(1.4)
    assert exp["expected_move"] == pytest.approx(3.5)
    assert exp["expected_move_pct"] == pytest.approx(0.035)

    # Skew: 25Δ put IV (0.35) - 25Δ call IV (0.30) = 5pp.
    assert exp["skew_25d_pp"] == pytest.approx(5.0)

    # Walls (overall)
    assert payload["walls_overall"]["calls"][0]["strike"] == 110.0
    assert payload["walls_overall"]["puts"][0]["strike"] == 90.0


def test_chain_report_writes_artifacts(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    day_dir = cache_dir / "AAA" / "2026-01-02"
    day_dir.mkdir(parents=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_100",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "lastPrice": 2.0,
                "openInterest": 1,
            },
            {
                "contractSymbol": "AAA_P_100",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "lastPrice": 1.0,
                "openInterest": 1,
            },
        ]
    )
    df.to_csv(day_dir / "2026-02-20.csv", index=False)

    out_dir = tmp_path / "reports"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "chain-report",
            "--symbol",
            "AAA",
            "--as-of",
            "2026-01-02",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    assert (out_dir / "chains" / "AAA" / "2026-01-02.json").exists()
    assert (out_dir / "chains" / "AAA" / "2026-01-02.md").exists()

    payload = json.loads((out_dir / "chains" / "AAA" / "2026-01-02.json").read_text(encoding="utf-8"))
    assert "generated_at" in payload
