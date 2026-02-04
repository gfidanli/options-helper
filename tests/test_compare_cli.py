from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_compare_cli_outputs_and_writes_json(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"

    day1 = cache_dir / "AAA" / "2026-01-01"
    day2 = cache_dir / "AAA" / "2026-01-02"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)

    (day1 / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")
    (day2 / "meta.json").write_text(json.dumps({"spot": 102.0}), encoding="utf-8")

    df1 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_110",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 110.0,
                "lastPrice": 1.0,
                "openInterest": 100,
                "volume": 10,
            },
            {
                "contractSymbol": "AAA_P_90",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 90.0,
                "lastPrice": 1.5,
                "openInterest": 200,
                "volume": 5,
            },
            # ATM straddle + IV/skew fields
            {
                "contractSymbol": "AAA_C_100",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "lastPrice": 2.0,
                "openInterest": 50,
                "impliedVolatility": 0.25,
                "bs_delta": 0.50,
                "bs_gamma": 0.010,
            },
            {
                "contractSymbol": "AAA_P_100",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "lastPrice": 1.0,
                "openInterest": 75,
                "impliedVolatility": 0.25,
                "bs_delta": -0.50,
                "bs_gamma": 0.010,
            },
        ]
    )
    df2 = df1.copy()
    df2.loc[df2["contractSymbol"] == "AAA_C_110", "openInterest"] = 150
    df2.loc[df2["contractSymbol"] == "AAA_P_90", "openInterest"] = 180

    df1.to_csv(day1 / "2026-02-20.csv", index=False)
    df2.to_csv(day2 / "2026-02-20.csv", index=False)

    out_dir = tmp_path / "reports"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compare",
            "--symbol",
            "AAA",
            "--from",
            "2026-01-01",
            "--to",
            "2026-01-02",
            "--cache-dir",
            str(cache_dir),
            "--top",
            "3",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA compare 2026-01-01 \u2192 2026-01-02" in res.output

    out_path = out_dir / "compare" / "AAA" / "2026-01-01_to_2026-01-02.json"
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["as_of"] == "2026-01-02"
    assert "generated_at" in payload
    assert payload["symbol"] == "AAA"
    assert payload["diff"]["spot_change"] == 2.0


def test_compare_cli_relative_from_latest(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    day1 = cache_dir / "AAA" / "2026-01-01"
    day2 = cache_dir / "AAA" / "2026-01-02"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)

    (day1 / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")
    (day2 / "meta.json").write_text(json.dumps({"spot": 101.0}), encoding="utf-8")

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
    df.to_csv(day1 / "2026-02-20.csv", index=False)
    df.to_csv(day2 / "2026-02-20.csv", index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compare",
            "--symbol",
            "AAA",
            "--to",
            "latest",
            "--from",
            "-1",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA compare 2026-01-01 \u2192 2026-01-02" in res.output


def test_compare_cli_mixed_schema_osi_no_deltas_omits_flow_tables(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    day1 = cache_dir / "AAA" / "2026-01-01"
    day2 = cache_dir / "AAA" / "2026-01-02"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)

    (day1 / "meta.json").write_text(json.dumps({"spot": 100.0}), encoding="utf-8")
    (day2 / "meta.json").write_text(json.dumps({"spot": 101.0}), encoding="utf-8")

    df1 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_100",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "lastPrice": 2.0,
                "openInterest": 10,
                "volume": 1,
            },
            {
                "contractSymbol": "AAA_P_100",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "lastPrice": 1.0,
                "openInterest": 20,
                "volume": 1,
            },
        ]
    )
    df2 = df1.copy()
    df2["osi"] = ["AAA   260220C00100000", "AAA   260220P00100000"]

    df1.to_csv(day1 / "2026-02-20.csv", index=False)
    df2.to_csv(day2 / "2026-02-20.csv", index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "compare",
            "--symbol",
            "AAA",
            "--from",
            "2026-01-01",
            "--to",
            "2026-01-02",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "oi_unchanged_between_snapshots" in res.output
    assert "Top contracts by |ΔOI$|" not in res.output
