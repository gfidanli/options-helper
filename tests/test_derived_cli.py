from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from options_helper.cli import app


def _write_snapshot_day(cache_dir: Path, *, symbol: str, day: str, spot: float) -> None:
    day_dir = cache_dir / symbol / day
    day_dir.mkdir(parents=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": spot}), encoding="utf-8")

    df = pd.DataFrame(
        [
            # Calls
            {
                "contractSymbol": f"{symbol}_C_100",
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
                "contractSymbol": f"{symbol}_C_110",
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
                "contractSymbol": f"{symbol}_P_100",
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
                "contractSymbol": f"{symbol}_P_90",
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


def test_derived_update_is_idempotent(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    derived_dir = tmp_path / "derived"
    _write_snapshot_day(cache_dir, symbol="AAA", day="2026-01-02", spot=100.0)

    runner = CliRunner()
    for _ in range(2):
        res = runner.invoke(
            app,
            [
                "derived",
                "update",
                "--symbol",
                "AAA",
                "--as-of",
                "2026-01-02",
                "--cache-dir",
                str(cache_dir),
                "--derived-dir",
                str(derived_dir),
            ],
        )
        assert res.exit_code == 0, res.output

    out_path = derived_dir / "AAA.csv"
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert df.shape[0] == 1
    row = df.iloc[0].to_dict()

    assert row["date"] == "2026-01-02"
    assert float(row["spot"]) == pytest.approx(100.0)
    assert float(row["call_wall"]) == pytest.approx(110.0)
    assert float(row["put_wall"]) == pytest.approx(90.0)
    assert float(row["gamma_peak_strike"]) == pytest.approx(100.0)
    assert float(row["atm_iv_near"]) == pytest.approx(0.25)
    assert float(row["em_near_pct"]) == pytest.approx(0.035)
    assert float(row["skew_near_pp"]) == pytest.approx(5.0)


def test_derived_show_prints_rows(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    derived_dir = tmp_path / "derived"
    _write_snapshot_day(cache_dir, symbol="AAA", day="2026-01-02", spot=100.0)

    runner = CliRunner()
    res_up = runner.invoke(
        app,
        [
            "derived",
            "update",
            "--symbol",
            "AAA",
            "--as-of",
            "2026-01-02",
            "--cache-dir",
            str(cache_dir),
            "--derived-dir",
            str(derived_dir),
        ],
    )
    assert res_up.exit_code == 0, res_up.output

    res = runner.invoke(
        app,
        [
            "derived",
            "show",
            "--symbol",
            "AAA",
            "--derived-dir",
            str(derived_dir),
            "--last",
            "10",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA derived metrics" in res.output
    assert "2026-01-02" in res.output


def test_derived_stats_json_percentiles_and_trends(tmp_path: Path) -> None:
    cache_dir = tmp_path / "snapshots"
    derived_dir = tmp_path / "derived"

    _write_snapshot_day(cache_dir, symbol="AAA", day="2026-01-01", spot=95.0)
    _write_snapshot_day(cache_dir, symbol="AAA", day="2026-01-02", spot=100.0)
    _write_snapshot_day(cache_dir, symbol="AAA", day="2026-01-03", spot=105.0)

    runner = CliRunner()
    for day in ["2026-01-01", "2026-01-02", "2026-01-03"]:
        res = runner.invoke(
            app,
            [
                "derived",
                "update",
                "--symbol",
                "AAA",
                "--as-of",
                day,
                "--cache-dir",
                str(cache_dir),
                "--derived-dir",
                str(derived_dir),
            ],
        )
        assert res.exit_code == 0, res.output

    res = runner.invoke(
        app,
        [
            "derived",
            "stats",
            "--symbol",
            "AAA",
            "--as-of",
            "latest",
            "--derived-dir",
            str(derived_dir),
            "--window",
            "3",
            "--trend-window",
            "3",
            "--format",
            "json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)

    assert payload["symbol"] == "AAA"
    assert payload["as_of"] == "2026-01-03"

    metrics = {m["name"]: m for m in payload["metrics"]}
    assert metrics["spot"]["value"] == pytest.approx(105.0)
    assert metrics["spot"]["percentile"] == pytest.approx(100.0)
    assert metrics["spot"]["trend_direction"] == "up"
    assert metrics["spot"]["trend_delta"] == pytest.approx(10.0)

    # Constant values should be ~50th percentile and flat.
    assert metrics["atm_iv_near"]["percentile"] == pytest.approx(50.0)
    assert metrics["atm_iv_near"]["trend_direction"] == "flat"
