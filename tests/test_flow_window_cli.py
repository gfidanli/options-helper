from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_flow_cli_windowed_groupby_writes_artifact(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    cache_dir = tmp_path / "snapshots"
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    spots = [100.0, 101.0, 102.0]

    for d, spot in zip(dates, spots, strict=False):
        day_dir = cache_dir / "AAA" / d
        day_dir.mkdir(parents=True)
        (day_dir / "meta.json").write_text(json.dumps({"spot": spot}), encoding="utf-8")

    expiry = "2026-02-20"
    df_day1 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_100",
                "optionType": "call",
                "expiry": expiry,
                "strike": 100.0,
                "lastPrice": 2.0,
                "volume": 10,
                "openInterest": 100,
                "bs_delta": 0.50,
            },
            {
                "contractSymbol": "AAA_P_90",
                "optionType": "put",
                "expiry": expiry,
                "strike": 90.0,
                "lastPrice": 1.5,
                "volume": 5,
                "openInterest": 200,
                "bs_delta": -0.25,
            },
        ]
    )
    df_day2 = df_day1.copy()
    df_day2.loc[df_day2["contractSymbol"] == "AAA_C_100", "openInterest"] = 150
    df_day2.loc[df_day2["contractSymbol"] == "AAA_P_90", "openInterest"] = 180
    df_day3 = df_day2.copy()
    df_day3.loc[df_day3["contractSymbol"] == "AAA_C_100", "openInterest"] = 130
    df_day3.loc[df_day3["contractSymbol"] == "AAA_P_90", "openInterest"] = 220

    (cache_dir / "AAA" / "2026-01-01" / f"{expiry}.csv").write_text(df_day1.to_csv(index=False), encoding="utf-8")
    (cache_dir / "AAA" / "2026-01-02" / f"{expiry}.csv").write_text(df_day2.to_csv(index=False), encoding="utf-8")
    (cache_dir / "AAA" / "2026-01-03" / f"{expiry}.csv").write_text(df_day3.to_csv(index=False), encoding="utf-8")

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
            "2",
            "--group-by",
            "expiry-strike",
            "--top",
            "10",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA flow net window=2 (2026-01-01 \u2192 2026-01-03) | group-by=expiry-strike" in res.output

    out_path = out_dir / "flow" / "AAA" / "2026-01-01_to_2026-01-03_w2_expiry-strike.json"
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "AAA"
    assert payload["window"] == 2
    assert payload["group_by"] == "expiry-strike"

    rows = payload["net"]
    call_row = next(r for r in rows if r["option_type"] == "call" and r["expiry"] == expiry and r["strike"] == 100.0)
    put_row = next(r for r in rows if r["option_type"] == "put" and r["expiry"] == expiry and r["strike"] == 90.0)

    # Net ΔOI: (150-100) + (130-150) = 30
    assert call_row["delta_oi"] == 30.0
    assert call_row["delta_oi_notional"] == 30.0 * 2.0 * 100.0

    # Delta-notional uses spot of each "today" snapshot for each pair:
    # (150-100)*0.50*101*100 + (130-150)*0.50*102*100 = 150_500
    assert call_row["delta_notional"] == 150_500.0

    # Put net ΔOI: (180-200) + (220-180) = 20
    assert put_row["delta_oi"] == 20.0
    assert put_row["delta_oi_notional"] == 20.0 * 1.5 * 100.0
    assert put_row["delta_notional"] == -51_500.0
