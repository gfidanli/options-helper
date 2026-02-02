from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def _write_minimal_snapshot_day(day_dir: Path, *, spot: float, oi: int) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "meta.json").write_text(json.dumps({"spot": spot}), encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260220C00010000",
                "optionType": "call",
                "expiry": "2026-02-20",
                "strike": 10.0,
                "lastPrice": 2.0,
                "volume": 10,
                "openInterest": oi,
                "impliedVolatility": 0.25,
                "bs_delta": 0.5,
                "bs_gamma": 0.01,
            },
            {
                "contractSymbol": "AAA260220P00010000",
                "optionType": "put",
                "expiry": "2026-02-20",
                "strike": 10.0,
                "lastPrice": 1.5,
                "volume": 5,
                "openInterest": oi,
                "impliedVolatility": 0.30,
                "bs_delta": -0.5,
                "bs_gamma": 0.01,
            },
        ]
    )
    df.to_csv(day_dir / "2026-02-20.csv", index=False)


def test_report_pack_writes_core_artifacts(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"positions":["AAA"],"monitor":["AAA"],"Scanner - Shortlist":["AAA"]}}',
        encoding="utf-8",
    )

    cache_dir = tmp_path / "snapshots"
    _write_minimal_snapshot_day(cache_dir / "AAA" / "2026-01-01", spot=10.0, oi=100)
    _write_minimal_snapshot_day(cache_dir / "AAA" / "2026-01-02", spot=10.5, oi=150)

    out_dir = tmp_path / "reports"
    derived_dir = tmp_path / "derived"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "report-pack",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--cache-dir",
            str(cache_dir),
            "--derived-dir",
            str(derived_dir),
            "--out",
            str(out_dir),
            "--require-snapshot-date",
            "2026-01-02",
            "--no-technicals",
        ],
    )
    assert res.exit_code == 0, res.output

    assert (out_dir / "chains" / "AAA" / "2026-01-02.json").exists()
    assert (out_dir / "chains" / "AAA" / "2026-01-02.md").exists()
    assert (out_dir / "compare" / "AAA" / "2026-01-01_to_2026-01-02.json").exists()
    assert (out_dir / "flow" / "AAA" / "2026-01-01_to_2026-01-02_w1_contract.json").exists()
    assert (out_dir / "flow" / "AAA" / "2026-01-01_to_2026-01-02_w1_expiry-strike.json").exists()
    assert (out_dir / "derived" / "AAA" / "2026-01-02_w60_tw5.json").exists()


def test_report_pack_can_write_technicals_artifacts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"positions":["AAA"],"monitor":["AAA"],"Scanner - Shortlist":["AAA"]}}',
        encoding="utf-8",
    )

    cache_dir = tmp_path / "snapshots"
    _write_minimal_snapshot_day(cache_dir / "AAA" / "2026-01-02", spot=10.5, oi=150)

    out_dir = tmp_path / "reports"
    derived_dir = tmp_path / "derived"

    def _stub_extension_stats(*, symbol, out, **_kwargs):  # noqa: ANN001
        base = Path(out) / str(symbol).upper()
        base.mkdir(parents=True, exist_ok=True)
        (base / "2026-01-02.json").write_text("{}", encoding="utf-8")
        (base / "2026-01-02.md").write_text("# stub\n", encoding="utf-8")

    monkeypatch.setattr("options_helper.cli.technicals_extension_stats", _stub_extension_stats)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "report-pack",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--cache-dir",
            str(cache_dir),
            "--derived-dir",
            str(derived_dir),
            "--out",
            str(out_dir),
            "--require-snapshot-date",
            "2026-01-02",
        ],
    )
    assert res.exit_code == 0, res.output

    assert (out_dir / "technicals" / "extension" / "AAA" / "2026-01-02.json").exists()
    assert (out_dir / "technicals" / "extension" / "AAA" / "2026-01-02.md").exists()

