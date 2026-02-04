from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import options_helper.cli as cli


def test_analyze_offline_uses_snapshots_and_never_instantiates_yfinance(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                },
                "positions": [
                    {
                        "id": "a",
                        "symbol": "AAA",
                        "option_type": "call",
                        "expiry": "2026-04-17",
                        "strike": 5.0,
                        "contracts": 1,
                        "cost_basis": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candle_dir = tmp_path / "candles"
    candle_dir.mkdir(parents=True, exist_ok=True)
    candles = pd.DataFrame(
        {"Close": [10.0, 11.0], "Volume": [1000, 1200]},
        index=pd.to_datetime(["2026-01-31", "2026-02-02"]),
    )
    candles.to_csv(candle_dir / "AAA.csv")

    snapshots_dir = tmp_path / "snapshots"
    day_dir = snapshots_dir / "AAA" / "2026-02-02"
    day_dir.mkdir(parents=True)
    snap = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "optionType": "call",
                "expiry": "2026-04-17",
                "strike": 5.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.50,
                "openInterest": 500,
                "volume": 20,
            }
        ]
    )
    snap.to_csv(day_dir / "2026-04-17.csv", index=False)

    def _boom_provider(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("build_provider should not be called in --offline mode")

    monkeypatch.setattr("options_helper.cli_deps.build_provider", _boom_provider)

    original_position_metrics = cli._position_metrics
    seen: dict[str, object] = {}

    def _wrapped_position_metrics(*args, **kwargs):  # noqa: ANN001
        seen["snapshot_row"] = kwargs.get("snapshot_row")
        return original_position_metrics(*args, **kwargs)

    monkeypatch.setattr(cli, "_position_metrics", _wrapped_position_metrics)

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "analyze",
            str(portfolio_path),
            "--offline",
            "--as-of",
            "latest",
            "--snapshots-dir",
            str(snapshots_dir),
            "--cache-dir",
            str(candle_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA" in res.output
    row = seen.get("snapshot_row")
    assert row is not None
    assert getattr(row, "get", lambda *_: None)("contractSymbol") == "AAA260417C00005000"


def _write_offline_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                },
                "positions": [
                    {
                        "id": "a",
                        "symbol": "AAA",
                        "option_type": "call",
                        "expiry": "2026-04-17",
                        "strike": 5.0,
                        "contracts": 1,
                        "cost_basis": 1.0,
                    },
                    {
                        "id": "b",
                        "symbol": "AAA",
                        "option_type": "call",
                        "expiry": "2026-04-17",
                        "strike": 6.0,
                        "contracts": 1,
                        "cost_basis": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candle_dir = tmp_path / "candles"
    candle_dir.mkdir(parents=True, exist_ok=True)
    candles = pd.DataFrame(
        {"Close": [10.0, 11.0], "Volume": [1000, 1200]},
        index=pd.to_datetime(["2026-01-31", "2026-02-02"]),
    )
    candles.to_csv(candle_dir / "AAA.csv")

    snapshots_dir = tmp_path / "snapshots"
    day_dir = snapshots_dir / "AAA" / "2026-02-02"
    day_dir.mkdir(parents=True)
    snap = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "optionType": "call",
                "expiry": "2026-04-17",
                "strike": 5.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.50,
                "openInterest": 500,
                "volume": 20,
            }
        ]
    )
    snap.to_csv(day_dir / "2026-04-17.csv", index=False)

    return portfolio_path, candle_dir, snapshots_dir


def test_analyze_offline_warns_when_snapshot_row_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path, candle_dir, snapshots_dir = _write_offline_fixtures(tmp_path)

    def _boom_provider(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("build_provider should not be called in --offline mode")

    monkeypatch.setattr("options_helper.cli_deps.build_provider", _boom_provider)

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "analyze",
            str(portfolio_path),
            "--offline",
            "--as-of",
            "latest",
            "--snapshots-dir",
            str(snapshots_dir),
            "--cache-dir",
            str(candle_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Warning:" in res.output
    assert "b: missing snapshot row" in res.output


def test_analyze_offline_strict_exits_nonzero_when_snapshot_row_missing(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    portfolio_path, candle_dir, snapshots_dir = _write_offline_fixtures(tmp_path)

    def _boom_provider(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("build_provider should not be called in --offline mode")

    monkeypatch.setattr("options_helper.cli_deps.build_provider", _boom_provider)

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "analyze",
            str(portfolio_path),
            "--offline",
            "--offline-strict",
            "--as-of",
            "latest",
            "--snapshots-dir",
            str(snapshots_dir),
            "--cache-dir",
            str(candle_dir),
        ],
    )
    assert res.exit_code == 1, res.output
    assert "b: missing snapshot row" in res.output


def test_analyze_offline_renders_multileg_summary(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                },
                "positions": [
                    {
                        "id": "spread-1",
                        "symbol": "AAA",
                        "net_debit": 150.0,
                        "legs": [
                            {
                                "side": "long",
                                "option_type": "call",
                                "expiry": "2026-04-17",
                                "strike": 5.0,
                                "contracts": 1,
                            },
                            {
                                "side": "short",
                                "option_type": "call",
                                "expiry": "2026-04-17",
                                "strike": 6.0,
                                "contracts": 1,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candle_dir = tmp_path / "candles"
    candle_dir.mkdir(parents=True, exist_ok=True)
    candles = pd.DataFrame(
        {"Close": [10.0, 11.0], "Volume": [1000, 1200]},
        index=pd.to_datetime(["2026-01-31", "2026-02-02"]),
    )
    candles.to_csv(candle_dir / "AAA.csv")

    snapshots_dir = tmp_path / "snapshots"
    day_dir = snapshots_dir / "AAA" / "2026-02-02"
    day_dir.mkdir(parents=True)
    snap = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "optionType": "call",
                "expiry": "2026-04-17",
                "strike": 5.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.50,
                "openInterest": 500,
                "volume": 20,
            },
            {
                "contractSymbol": "AAA260417C00006000",
                "optionType": "call",
                "expiry": "2026-04-17",
                "strike": 6.0,
                "bid": 0.4,
                "ask": 0.6,
                "lastPrice": 0.5,
                "impliedVolatility": 0.45,
                "openInterest": 400,
                "volume": 18,
            },
        ]
    )
    snap.to_csv(day_dir / "2026-04-17.csv", index=False)

    def _boom_provider(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("build_provider should not be called in --offline mode")

    monkeypatch.setattr("options_helper.cli_deps.build_provider", _boom_provider)

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "analyze",
            str(portfolio_path),
            "--offline",
            "--as-of",
            "latest",
            "--snapshots-dir",
            str(snapshots_dir),
            "--cache-dir",
            str(candle_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Multi-leg Positions" in res.output
    assert "spread-1" in res.output
