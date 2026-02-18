from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import options_helper.cli as cli
from options_helper.data.candles import CandleStore
from options_helper.data.technical_backtesting_config import ConfigError as TechnicalConfigError
from options_helper.data.journal import JournalStore, SignalContext, SignalEvent
from options_helper.data.market_types import OptionsChain
from options_helper.data.options_snapshots import OptionsSnapshotStore


def _write_offline_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "medium",
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

    return portfolio_path, candle_dir, snapshots_dir


def test_journal_log_offline_writes_position_event(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path, candle_dir, snapshots_dir = _write_offline_fixtures(tmp_path)
    journal_dir = tmp_path / "journal"

    def _boom_provider(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("build_provider should not be called in --offline mode")

    monkeypatch.setattr("options_helper.cli_deps.build_provider", _boom_provider)

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "journal",
            "log",
            str(portfolio_path),
            "--offline",
            "--as-of",
            "latest",
            "--snapshots-dir",
            str(snapshots_dir),
            "--cache-dir",
            str(candle_dir),
            "--journal-dir",
            str(journal_dir),
        ],
    )

    assert res.exit_code == 0, res.output

    store = JournalStore(journal_dir)
    result = store.read_events()
    assert result.errors == []
    assert len(result.events) == 1

    event = result.events[0]
    assert event.context == SignalContext.POSITION
    assert event.symbol == "AAA"
    assert event.snapshot_date == date(2026, 2, 2)
    assert event.contract_symbol == "AAA260417C00005000"

    payload = event.payload
    assert payload["position"]["id"] == "a"
    assert payload["as_of"] == "2026-02-02"
    assert payload["metrics"]["mark"] == pytest.approx(1.1)


def test_journal_log_scanner_reads_latest_run(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(json.dumps({"cash": 0, "positions": []}), encoding="utf-8")

    run_dir = tmp_path / "scanner_runs" / "2026-02-02_120000"
    run_dir.mkdir(parents=True)
    shortlist_csv = run_dir / "shortlist.csv"
    shortlist_csv.write_text(
        "symbol,score,coverage,top_reasons\nAAA,88,0.9,trend;iv\nBBB,77,0.8,trend\n",
        encoding="utf-8",
    )

    journal_dir = tmp_path / "journal"
    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "journal",
            "log",
            str(portfolio_path),
            "--no-positions",
            "--scanner",
            "--scanner-run-dir",
            str(tmp_path / "scanner_runs"),
            "--journal-dir",
            str(journal_dir),
        ],
    )

    assert res.exit_code == 0, res.output

    store = JournalStore(journal_dir)
    result = store.read_events()
    assert len(result.events) == 2
    assert {e.symbol for e in result.events} == {"AAA", "BBB"}
    assert all(e.context == SignalContext.SCANNER for e in result.events)
    assert all(e.date == date(2026, 2, 2) for e in result.events)


def test_journal_log_research_uses_stubbed_client(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(json.dumps({"cash": 0, "positions": []}), encoding="utf-8")

    history = pd.DataFrame(
        {"Close": [10.0, 10.5, 11.0, 11.5, 12.0]},
        index=pd.to_datetime(
            ["2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30", "2026-02-02"]
        ),
    )

    def _fake_history(self, symbol: str, period: str = "5y") -> pd.DataFrame:  # noqa: ANN001
        return history

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _fake_history)

    def _fake_config(*_args, **_kwargs):  # noqa: ANN001
        raise TechnicalConfigError("no config")

    monkeypatch.setattr("options_helper.commands.journal.load_technical_backtesting_config", _fake_config)

    class StubProvider:
        def list_option_expiries(self, symbol: str):  # noqa: ARG002
            return [date(2026, 3, 15), date(2027, 1, 15)]

        def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
            calls = pd.DataFrame(
                [
                    {
                        "contractSymbol": "AAA260315C00005000",
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
            puts = pd.DataFrame(
                [
                    {
                        "contractSymbol": "AAA260315P00005000",
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
            return OptionsChain(symbol=symbol.upper(), expiry=expiry, calls=calls, puts=puts)

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda *_args, **_kwargs: StubProvider())

    journal_dir = tmp_path / "journal"
    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "journal",
            "log",
            str(portfolio_path),
            "--no-positions",
            "--research",
            "--research-symbol",
            "AAA",
            "--journal-dir",
            str(journal_dir),
        ],
    )

    assert res.exit_code == 0, res.output

    store = JournalStore(journal_dir)
    result = store.read_events()
    assert len(result.events) == 1
    event = result.events[0]
    assert event.context == SignalContext.RESEARCH
    assert event.symbol == "AAA"
    payload = event.payload
    assert payload["confluence"] is not None


def test_journal_evaluate_writes_reports(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    journal_dir = tmp_path / "journal"
    candle_dir = tmp_path / "candles"
    snapshots_dir = tmp_path / "snapshots"
    out_dir = tmp_path / "reports"
    monkeypatch.setattr("options_helper.cli_deps.build_journal_store", lambda root: JournalStore(root))
    monkeypatch.setattr("options_helper.cli_deps.build_candle_store", lambda root: CandleStore(root))
    monkeypatch.setattr("options_helper.cli_deps.build_snapshot_store", lambda root: OptionsSnapshotStore(root))

    store = JournalStore(journal_dir)
    store.append_event(
        SignalEvent(
            date=date(2026, 2, 2),
            symbol="AAA",
            context=SignalContext.POSITION,
            payload={"advice": {"action": "HOLD"}},
            snapshot_date=date(2026, 2, 2),
            contract_symbol="AAA260417C00005000",
        )
    )

    candle_dir.mkdir(parents=True, exist_ok=True)
    candles = pd.DataFrame(
        {"Close": [10.0, 12.0], "Volume": [1000, 1200]},
        index=pd.to_datetime(["2026-02-02", "2026-02-03"]),
    )
    candles.to_csv(candle_dir / "AAA.csv")

    start_day = snapshots_dir / "AAA" / "2026-02-02"
    start_day.mkdir(parents=True)
    pd.DataFrame(
        [{"contractSymbol": "AAA260417C00005000", "bid": 1.0, "ask": 1.0, "lastPrice": 1.0}]
    ).to_csv(start_day / "2026-04-17.csv", index=False)

    end_day = snapshots_dir / "AAA" / "2026-02-03"
    end_day.mkdir(parents=True)
    pd.DataFrame(
        [{"contractSymbol": "AAA260417C00005000", "bid": 1.5, "ask": 1.5, "lastPrice": 1.5}]
    ).to_csv(end_day / "2026-04-17.csv", index=False)

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "journal",
            "evaluate",
            "--journal-dir",
            str(journal_dir),
            "--cache-dir",
            str(candle_dir),
            "--snapshots-dir",
            str(snapshots_dir),
            "--as-of",
            "2026-02-03",
            "--window",
            "10",
            "--horizons",
            "1",
            "--out-dir",
            str(out_dir),
            "--top",
            "2",
        ],
    )

    assert res.exit_code == 0, res.output
    report_path = out_dir / "2026-02-03.json"
    md_path = out_dir / "2026-02-03.md"
    assert report_path.exists()
    assert md_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["as_of"] == "2026-02-03"
    assert report["window_days"] == 10
    event_outcomes = report["events"][0]["outcomes"]["1"]
    assert event_outcomes["underlying_return"] == pytest.approx(0.2)
    assert event_outcomes["option_return"] == pytest.approx(0.5)


def test_journal_evaluate_exits_when_window_has_no_events(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    journal_dir = tmp_path / "journal"
    monkeypatch.setattr("options_helper.cli_deps.build_journal_store", lambda root: JournalStore(root))
    monkeypatch.setattr("options_helper.cli_deps.build_candle_store", lambda root: CandleStore(root))
    monkeypatch.setattr("options_helper.cli_deps.build_snapshot_store", lambda root: OptionsSnapshotStore(root))
    store = JournalStore(journal_dir)
    store.append_event(
        SignalEvent(
            date=date(2026, 1, 1),
            symbol="AAA",
            context=SignalContext.SCANNER,
            payload={},
        )
    )

    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "journal",
            "evaluate",
            "--journal-dir",
            str(journal_dir),
            "--as-of",
            "2026-02-01",
            "--window",
            "5",
            "--out-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert res.exit_code == 0, res.output
    assert "No journal events within the window." in res.output
