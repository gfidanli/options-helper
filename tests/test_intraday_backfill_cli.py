from __future__ import annotations

import json
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.pipelines import intraday_backfill_runtime as runtime


class _StubProvider:
    name = "alpaca"


class _StubClient:
    provider_version = "test"

    def __init__(self, *, stock_feed: str | None = None) -> None:
        self.stock_feed = stock_feed
        self.calls: list[dict[str, object]] = []

    def get_stock_bars(self, symbol: str, *, start, end, interval: str, adjustment: str):  # noqa: ANN001
        self.calls.append(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "interval": interval,
                "adjustment": adjustment,
            }
        )
        if interval == "1d":
            return pd.DataFrame(
                {
                    "Open": [10.0],
                    "High": [11.0],
                    "Low": [9.5],
                    "Close": [10.5],
                    "Volume": [1000],
                },
                index=pd.to_datetime(["2026-01-02T00:00:00Z"], utc=True),
            )
        return pd.DataFrame(
            {
                "Open": [10.0, 10.1],
                "High": [10.2, 10.3],
                "Low": [9.9, 10.0],
                "Close": [10.1, 10.2],
                "Volume": [100, 120],
            },
            index=pd.to_datetime(["2026-01-02T14:30:00Z", "2026-01-02T14:31:00Z"], utc=True),
        )


def _monkeypatch_common(monkeypatch, *, symbols: list[str]) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.commands.intraday_backfill.cli_deps.build_provider", lambda: _StubProvider())
    monkeypatch.setattr(runtime, "AlpacaClient", _StubClient)
    monkeypatch.setattr(runtime, "_load_market_tz", lambda: ZoneInfo("America/New_York"))
    monkeypatch.setattr(runtime, "resolve_end_day", lambda **_k: date(2026, 1, 31))
    monkeypatch.setattr(runtime, "load_market_days", lambda **_k: [date(2026, 1, 2)])
    monkeypatch.setattr(runtime, "load_target_symbols", lambda **_k: list(symbols))


def test_intraday_backfill_writes_symbol_month_status_and_pauses(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    _monkeypatch_common(monkeypatch, symbols=["AAPL"])
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "intraday-backfill",
            "stocks-history",
            "--out-dir",
            str(tmp_path / "intraday"),
            "--status-dir",
            str(tmp_path / "status"),
            "--run-id",
            "run-a",
            "--checkpoint-symbols",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Paused after checkpoint" in result.output

    day_path = tmp_path / "intraday" / "stocks" / "bars" / "1Min" / "AAPL" / "2026-01-02.csv.gz"
    assert day_path.exists()

    overall = json.loads((tmp_path / "status" / "run-a" / "status" / "overall.json").read_text())
    symbol_status = json.loads((tmp_path / "status" / "run-a" / "status" / "symbols" / "AAPL.json").read_text())
    assert overall["symbols_processed"] == 1
    assert overall["months_processed"] == 1
    assert symbol_status["months"][0]["year_month"] == "2026-01"
    assert symbol_status["months"][0]["status"] == "ok"

    checkpoint_md = tmp_path / "status" / "run-a" / "performance_checkpoint_25_symbols.md"
    checkpoint_json = tmp_path / "status" / "run-a" / "performance_checkpoint_25_symbols.json"
    assert checkpoint_md.exists()
    assert checkpoint_json.exists()


def test_intraday_backfill_no_pause_continues_past_checkpoint(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    _monkeypatch_common(monkeypatch, symbols=["AAPL", "MSFT"])
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "intraday-backfill",
            "stocks-history",
            "--out-dir",
            str(tmp_path / "intraday"),
            "--status-dir",
            str(tmp_path / "status"),
            "--run-id",
            "run-b",
            "--checkpoint-symbols",
            "1",
            "--no-pause-at-checkpoint",
        ],
    )

    assert result.exit_code == 0, result.output
    overall = json.loads((tmp_path / "status" / "run-b" / "status" / "overall.json").read_text())
    assert overall["symbols_processed"] == 2

    aapl_path = tmp_path / "intraday" / "stocks" / "bars" / "1Min" / "AAPL" / "2026-01-02.csv.gz"
    msft_path = tmp_path / "intraday" / "stocks" / "bars" / "1Min" / "MSFT" / "2026-01-02.csv.gz"
    assert aapl_path.exists()
    assert msft_path.exists()


def test_intraday_backfill_skips_month_when_existing_coverage(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    _monkeypatch_common(monkeypatch, symbols=["AAPL"])
    runner = CliRunner()

    existing_root = tmp_path / "intraday"
    existing_day = existing_root / "stocks" / "bars" / "1Min" / "AAPL"
    existing_day.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-02T14:30:00Z"],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [100],
        }
    ).to_csv(existing_day / "2026-01-02.csv.gz", index=False, compression="gzip")

    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "intraday-backfill",
            "stocks-history",
            "--out-dir",
            str(existing_root),
            "--status-dir",
            str(tmp_path / "status"),
            "--run-id",
            "run-c",
            "--checkpoint-symbols",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    symbol_status = json.loads((tmp_path / "status" / "run-c" / "status" / "symbols" / "AAPL.json").read_text())
    assert symbol_status["months"][0]["status"] == "skipped_existing"
