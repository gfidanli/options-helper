from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


class _StubProvider:
    name = "alpaca"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_history(
        self,
        symbol: str,
        *,
        start,  # noqa: ANN001
        end,  # noqa: ANN001
        interval: str,  # noqa: ARG002
        auto_adjust: bool,  # noqa: ARG002
        back_adjust: bool,  # noqa: ARG002
    ):
        self.calls.append(symbol)
        idx = pd.date_range(datetime(2026, 2, 1), periods=3, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0, 1.1, 1.2],
                "High": [1.2, 1.3, 1.4],
                "Low": [0.9, 1.0, 1.1],
                "Close": [1.1, 1.2, 1.3],
                "Volume": [100, 110, 120],
            },
            index=idx,
        )


def test_ingest_candles_defaults(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"positions":["AAA"],"monitor":["BBB"]}}',
        encoding="utf-8",
    )

    provider = _StubProvider()
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: provider)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    result = runner.invoke(
        app,
        [
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "candles",
            "--watchlists-path",
            str(watchlists_path),
            "--candle-cache-dir",
            str(tmp_path / "candles"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(provider.calls) == ["AAA", "BBB"]

    wh = DuckDBWarehouse(duckdb_path)
    ensure_schema(wh)
    df = wh.fetch_df(
        "SELECT DISTINCT symbol FROM candles_daily ORDER BY symbol ASC"
    )
    assert list(df["symbol"]) == ["AAA", "BBB"]


def test_ingest_candles_auto_tune_writes_profile(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        '{"watchlists":{"positions":["AAA"]}}',
        encoding="utf-8",
    )

    provider = _StubProvider()
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: provider)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    tuning_path = tmp_path / "ingest_tuning.json"
    result = runner.invoke(
        app,
        [
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "candles",
            "--watchlists-path",
            str(watchlists_path),
            "--watchlist",
            "positions",
            "--candles-concurrency",
            "2",
            "--candles-max-rps",
            "5",
            "--auto-tune",
            "--tune-config",
            str(tuning_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert tuning_path.exists()
