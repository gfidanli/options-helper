from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.db.warehouse import DuckDBWarehouse


class _NegativeCandleProvider:
    name = "stub"

    def get_history(
        self,
        symbol: str,  # noqa: ARG002
        *,
        start,  # noqa: ANN001
        end,  # noqa: ANN001
        interval: str,  # noqa: ARG002
        auto_adjust: bool,  # noqa: ARG002
        back_adjust: bool,  # noqa: ARG002
    ) -> pd.DataFrame:
        idx = pd.to_datetime([datetime(2026, 1, 8)])
        return pd.DataFrame(
            {
                "Open": [-10.0],
                "High": [-9.5],
                "Low": [-10.5],
                "Close": [-9.8],
                "Volume": [100],
            },
            index=idx,
        )


class _PositiveCandleProvider:
    name = "stub"

    def get_history(
        self,
        symbol: str,  # noqa: ARG002
        *,
        start,  # noqa: ANN001
        end,  # noqa: ANN001
        interval: str,  # noqa: ARG002
        auto_adjust: bool,  # noqa: ARG002
        back_adjust: bool,  # noqa: ARG002
    ) -> pd.DataFrame:
        idx = pd.to_datetime([datetime(2026, 1, 8), datetime(2026, 1, 9)])
        return pd.DataFrame(
            {
                "Open": [10.0, 10.5],
                "High": [10.5, 11.0],
                "Low": [9.5, 10.0],
                "Close": [10.1, 10.8],
                "Volume": [100, 110],
            },
            index=idx,
        )


def _write_watchlists(path: Path) -> None:
    path.write_text('{"watchlists": {"positions": ["AAA"]}}', encoding="utf-8")


def _latest_run_for_job(warehouse: DuckDBWarehouse, job_name: str) -> tuple[str, str, str | None]:
    row = warehouse.fetch_df(
        """
        SELECT run_id, status, error_message
        FROM meta.ingestion_runs
        WHERE job_name = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        [job_name],
    ).iloc[0]
    return str(row["run_id"]), str(row["status"]), None if pd.isna(row["error_message"]) else str(row["error_message"])


def test_check_failures_are_persisted_and_non_blocking(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    duckdb_path = tmp_path / "options.duckdb"
    watchlists_path = tmp_path / "watchlists.json"
    _write_watchlists(watchlists_path)

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _NegativeCandleProvider())

    runner = CliRunner()
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
            "--candle-cache-dir",
            str(tmp_path / "candles"),
        ],
    )
    assert result.exit_code == 0, result.output

    warehouse = DuckDBWarehouse(duckdb_path)
    run_id, run_status, error_message = _latest_run_for_job(warehouse, "ingest_candles")
    assert run_status == "success"
    assert error_message is None

    checks = warehouse.fetch_df(
        """
        SELECT check_name, severity, status, partition_key
        FROM meta.asset_checks
        WHERE run_id = ?
        ORDER BY check_name ASC
        """,
        [run_id],
    )
    assert set(checks["check_name"]) == {
        "candles_gap_days_last_30",
        "candles_monotonic_date",
        "candles_no_negative_prices",
        "candles_unique_symbol_date",
    }

    statuses = {row["check_name"]: row["status"] for row in checks.to_dict(orient="records")}
    assert statuses["candles_no_negative_prices"] == "fail"
    assert statuses["candles_gap_days_last_30"] == "fail"



def test_quality_check_runtime_exception_fails_run(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    duckdb_path = tmp_path / "options.duckdb"
    watchlists_path = tmp_path / "watchlists.json"
    _write_watchlists(watchlists_path)

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _PositiveCandleProvider())

    def _boom(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("quality check runtime boom")

    monkeypatch.setattr("options_helper.pipelines.visibility_jobs.run_candle_quality_checks", _boom)

    runner = CliRunner()
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
            "--candle-cache-dir",
            str(tmp_path / "candles"),
        ],
    )
    assert result.exit_code != 0

    warehouse = DuckDBWarehouse(duckdb_path)
    _, run_status, error_message = _latest_run_for_job(warehouse, "ingest_candles")
    assert run_status == "failed"
    assert error_message is not None
    assert "quality check runtime boom" in error_message
