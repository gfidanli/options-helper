from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.storage_runtime import (
    get_default_duckdb_path,
    get_default_storage_backend,
    reset_default_duckdb_path,
    reset_default_storage_backend,
    set_default_duckdb_path,
    set_default_storage_backend,
)


def test_db_info_reports_uninitialized_schema(tmp_path: Path) -> None:
    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"

    res = runner.invoke(app, ["db", "info", "--duckdb-path", str(duckdb_path)])

    assert res.exit_code == 0, res.output
    assert duckdb_path.name in res.output
    assert "schema v0" in res.output
    assert not duckdb_path.exists()


def test_db_init_creates_duckdb_file(tmp_path: Path) -> None:
    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"

    res = runner.invoke(app, ["db", "init", "--duckdb-path", str(duckdb_path)])

    assert res.exit_code == 0, res.output
    assert duckdb_path.exists()
    assert "schema v" in res.output


def test_cli_resets_storage_runtime_contextvars(tmp_path: Path) -> None:
    runner = CliRunner()
    default_backend = get_default_storage_backend()
    default_duckdb_path = get_default_duckdb_path()
    assert default_backend == "duckdb"
    storage_token = set_default_storage_backend("filesystem")
    duckdb_token = set_default_duckdb_path(default_duckdb_path)

    try:
        duckdb_path = tmp_path / "options.duckdb"
        res = runner.invoke(
            app,
            [
                "--storage",
                "duckdb",
                "--duckdb-path",
                str(duckdb_path),
                "db",
                "info",
            ],
        )

        assert res.exit_code == 0, res.output
        assert get_default_storage_backend() == "filesystem"
        assert get_default_duckdb_path() == default_duckdb_path
    finally:
        reset_default_duckdb_path(duckdb_token)
        reset_default_storage_backend(storage_token)

    assert get_default_storage_backend() == default_backend
    assert get_default_duckdb_path() == default_duckdb_path
