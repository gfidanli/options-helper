from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_duckdb(path: Path) -> None:
    wh = DuckDBWarehouse(path)
    ensure_schema(wh)

    wh.execute(
        """
        INSERT INTO candles_daily(symbol, interval, auto_adjust, back_adjust, ts, open, high, low, close, volume)
        VALUES
          ('AAA', '1d', TRUE, FALSE, '2026-01-06', 100, 101, 99, 100, 1000),
          ('AAA', '1d', TRUE, FALSE, '2026-01-07', 101, 102, 100, 101, 1100),
          ('AAA', '1d', TRUE, FALSE, '2026-01-09', 102, 103, 101, 102, 1200)
        """
    )

    wh.execute(
        """
        INSERT INTO options_snapshot_headers(symbol, snapshot_date, provider, chain_path, contracts)
        VALUES ('AAA', '2026-01-09', 'alpaca', '/tmp/chain.parquet', 2)
        """
    )

    wh.execute(
        """
        INSERT INTO option_contracts(contract_symbol, underlying, expiry, option_type, strike, multiplier, provider)
        VALUES
          ('AAA260220C00100000', 'AAA', '2026-02-20', 'call', 100, 100, 'alpaca'),
          ('AAA260220P00100000', 'AAA', '2026-02-20', 'put', 100, 100, 'alpaca')
        """
    )

    wh.execute(
        """
        INSERT INTO option_contract_snapshots(
          contract_symbol,
          as_of_date,
          open_interest,
          open_interest_date,
          close_price,
          close_price_date,
          provider
        ) VALUES
          ('AAA260220C00100000', '2026-01-08', 10, '2026-01-08', 1.2, '2026-01-08', 'alpaca'),
          ('AAA260220P00100000', '2026-01-08', 15, '2026-01-08', 1.3, '2026-01-08', 'alpaca')
        """
    )

    wh.execute(
        """
        INSERT INTO option_bars_meta(
          contract_symbol,
          interval,
          provider,
          status,
          rows,
          start_ts,
          end_ts,
          last_success_at,
          last_attempt_at,
          last_error,
          error_count
        ) VALUES
          ('AAA260220C00100000', '1d', 'alpaca', 'ok', 5, '2026-01-03', '2026-01-09', NOW(), NOW(), NULL, 0)
        """
    )


def test_coverage_cli_json_and_out_file(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "options.duckdb"
    _seed_duckdb(duckdb_path)

    runner = CliRunner()
    out_path = tmp_path / "coverage.json"
    result = runner.invoke(
        app,
        [
            "--duckdb-path",
            str(duckdb_path),
            "coverage",
            "AAA",
            "--days",
            "5",
            "--json",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["symbol"] == "AAA"
    assert payload["days"] == 5
    assert "candles" in payload
    assert "contracts_oi" in payload
    assert "option_bars" in payload
    assert isinstance(payload["repair_suggestions"], list)

    assert out_path.exists()
    saved_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved_payload["symbol"] == "AAA"


def test_coverage_cli_human_output(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "options.duckdb"
    _seed_duckdb(duckdb_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--duckdb-path",
            str(duckdb_path),
            "coverage",
            "AAA",
            "--days",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Coverage for AAA" in result.output
    assert "Repair Suggestions" in result.output
