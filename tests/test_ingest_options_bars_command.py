from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.commands import ingest
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


class _StubProvider:
    name = "alpaca"


class _FakeAlpacaClient:
    def __init__(self) -> None:
        self.contract_calls: list[dict[str, object]] = []
        self.bars_calls: list[dict[str, object]] = []

    def list_option_contracts(
        self,
        underlying: str,
        *,
        exp_gte: date | None = None,
        exp_lte: date | None = None,
        limit: int | None = None,
        page_limit: int | None = None,  # noqa: ARG002
    ) -> list[dict[str, object]]:
        self.contract_calls.append(
            {"underlying": underlying, "exp_gte": exp_gte, "exp_lte": exp_lte, "limit": limit}
        )
        contracts = [
            {
                "contractSymbol": "AAA_BAD260117C00100000",
                "underlying": underlying,
                "expiration_date": "2026-01-17",
                "option_type": "call",
                "strike_price": 100.0,
                "multiplier": 100,
                "open_interest": 10,
                "open_interest_date": "2026-01-15",
                "close_price": 1.1,
                "close_price_date": "2026-01-15",
            },
            {
                "contractSymbol": "BBB_GOOD260117C00100000",
                "underlying": underlying,
                "expiration_date": "2026-01-17",
                "option_type": "call",
                "strike_price": 105.0,
                "multiplier": 100,
                "open_interest": 12,
                "open_interest_date": "2026-01-15",
                "close_price": 1.2,
                "close_price_date": "2026-01-15",
            },
            {
                "contractSymbol": "CCC_GOOD260117P00100000",
                "underlying": underlying,
                "expiration_date": "2026-01-17",
                "option_type": "put",
                "strike_price": 100.0,
                "multiplier": 100,
                "open_interest": 9,
                "open_interest_date": "2026-01-15",
                "close_price": 1.0,
                "close_price_date": "2026-01-15",
            },
        ]

        def _within(raw: dict[str, object]) -> bool:
            expiry = date.fromisoformat(str(raw["expiration_date"]))
            if exp_gte and expiry < exp_gte:
                return False
            if exp_lte and expiry > exp_lte:
                return False
            return True

        return [c for c in contracts if _within(c)]

    def get_option_bars_daily_full(
        self,
        symbols: list[str],
        *,
        start,  # noqa: ANN001
        end,  # noqa: ANN001
        interval: str = "1d",  # noqa: ARG002
        feed: str | None = None,  # noqa: ARG002
        chunk_size: int = 200,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        self.bars_calls.append({"symbols": symbols, "start": start, "end": end})
        if any("AAA_BAD" in sym for sym in symbols):
            raise RuntimeError("boom")
        rows: list[dict[str, object]] = []
        for sym in symbols:
            rows.append(
                {
                    "contractSymbol": sym,
                    "ts": "2026-01-02",
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "volume": 10,
                    "vwap": 1.02,
                    "trade_count": 2,
                }
            )
        return pd.DataFrame(rows)


def test_ingest_options_bars_per_symbol_records_errors(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    fake_client = _FakeAlpacaClient()
    monkeypatch.setattr(ingest, "AlpacaClient", lambda: fake_client)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "SPY",
            "--contracts-exp-start",
            "2025-01-01",
            "--contracts-exp-end",
            "2026-12-31",
            "--lookback-years",
            "1",
            "--bars-batch-mode",
            "per-contract",
        ],
    )

    assert result.exit_code == 0, result.output

    wh = DuckDBWarehouse(duckdb_path)
    ensure_schema(wh)

    contracts = wh.fetch_df(
        "SELECT contract_symbol FROM option_contracts ORDER BY contract_symbol ASC"
    )
    assert list(contracts["contract_symbol"]) == [
        "AAA_BAD260117C00100000",
        "BBB_GOOD260117C00100000",
        "CCC_GOOD260117P00100000",
    ]

    bars = wh.fetch_df(
        "SELECT DISTINCT contract_symbol FROM option_bars ORDER BY contract_symbol ASC"
    )
    assert list(bars["contract_symbol"]) == [
        "BBB_GOOD260117C00100000",
        "CCC_GOOD260117P00100000",
    ]

    meta = wh.fetch_df(
        """
        SELECT contract_symbol, status
        FROM option_bars_meta
        ORDER BY contract_symbol ASC
        """
    )
    statuses = {row["contract_symbol"]: str(row["status"]) for row in meta.to_dict("records")}
    assert statuses["AAA_BAD260117C00100000"] == "error"
    assert statuses["BBB_GOOD260117C00100000"] == "ok"
    assert statuses["CCC_GOOD260117P00100000"] == "ok"

    assert len(fake_client.bars_calls) == 3
    assert all(len(call["symbols"]) == 1 for call in fake_client.bars_calls)


def test_ingest_options_bars_passes_http_pool_flags(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    fake_client = _FakeAlpacaClient()
    seen_kwargs: list[dict[str, object]] = []

    def _client_factory(**kwargs):  # type: ignore[no-untyped-def]
        seen_kwargs.append(dict(kwargs))
        return fake_client

    monkeypatch.setattr(ingest, "AlpacaClient", _client_factory)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "SPY",
            "--contracts-exp-start",
            "2025-01-01",
            "--contracts-exp-end",
            "2026-12-31",
            "--lookback-years",
            "1",
            "--alpaca-http-pool-maxsize",
            "128",
            "--alpaca-http-pool-connections",
            "64",
            "--bars-batch-mode",
            "per-contract",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen_kwargs, "expected AlpacaClient to receive keyword args"
    assert seen_kwargs[0]["http_pool_maxsize"] == 128
    assert seen_kwargs[0]["http_pool_connections"] == 64


def test_ingest_options_bars_fetch_only_skips_warehouse_writes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    fake_client = _FakeAlpacaClient()
    monkeypatch.setattr(ingest, "AlpacaClient", lambda: fake_client)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "SPY",
            "--contracts-exp-start",
            "2025-01-01",
            "--contracts-exp-end",
            "2026-12-31",
            "--lookback-years",
            "1",
            "--fetch-only",
            "--bars-batch-mode",
            "per-contract",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Fetch-only summary:" in result.output

    wh = DuckDBWarehouse(duckdb_path)
    ensure_schema(wh)

    contracts = wh.fetch_df("SELECT COUNT(*) AS n FROM option_contracts")
    bars = wh.fetch_df("SELECT COUNT(*) AS n FROM option_bars")
    meta = wh.fetch_df("SELECT COUNT(*) AS n FROM option_bars_meta")

    assert int(contracts.iloc[0]["n"]) == 0
    assert int(bars.iloc[0]["n"]) == 0
    assert int(meta.iloc[0]["n"]) == 0
    assert len(fake_client.bars_calls) == 3


def test_ingest_options_bars_contracts_only_writes_contract_snapshots(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    fake_client = _FakeAlpacaClient()
    monkeypatch.setattr(ingest, "AlpacaClient", lambda: fake_client)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "SPY",
            "--contracts-exp-start",
            "2025-01-01",
            "--contracts-exp-end",
            "2026-12-31",
            "--contracts-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Contracts-only mode" in result.output

    wh = DuckDBWarehouse(duckdb_path)
    ensure_schema(wh)

    contracts = wh.fetch_df("SELECT COUNT(*) AS n FROM option_contracts")
    snapshots = wh.fetch_df("SELECT COUNT(*) AS n FROM option_contract_snapshots")
    bars = wh.fetch_df("SELECT COUNT(*) AS n FROM option_bars")
    bars_meta = wh.fetch_df("SELECT COUNT(*) AS n FROM option_bars_meta")

    assert int(contracts.iloc[0]["n"]) == 3
    assert int(snapshots.iloc[0]["n"]) == 3
    assert int(bars.iloc[0]["n"]) == 0
    assert int(bars_meta.iloc[0]["n"]) == 0
    assert len(fake_client.bars_calls) == 0


def test_ingest_options_bars_fetch_only_conflicts_with_dry_run(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    monkeypatch.setattr(ingest, "AlpacaClient", lambda: _FakeAlpacaClient())

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "SPY",
            "--dry-run",
            "--fetch-only",
        ],
    )

    assert result.exit_code != 0
    assert "Choose either --dry-run or --fetch-only, not both." in result.output


def test_ingest_options_bars_auto_tune_writes_profile_and_contract_limit(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    fake_client = _FakeAlpacaClient()
    monkeypatch.setattr(ingest, "AlpacaClient", lambda: fake_client)

    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    tuning_path = tmp_path / "ingest_tuning.json"
    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "SPY",
            "--contracts-exp-start",
            "2025-01-01",
            "--contracts-exp-end",
            "2026-12-31",
            "--lookback-years",
            "1",
            "--contracts-page-size",
            "9999",
            "--bars-batch-mode",
            "per-contract",
            "--auto-tune",
            "--tune-config",
            str(tuning_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert tuning_path.exists()
    assert any(call.get("limit") == 9999 for call in fake_client.contract_calls)
