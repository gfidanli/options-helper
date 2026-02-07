from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.ingestion.options_bars import BarsBackfillSummary, ContractDiscoveryOutput
from options_helper.pipelines.visibility_jobs import run_ingest_options_bars_job


class _Provider:
    name = "alpaca"


class _ContractsStoreRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.DataFrame, dict[str, object]]] = []

    def upsert_contracts(self, df_contracts: pd.DataFrame, **kwargs: object) -> None:
        self.calls.append((df_contracts.copy(), dict(kwargs)))


def test_run_ingest_options_bars_job_batches_contract_upserts(monkeypatch, tmp_path) -> None:
    recorder = _ContractsStoreRecorder()

    discovery_df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260117C00100000",
                "underlying": "AAA",
                "expiry": "2026-01-17",
                "optionType": "call",
                "strike": 100.0,
                "multiplier": 100,
            },
            {
                "contractSymbol": "AAA260220C00100000",
                "underlying": "AAA",
                "expiry": "2026-02-20",
                "optionType": "call",
                "strike": 100.0,
                "multiplier": 100,
            },
            {
                "contractSymbol": "AAA260320C00100000",
                "underlying": "AAA",
                "expiry": "2026-03-20",
                "optionType": "call",
                "strike": 100.0,
                "multiplier": 100,
            },
        ]
    )
    discovery = ContractDiscoveryOutput(
        contracts=discovery_df,
        raw_by_symbol={
            "AAA260117C00100000": {"contractSymbol": "AAA260117C00100000"},
            "AAA260220C00100000": {"contractSymbol": "AAA260220C00100000"},
            "AAA260320C00100000": {"contractSymbol": "AAA260320C00100000"},
        },
        summaries=[],
    )

    monkeypatch.setattr(
        "options_helper.pipelines.visibility_jobs.discover_option_contracts",
        lambda *args, **kwargs: discovery,
    )
    monkeypatch.setattr(
        "options_helper.pipelines.visibility_jobs.backfill_option_bars",
        lambda *args, **kwargs: BarsBackfillSummary(
            total_contracts=3,
            total_expiries=3,
            planned_contracts=3,
            skipped_contracts=0,
            ok_contracts=3,
            error_contracts=0,
            bars_rows=0,
            requests_attempted=0,
        ),
    )
    monkeypatch.setattr(
        "options_helper.pipelines.visibility_jobs.run_options_bars_quality_checks",
        lambda *args, **kwargs: [],
    )

    result = run_ingest_options_bars_job(
        watchlists_path=tmp_path / "watchlists.json",
        watchlist=["positions"],
        symbol=["AAA"],
        contracts_root_symbols=None,
        contract_symbol_prefix=None,
        contracts_exp_start="2026-01-01",
        contracts_exp_end="2026-12-31",
        lookback_years=5,
        page_limit=200,
        contracts_page_size=2,
        max_underlyings=None,
        max_contracts=None,
        max_expiries=None,
        contracts_max_requests_per_second=None,
        bars_concurrency=1,
        bars_max_requests_per_second=None,
        bars_batch_mode="adaptive",
        bars_batch_size=8,
        bars_write_batch_size=200,
        resume=False,
        dry_run=False,
        fail_fast=False,
        fetch_only=False,
        provider_builder=lambda: _Provider(),
        contracts_store_builder=lambda _path: recorder,
        bars_store_builder=lambda _path: object(),
        client_factory=lambda: object(),  # type: ignore[return-value]
        today=date(2026, 2, 3),
    )

    assert result.no_symbols is False
    assert result.no_contracts is False
    assert len(recorder.calls) == 2
    assert [len(call_df) for call_df, _ in recorder.calls] == [2, 1]
    assert all(call_kwargs.get("provider") == "alpaca" for _, call_kwargs in recorder.calls)
