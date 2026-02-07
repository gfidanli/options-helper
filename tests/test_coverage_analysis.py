from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.coverage import (
    compute_candle_coverage,
    compute_contract_oi_coverage,
    compute_option_bars_coverage,
)
from options_helper.analysis.coverage_repair import build_repair_suggestions


def test_compute_candle_coverage_reports_missing_business_days_and_null_cells() -> None:
    candles = pd.DataFrame(
        [
            {"ts": "2026-01-06", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 10},
            {"ts": "2026-01-07", "open": 1.0, "high": 1.1, "low": 0.9, "close": None, "volume": 11},
            {"ts": "2026-01-09", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": None},
        ]
    )

    coverage = compute_candle_coverage(candles, lookback_days=5)

    assert coverage.rows_total == 3
    assert coverage.rows_lookback == 3
    assert coverage.start_date == "2026-01-06"
    assert coverage.end_date == "2026-01-09"
    assert coverage.expected_business_days == 4
    assert coverage.missing_business_days == 1
    assert coverage.missing_business_dates == ["2026-01-08"]
    assert coverage.missing_value_cells == 2
    assert coverage.missing_values_by_column == {"close": 1, "volume": 1}


def test_compute_contract_oi_coverage_and_oi_delta() -> None:
    contracts = pd.DataFrame(
        [
            {"contract_symbol": "AAA260220C00100000"},
            {"contract_symbol": "AAA260220P00100000"},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "contract_symbol": "AAA260220C00100000",
                "as_of_date": "2026-01-06",
                "open_interest": 10,
            },
            {
                "contract_symbol": "AAA260220C00100000",
                "as_of_date": "2026-01-07",
                "open_interest": 11,
            },
            {
                "contract_symbol": "AAA260220C00100000",
                "as_of_date": "2026-01-08",
                "open_interest": 12,
            },
            {
                "contract_symbol": "AAA260220P00100000",
                "as_of_date": "2026-01-06",
                "open_interest": 5,
            },
            {
                "contract_symbol": "AAA260220P00100000",
                "as_of_date": "2026-01-08",
                "open_interest": 7,
            },
        ]
    )

    coverage = compute_contract_oi_coverage(
        contracts,
        snapshots,
        lookback_days=3,
        as_of=date(2026, 1, 8),
    )

    assert coverage.contracts_total == 2
    assert coverage.contracts_with_snapshots == 2
    assert coverage.contracts_with_oi == 2
    assert coverage.expected_contract_days == 6
    assert coverage.observed_contract_days == 5
    assert coverage.observed_oi_contract_days == 5
    assert coverage.snapshot_day_coverage_ratio == pytest.approx(5.0 / 6.0)
    assert coverage.oi_day_coverage_ratio == pytest.approx(5.0 / 6.0)
    assert coverage.snapshot_days_missing == 0

    delta_1d = next(item for item in coverage.oi_delta_coverage if item.lag_days == 1)
    assert delta_1d.contracts_with_oi == 2
    assert delta_1d.contracts_with_delta == 1
    assert delta_1d.coverage_ratio == pytest.approx(0.5)


def test_compute_option_bars_coverage_counts_statuses() -> None:
    meta = pd.DataFrame(
        [
            {
                "contract_symbol": "AAA260220C00100000",
                "status": "ok",
                "rows": 10,
                "start_ts": "2026-01-01",
                "end_ts": "2026-01-08",
            },
            {
                "contract_symbol": "AAA260220P00100000",
                "status": "error",
                "rows": 0,
                "start_ts": None,
                "end_ts": None,
            },
        ]
    )

    coverage = compute_option_bars_coverage(meta, lookback_days=3, as_of=date(2026, 1, 8))

    assert coverage.contracts_total == 2
    assert coverage.contracts_with_rows == 1
    assert coverage.rows_total == 10
    assert coverage.status_counts == {"ok": 1, "error": 1}
    assert coverage.contracts_covering_lookback_end == 1
    assert coverage.covering_lookback_end_ratio == pytest.approx(0.5)


def test_build_repair_suggestions_prioritizes_gap_commands() -> None:
    rows = build_repair_suggestions(
        symbol="spy",
        days=60,
        candles={"rows_total": 0, "missing_business_days": 5, "missing_value_cells": 0},
        snapshots={"days_present_lookback": 0, "expected_business_days": 10},
        contracts_oi={
            "contracts_total": 2,
            "contracts_with_snapshots": 1,
            "snapshot_days_missing": 2,
            "contracts_with_oi": 1,
        },
        option_bars={"contracts_total": 2, "contracts_covering_lookback_end": 0, "contracts_with_rows": 0},
    )

    assert rows
    commands = [str(row.get("command") or "") for row in rows]
    assert any("ingest candles --symbol SPY" in cmd for cmd in commands)
    assert any("--contracts-only" in cmd for cmd in commands)
    assert any("ingest options-bars --symbol SPY" in cmd for cmd in commands)
