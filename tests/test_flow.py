from __future__ import annotations

import pandas as pd

from options_helper.analysis.flow import FlowClass, classify_flow, compute_flow


def test_classify_building() -> None:
    assert (
        classify_flow(oi_prev=100.0, oi_today=140.0, delta_oi=40.0, volume=50.0)
        == FlowClass.BUILDING
    )


def test_classify_unwinding() -> None:
    assert (
        classify_flow(oi_prev=200.0, oi_today=150.0, delta_oi=-50.0, volume=80.0)
        == FlowClass.UNWINDING
    )


def test_classify_churn() -> None:
    # Small delta, but large volume relative to OI
    assert (
        classify_flow(oi_prev=100.0, oi_today=105.0, delta_oi=5.0, volume=80.0)
        == FlowClass.CHURN
    )


def test_classify_unknown_without_prev() -> None:
    assert classify_flow(oi_prev=None, oi_today=50.0, delta_oi=None, volume=100.0) == FlowClass.UNKNOWN


def test_compute_flow_merges_and_calculates() -> None:
    today = pd.DataFrame(
        [
            {
                "contractSymbol": "ABC240119C00010000",
                "optionType": "call",
                "expiry": "2024-01-19",
                "strike": 10.0,
                "lastPrice": 2.0,
                "volume": 100,
                "openInterest": 150,
            }
        ]
    )
    prev = pd.DataFrame(
        [
            {
                "contractSymbol": "ABC240119C00010000",
                "openInterest": 100,
            }
        ]
    )

    out = compute_flow(today, prev)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["deltaOI"] == 50
    assert row["deltaOI_notional"] == 50 * 2.0 * 100.0
    assert row["volume_notional"] == 100 * 2.0 * 100.0


def test_compute_flow_prefers_osi_join_when_available() -> None:
    today = pd.DataFrame(
        [
            {
                "contractSymbol": "ABC_NEW",
                "osi": "ABC   240119C00010000",
                "optionType": "call",
                "expiry": "2024-01-19",
                "strike": 10.0,
                "lastPrice": 2.0,
                "volume": 100,
                "openInterest": 150,
            }
        ]
    )
    prev = pd.DataFrame(
        [
            {
                "contractSymbol": "ABC_OLD",
                "osi": "ABC   240119C00010000",
                "openInterest": 100,
            }
        ]
    )

    out = compute_flow(today, prev)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["deltaOI"] == 50


def test_compute_flow_falls_back_to_contract_symbol_when_prev_missing_osi() -> None:
    today = pd.DataFrame(
        [
            {
                "contractSymbol": "ABC240119C00010000",
                "osi": "ABC   240119C00010000",
                "optionType": "call",
                "expiry": "2024-01-19",
                "strike": 10.0,
                "lastPrice": 2.0,
                "volume": 100,
                "openInterest": 150,
            }
        ]
    )
    prev = pd.DataFrame(
        [
            {
                "contractSymbol": "ABC240119C00010000",
                "openInterest": 100,
            }
        ]
    )

    out = compute_flow(today, prev)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["openInterest_prev"] == 100
    assert row["deltaOI"] == 50
