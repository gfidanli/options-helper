from __future__ import annotations

import pandas as pd
import pytest

from options_helper.analysis.intraday_flow import (
    aggregate_intraday_flow_by_contract_terms,
    classify_intraday_trades,
    summarize_intraday_contract_flow,
    summarize_intraday_time_buckets,
)
from options_helper.schemas.research_metrics_contracts import (
    INTRADAY_FLOW_CONTRACT_FIELDS,
    INTRADAY_FLOW_TIME_BUCKET_FIELDS,
)


def _sample_trades_and_quotes() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.DataFrame(
        [
            # Out-of-order on purpose: first row has no earlier quote.
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:29:50Z",
                "price": 1.15,
                "size": 3,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:30:10Z",
                "price": 1.20,
                "size": 10,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:30:40Z",
                "price": 0.90,
                "size": 5,
            },
            # Exact duplicate should be de-duped pre-merge.
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:30:40Z",
                "price": 0.90,
                "size": 5,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:31:10Z",
                "price": 1.10,
                "size": 2,
            },
            # Missing/invalid quote (bid=ask=0) should classify unknown.
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:32:05Z",
                "price": 1.10,
                "size": 1,
            },
            # Dropped trades:
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:32:00Z",
                "price": None,
                "size": 1,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:32:30Z",
                "price": 1.25,
                "size": None,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "optionType": "call",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": 0.52,
                "timestamp": "2026-02-05T14:33:00Z",
                "price": 1.30,
                "size": 0,
            },
            # Separate contract
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320P00450000",
                "optionType": "put",
                "expiry": "2026-03-20",
                "strike": 450.0,
                "bs_delta": -0.48,
                "timestamp": "2026-02-05T14:30:20Z",
                "price": 2.11,
                "size": 4,
            },
        ]
    )

    quotes = pd.DataFrame(
        [
            # Out-of-order + duplicate timestamp with bad bid/ask.
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "timestamp": "2026-02-05T14:30:00Z",
                "bid": 1.00,
                "ask": 1.10,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "timestamp": "2026-02-05T14:31:00Z",
                "bid": 1.05,
                "ask": 1.15,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320P00450000",
                "timestamp": "2026-02-05T14:30:00Z",
                "bid": 2.00,
                "ask": 2.10,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "timestamp": "2026-02-05T14:30:30Z",
                "bid": 0.95,
                "ask": 1.05,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "timestamp": "2026-02-05T14:30:00Z",
                "bid": 0.0,
                "ask": 0.0,
            },
            {
                "symbol": "SPY",
                "contractSymbol": "SPY260320C00450000",
                "timestamp": "2026-02-05T14:32:00Z",
                "bid": 0.0,
                "ask": 0.0,
            },
        ]
    )
    return trades, quotes


def test_classify_intraday_trades_handles_sorting_dedup_and_edge_cases() -> None:
    trades, quotes = _sample_trades_and_quotes()
    classified = classify_intraday_trades(trades, quotes)

    assert len(classified) == 9
    assert classified["timestamp"].is_monotonic_increasing

    call = classified[classified["contract_symbol"] == "SPY260320C00450000"]
    assert len(call) == 8

    early = call.loc[call["timestamp"] == pd.Timestamp("2026-02-05T14:29:50Z")].iloc[0]
    assert early["direction"] == "unknown"
    assert early["direction_reason"] == "missing_quote"
    assert "unknown_missing_quote" in early["warning_codes"]

    buy = call.loc[call["timestamp"] == pd.Timestamp("2026-02-05T14:30:10Z")].iloc[0]
    assert buy["direction"] == "buy"
    assert bool(buy["has_valid_quote"])
    assert buy["delta_bucket"] == "d40_60"

    sell = call.loc[call["timestamp"] == pd.Timestamp("2026-02-05T14:30:40Z")].iloc[0]
    assert sell["direction"] == "sell"

    invalid_quote = call.loc[call["timestamp"] == pd.Timestamp("2026-02-05T14:32:05Z")].iloc[0]
    assert invalid_quote["direction"] == "unknown"
    assert invalid_quote["direction_reason"] == "invalid_quote"
    assert "unknown_invalid_quote" in invalid_quote["warning_codes"]

    dropped_missing_price = call.loc[call["timestamp"] == pd.Timestamp("2026-02-05T14:32:00Z")].iloc[0]
    assert dropped_missing_price["direction"] == "dropped"
    assert "dropped_invalid_price" in dropped_missing_price["warning_codes"]

    dropped_missing_size = call.loc[call["timestamp"] == pd.Timestamp("2026-02-05T14:32:30Z")].iloc[0]
    assert dropped_missing_size["direction"] == "dropped"
    assert "missing_size_coerced_zero" in dropped_missing_size["warning_codes"]
    assert "dropped_zero_size" in dropped_missing_size["warning_codes"]


def test_summarize_intraday_contract_flow_outputs_contract_day_metrics() -> None:
    trades, quotes = _sample_trades_and_quotes()
    classified = classify_intraday_trades(trades, quotes)
    summary = summarize_intraday_contract_flow(classified, source="stream")

    assert list(summary.columns) == list(INTRADAY_FLOW_CONTRACT_FIELDS)
    assert len(summary) == 2

    call_row = summary.loc[summary["contract_symbol"] == "SPY260320C00450000"].iloc[0]
    assert call_row["buy_volume"] == pytest.approx(10.0)
    assert call_row["sell_volume"] == pytest.approx(5.0)
    assert call_row["unknown_volume"] == pytest.approx(6.0)
    assert call_row["buy_notional"] == pytest.approx(1200.0)
    assert call_row["sell_notional"] == pytest.approx(450.0)
    assert call_row["net_notional"] == pytest.approx(750.0)
    assert call_row["trade_count"] == 5
    assert call_row["unknown_trade_share"] == pytest.approx(0.6)
    assert call_row["quote_coverage_pct"] == pytest.approx(0.6)
    assert call_row["delta_bucket"] == "d40_60"
    assert "dropped_invalid_price" in call_row["warnings"]
    assert "dropped_zero_size:2" in call_row["warnings"]
    assert "missing_size_coerced_zero" in call_row["warnings"]

    put_row = summary.loc[summary["contract_symbol"] == "SPY260320P00450000"].iloc[0]
    assert put_row["buy_volume"] == pytest.approx(4.0)
    assert put_row["sell_volume"] == pytest.approx(0.0)
    assert put_row["unknown_volume"] == pytest.approx(0.0)
    assert put_row["unknown_trade_share"] == pytest.approx(0.0)
    assert put_row["quote_coverage_pct"] == pytest.approx(1.0)
    assert put_row["delta_bucket"] == "d40_60"


def test_aggregate_intraday_flow_by_contract_terms_rolls_up_weighted_rates() -> None:
    trades, quotes = _sample_trades_and_quotes()
    classified = classify_intraday_trades(trades, quotes)
    summary = summarize_intraday_contract_flow(classified, source="stream")

    call_row = summary.loc[summary["contract_symbol"] == "SPY260320C00450000"].iloc[0].copy()
    call_row["contract_symbol"] = "SPY260320C00455000"
    call_row["source"] = "rest"
    call_row["buy_volume"] = 2.0
    call_row["sell_volume"] = 1.0
    call_row["unknown_volume"] = 1.0
    call_row["buy_notional"] = 200.0
    call_row["sell_notional"] = 100.0
    call_row["net_notional"] = 100.0
    call_row["trade_count"] = 3
    call_row["unknown_trade_share"] = 1.0 / 3.0
    call_row["quote_coverage_pct"] = 1.0
    call_row["warnings"] = ["synthetic_warning"]

    expanded = pd.concat([summary, pd.DataFrame([call_row])], ignore_index=True)
    grouped = aggregate_intraday_flow_by_contract_terms(expanded)

    assert len(grouped) == 2
    call_group = grouped.loc[grouped["option_type"] == "call"].iloc[0]
    assert call_group["source"] == "mixed"
    assert call_group["trade_count"] == 8
    assert call_group["buy_notional"] == pytest.approx(1400.0)
    assert call_group["sell_notional"] == pytest.approx(550.0)
    assert call_group["net_notional"] == pytest.approx(850.0)
    assert call_group["unknown_trade_share"] == pytest.approx(0.5)
    assert call_group["quote_coverage_pct"] == pytest.approx(0.75)
    assert "synthetic_warning" in call_group["warnings"]


def test_summarize_intraday_time_buckets_supports_5m_and_15m() -> None:
    trades, quotes = _sample_trades_and_quotes()
    classified = classify_intraday_trades(trades, quotes)

    buckets_5m = summarize_intraday_time_buckets(classified, bucket_minutes=5)
    assert list(buckets_5m.columns) == list(INTRADAY_FLOW_TIME_BUCKET_FIELDS)
    assert set(buckets_5m["bucket_minutes"].tolist()) == {5}

    call_bucket = buckets_5m[
        (buckets_5m["contract_symbol"] == "SPY260320C00450000")
        & (buckets_5m["bucket_start_utc"] == pd.Timestamp("2026-02-05T14:30:00Z"))
    ].iloc[0]
    assert call_bucket["buy_notional"] == pytest.approx(1200.0)
    assert call_bucket["sell_notional"] == pytest.approx(450.0)
    assert call_bucket["net_notional"] == pytest.approx(750.0)
    assert call_bucket["trade_count"] == 4
    assert call_bucket["unknown_trade_share"] == pytest.approx(0.5)

    early_bucket = buckets_5m[
        (buckets_5m["contract_symbol"] == "SPY260320C00450000")
        & (buckets_5m["bucket_start_utc"] == pd.Timestamp("2026-02-05T14:25:00Z"))
    ].iloc[0]
    assert early_bucket["trade_count"] == 1
    assert early_bucket["unknown_trade_share"] == pytest.approx(1.0)

    buckets_15m = summarize_intraday_time_buckets(classified, bucket_minutes=15)
    assert set(buckets_15m["bucket_minutes"].tolist()) == {15}
    assert pd.Timestamp("2026-02-05T14:15:00Z") in set(buckets_15m["bucket_start_utc"])

    with pytest.raises(ValueError, match="bucket_minutes"):
        summarize_intraday_time_buckets(classified, bucket_minutes=10)
