from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from options_helper.analysis.strategy_modeling_trade_review import rank_trades_for_review


def test_rank_trades_for_review_fallback_scope_filters_and_orders_deterministically() -> None:
    trade_rows = [
        {
            "trade_id": "trade-1",
            "status": " closed ",
            "realized_r": "1.5",
            "entry_ts": "2026-01-02T14:30:00+00:00",
            "reject_code": "",
        },
        {
            "trade_id": "trade-2",
            "status": "CLOSED",
            "realized_r": 1.5,
            "entry_ts": datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc),
            "reject_code": None,
        },
        {
            "trade_id": "trade-3",
            "status": "closed",
            "realized_r": 1.5,
            "entry_ts": pd.Timestamp("2026-01-01T09:30:00-05:00"),
            "reject_code": None,
        },
        {
            "trade_id": "trade-4",
            "status": "closed",
            "realized_r": -0.5,
            "entry_ts": None,
            "reject_code": None,
        },
        {
            "trade_id": "trade-11",
            "status": "closed",
            "realized_r": "0.3",
            "entry_ts": "bad-ts",
            "reject_code": None,
        },
        {"trade_id": "trade-5", "status": "open", "realized_r": 5.0, "reject_code": None},
        {"trade_id": "trade-6", "status": "closed", "realized_r": float("nan"), "reject_code": None},
        {"trade_id": "trade-7", "status": "closed", "realized_r": "inf", "reject_code": None},
        {"trade_id": "trade-8", "status": "closed", "realized_r": 0.2, "reject_code": "rsi_not_extreme"},
        {"trade_id": "trade-9", "status": None, "realized_r": 0.1, "reject_code": None},
        {"trade_id": "trade-10", "status": "closed", "realized_r": pd.NA, "reject_code": None},
    ]

    result = rank_trades_for_review(
        trade_rows,
        accepted_trade_ids=None,
        top_n=10,
    )

    assert result.metric == "realized_r"
    assert result.scope == "closed_nonrejected_trades"
    assert result.candidate_trade_count == 5

    assert [row["trade_id"] for row in result.top_best_rows] == [
        "trade-3",
        "trade-1",
        "trade-2",
        "trade-11",
        "trade-4",
    ]
    assert [row["trade_id"] for row in result.top_worst_rows] == [
        "trade-4",
        "trade-11",
        "trade-3",
        "trade-1",
        "trade-2",
    ]
    assert [row["rank"] for row in result.top_best_rows] == [1, 2, 3, 4, 5]
    assert [row["rank"] for row in result.top_worst_rows] == [1, 2, 3, 4, 5]
    assert isinstance(result.top_best_rows[1]["realized_r"], float)


def test_rank_trades_for_review_accepted_scope_is_authoritative_when_empty() -> None:
    trade_rows = [
        {"trade_id": "trade-a", "status": "closed", "realized_r": 1.25, "reject_code": None},
        {"trade_id": "trade-b", "status": "closed", "realized_r": -0.5, "reject_code": None},
    ]

    result = rank_trades_for_review(
        trade_rows,
        accepted_trade_ids=(),
        top_n=20,
    )

    assert result.scope == "accepted_closed_trades"
    assert result.candidate_trade_count == 0
    assert result.top_best_rows == ()
    assert result.top_worst_rows == ()


def test_rank_trades_for_review_accepted_scope_filters_to_ids_and_candidate_rules() -> None:
    trade_rows = [
        {
            "trade_id": "accepted-good",
            "status": "closed",
            "realized_r": "2.0",
            "entry_ts": "2026-01-03T14:30:00Z",
            "reject_code": None,
        },
        {
            "trade_id": "accepted-rejected",
            "status": "closed",
            "realized_r": 1.0,
            "entry_ts": "2026-01-03T14:31:00Z",
            "reject_code": "symbol_not_allowed",
        },
        {
            "trade_id": "accepted-open",
            "status": "open",
            "realized_r": 5.0,
            "entry_ts": "2026-01-03T14:32:00Z",
            "reject_code": None,
        },
        {
            "trade_id": "not-accepted",
            "status": "closed",
            "realized_r": -3.0,
            "entry_ts": "2026-01-03T14:29:00Z",
            "reject_code": None,
        },
    ]

    result = rank_trades_for_review(
        trade_rows,
        accepted_trade_ids=("accepted-good", "accepted-rejected", "accepted-open"),
        top_n=20,
    )

    assert result.scope == "accepted_closed_trades"
    assert result.candidate_trade_count == 1
    assert [row["trade_id"] for row in result.top_best_rows] == ["accepted-good"]
    assert [row["trade_id"] for row in result.top_worst_rows] == ["accepted-good"]
    assert result.top_best_rows[0]["rank"] == 1
    assert result.top_worst_rows[0]["rank"] == 1


def test_rank_trades_for_review_parses_object_rows_and_timestamp_tiebreaks() -> None:
    trade_rows = [
        SimpleNamespace(
            trade_id="trade-b",
            status="closed",
            realized_r=1.0,
            entry_ts=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            reject_code=None,
        ),
        SimpleNamespace(
            trade_id="trade-a",
            status="closed",
            realized_r=1,
            entry_ts=pd.Timestamp("2026-01-02T10:00:00-05:00"),
            reject_code=None,
        ),
        SimpleNamespace(
            trade_id="trade-c",
            status="closed",
            realized_r=1.0,
            entry_ts="not-a-timestamp",
            reject_code=None,
        ),
    ]

    result = rank_trades_for_review(
        trade_rows,
        accepted_trade_ids=None,
        top_n=3,
    )

    assert [row["trade_id"] for row in result.top_best_rows] == ["trade-a", "trade-b", "trade-c"]
    assert [row["trade_id"] for row in result.top_worst_rows] == ["trade-a", "trade-b", "trade-c"]


def test_rank_trades_for_review_rejects_invalid_metric_and_top_n() -> None:
    rows = [{"trade_id": "trade-1", "status": "closed", "realized_r": 1.0, "reject_code": None}]

    with pytest.raises(ValueError, match="metric"):
        rank_trades_for_review(rows, accepted_trade_ids=None, metric="expectancy_r")

    with pytest.raises(ValueError, match="top_n"):
        rank_trades_for_review(rows, accepted_trade_ids=None, top_n=-1)
