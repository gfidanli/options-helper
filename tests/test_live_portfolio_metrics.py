from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pytest

from options_helper.analysis.live_portfolio_metrics import (
    compute_live_multileg_rows,
    compute_live_position_rows,
)
from options_helper.models import MultiLegPosition, Portfolio, Position


@dataclass(frozen=True)
class _LiveSnapshot:
    as_of: datetime
    option_quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    option_trades: dict[str, dict[str, Any]] = field(default_factory=dict)


def _dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 2, 10, hour, minute, second, tzinfo=timezone.utc)


def test_compute_live_position_rows_uses_mid_and_single_pnl_math() -> None:
    portfolio = Portfolio(
        positions=[
            Position(
                id="brk-call",
                symbol="BRK-B",
                option_type="call",
                expiry="2026-03-20",
                strike=500.0,
                contracts=1,
                cost_basis=2.0,
            )
        ]
    )
    live = _LiveSnapshot(
        as_of=_dt(15, 0, 0),
        option_quotes={
            "BRK.B260320C00500000": {
                "timestamp": _dt(14, 59, 55),
                "bid_price": 2.0,
                "ask_price": 2.4,
            }
        },
    )

    df = compute_live_position_rows(portfolio, live, stale_after_seconds=30.0)
    assert list(df["id"]) == ["brk-call"]

    row = df.iloc[0].to_dict()
    assert row["contract_symbol"] == "BRK-B260320C00500000"
    assert row["live_symbol"] == "BRK.B260320C00500000"
    assert row["mark"] == pytest.approx(2.2)
    assert row["spread_pct"] == pytest.approx((2.4 - 2.0) / ((2.4 + 2.0) / 2.0))
    assert row["pnl_abs"] == pytest.approx((2.2 - 2.0) * 100.0)
    assert row["pnl_pct"] == pytest.approx(0.1)
    assert row["dte"] == 38
    assert row["warnings"] == []


def test_compute_live_position_rows_flags_stale_and_wide_spreads() -> None:
    portfolio = Portfolio(
        positions=[
            Position(
                id="aapl-put",
                symbol="AAPL",
                option_type="put",
                expiry="2026-04-17",
                strike=180.0,
                contracts=2,
                cost_basis=1.0,
            )
        ]
    )
    live = _LiveSnapshot(
        as_of=_dt(15, 0, 0),
        option_quotes={
            "AAPL260417P00180000": {
                "timestamp": _dt(14, 58, 0),
                "bid_price": 0.5,
                "ask_price": 1.5,
            }
        },
    )

    df = compute_live_position_rows(portfolio, live, stale_after_seconds=30.0)
    row = df.iloc[0].to_dict()
    warnings = set(row["warnings"])

    assert row["mark"] == pytest.approx(1.0)
    assert row["spread_pct"] == pytest.approx(1.0)
    assert "quote_stale" in warnings
    assert "wide_spread" in warnings
    assert "missing_quote" not in warnings


def test_compute_live_multileg_rows_calculates_signed_net_mark_and_pnl() -> None:
    portfolio = Portfolio(
        positions=[
            MultiLegPosition(
                id="msft-credit",
                symbol="MSFT",
                net_debit=-50.0,
                legs=[
                    {
                        "side": "short",
                        "option_type": "put",
                        "expiry": "2026-03-20",
                        "strike": 300.0,
                        "contracts": 1,
                    },
                    {
                        "side": "long",
                        "option_type": "put",
                        "expiry": "2026-03-20",
                        "strike": 290.0,
                        "contracts": 1,
                    },
                ],
            )
        ]
    )
    live = _LiveSnapshot(
        as_of=_dt(15, 0, 0),
        option_quotes={
            "MSFT260320P00300000": {
                "timestamp": _dt(14, 59, 56),
                "bid_price": 3.0,
                "ask_price": 3.4,
            },
            "MSFT260320P00290000": {
                "timestamp": _dt(14, 59, 54),
                "bid_price": 0.9,
                "ask_price": 1.1,
            },
        },
    )

    structure_df, legs_df = compute_live_multileg_rows(portfolio, live, stale_after_seconds=30.0)
    structure = structure_df.iloc[0].to_dict()

    assert structure["net_mark"] == pytest.approx(-220.0)
    assert structure["net_pnl_abs"] == pytest.approx(-170.0)
    assert structure["net_pnl_pct"] == pytest.approx(-3.4)
    assert structure["warnings"] == []

    short_leg = legs_df[legs_df["leg_index"] == 1].iloc[0].to_dict()
    long_leg = legs_df[legs_df["leg_index"] == 2].iloc[0].to_dict()
    assert short_leg["signed_contracts"] == -1
    assert long_leg["signed_contracts"] == 1
    assert short_leg["leg_value"] == pytest.approx(-320.0)
    assert long_leg["leg_value"] == pytest.approx(100.0)


def test_compute_live_multileg_rows_warns_on_missing_leg_quotes() -> None:
    portfolio = Portfolio(
        positions=[
            MultiLegPosition(
                id="msft-missing",
                symbol="MSFT",
                net_debit=120.0,
                legs=[
                    {
                        "side": "short",
                        "option_type": "call",
                        "expiry": "2026-03-20",
                        "strike": 320.0,
                        "contracts": 1,
                    },
                    {
                        "side": "long",
                        "option_type": "call",
                        "expiry": "2026-03-20",
                        "strike": 330.0,
                        "contracts": 1,
                    },
                ],
            )
        ]
    )
    live = _LiveSnapshot(
        as_of=_dt(15, 0, 0),
        option_quotes={
            "MSFT260320C00320000": {
                "timestamp": _dt(14, 59, 58),
                "bid_price": 3.0,
                "ask_price": 3.4,
            }
        },
    )

    structure_df, legs_df = compute_live_multileg_rows(portfolio, live, stale_after_seconds=30.0)
    structure = structure_df.iloc[0].to_dict()

    assert structure["net_mark"] is None
    assert structure["net_pnl_abs"] is None
    assert structure["missing_legs"] == 1
    assert "missing_legs" in set(structure["warnings"])

    missing_leg = legs_df[legs_df["leg_index"] == 2].iloc[0].to_dict()
    missing_warnings = set(missing_leg["warnings"])
    assert pd.isna(missing_leg["mark"])
    assert "missing_quote" in missing_warnings
    assert "missing_mark" in missing_warnings
