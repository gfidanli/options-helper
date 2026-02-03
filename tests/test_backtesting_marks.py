from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from options_helper.backtesting.marks import build_mark_series
from options_helper.data.options_snapshots import OptionsSnapshotStore


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "backtest"


def test_build_mark_series_contract_symbol() -> None:
    store = OptionsSnapshotStore(_fixture_root())
    result = build_mark_series(store, symbol="AAA", contract_symbol="AAA260220C00100000")
    series = result.series

    assert list(series.index.date) == [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)]

    row = series.loc[pd.Timestamp("2026-01-02")]
    assert row["mark"] == pytest.approx(1.1)
    assert row["spread"] == pytest.approx(0.2)
    assert bool(row["has_contract"]) is True
    assert bool(row["has_quote"]) is True

    row = series.loc[pd.Timestamp("2026-01-03")]
    assert row["mark"] == pytest.approx(1.5)
    assert pd.isna(row["spread"])
    assert bool(row["has_contract"]) is True
    assert bool(row["has_quote"]) is True

    row = series.loc[pd.Timestamp("2026-01-04")]
    assert bool(row["has_contract"]) is False
    assert bool(row["has_quote"]) is False
    assert pd.isna(row["mark"])


def test_build_mark_series_date_filter() -> None:
    store = OptionsSnapshotStore(_fixture_root())
    result = build_mark_series(
        store,
        symbol="AAA",
        contract_symbol="AAA260220C00100000",
        start=date(2026, 1, 3),
        end=date(2026, 1, 3),
    )
    series = result.series

    assert list(series.index.date) == [date(2026, 1, 3)]
