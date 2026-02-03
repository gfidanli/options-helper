from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from options_helper.data.candles import CandleStore
from options_helper.data.options_snapshots import OptionsSnapshotStore


def _filter_dates(
    dates: Iterable[date],
    *,
    start: date | None,
    end: date | None,
) -> list[date]:
    out: list[date] = []
    for d in dates:
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        out.append(d)
    return out


@dataclass(frozen=True)
class BacktestDataSource:
    candle_store: CandleStore
    snapshot_store: OptionsSnapshotStore

    def list_snapshot_dates(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        dates = self.snapshot_store.list_dates(symbol)
        return _filter_dates(dates, start=start, end=end)

    def iter_snapshot_days(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> Iterable[tuple[date, pd.DataFrame]]:
        for snapshot_date in self.list_snapshot_dates(symbol, start=start, end=end):
            yield snapshot_date, self.snapshot_store.load_day(symbol, snapshot_date)

    def load_candles(self, symbol: str) -> pd.DataFrame:
        return self.candle_store.load(symbol)

