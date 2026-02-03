from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


class DataFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class UnderlyingData:
    symbol: str
    last_price: float | None
    history: pd.DataFrame


@dataclass(frozen=True)
class OptionsChain:
    symbol: str
    expiry: date
    calls: pd.DataFrame
    puts: pd.DataFrame


@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    next_date: date | None
    window_start: date | None = None
    window_end: date | None = None
    source: str = "unknown"
    raw: dict[str, Any] | None = None
