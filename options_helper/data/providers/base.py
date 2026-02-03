from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import pandas as pd

from options_helper.data.yf_client import EarningsEvent, OptionsChain, UnderlyingData

OPTION_CHAIN_REQUIRED_COLUMNS = (
    "contractSymbol",
    "expiry",
    "strike",
    "optionType",
    "bid",
    "ask",
    "lastPrice",
    "openInterest",
    "volume",
    "impliedVolatility",
)


def normalize_option_chain(
    df: pd.DataFrame,
    *,
    option_type: str | None = None,
    expiry: date | None = None,
) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    for col in OPTION_CHAIN_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if option_type is not None:
        if "optionType" not in out.columns:
            out["optionType"] = option_type
        else:
            out["optionType"] = out["optionType"].fillna(option_type)
    if expiry is not None:
        expiry_val = expiry.isoformat()
        if "expiry" not in out.columns:
            out["expiry"] = expiry_val
        else:
            out["expiry"] = out["expiry"].fillna(expiry_val)
    return out


class MarketDataProvider(Protocol):
    name: str

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData: ...

    def get_quote(self, symbol: str) -> float | None: ...

    def list_option_expiries(self, symbol: str) -> list[date]: ...

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain: ...

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict[str, Any]: ...

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent: ...
