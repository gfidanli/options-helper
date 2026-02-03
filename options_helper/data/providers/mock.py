from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from options_helper.data.market_types import DataFetchError, EarningsEvent, OptionsChain, UnderlyingData
from options_helper.data.providers.base import MarketDataProvider


class MockProvider(MarketDataProvider):
    """
    Deterministic, in-memory provider for unit tests and demos.

    Notes:
    - This provider does not implement any adjustment logic; `auto_adjust/back_adjust`
      are accepted for interface compatibility but ignored.
    - Candle history slicing is best-effort based on the DataFrame index.
    """

    name = "mock"

    def __init__(
        self,
        *,
        history_by_symbol: dict[str, pd.DataFrame] | None = None,
        quote_by_symbol: dict[str, float] | None = None,
        expiries_by_symbol: dict[str, list[date]] | None = None,
        chains_by_symbol_expiry: dict[tuple[str, date], OptionsChain] | None = None,
        raw_by_symbol_expiry: dict[tuple[str, date], dict[str, Any]] | None = None,
        earnings_by_symbol: dict[str, EarningsEvent] | None = None,
    ) -> None:
        self._history_by_symbol = {k.upper(): v for k, v in (history_by_symbol or {}).items()}
        self._quote_by_symbol = {k.upper(): float(v) for k, v in (quote_by_symbol or {}).items()}
        self._expiries_by_symbol = {k.upper(): list(v) for k, v in (expiries_by_symbol or {}).items()}
        self._chains_by_symbol_expiry = {
            (sym.upper(), exp): chain for (sym, exp), chain in (chains_by_symbol_expiry or {}).items()
        }
        self._raw_by_symbol_expiry = {
            (sym.upper(), exp): raw for (sym, exp), raw in (raw_by_symbol_expiry or {}).items()
        }
        self._earnings_by_symbol = {k.upper(): v for k, v in (earnings_by_symbol or {}).items()}

    def get_history(
        self,
        symbol: str,
        *,
        start: date | None,
        end: date | None,
        interval: str,  # noqa: ARG002
        auto_adjust: bool,  # noqa: ARG002
        back_adjust: bool,  # noqa: ARG002
    ) -> pd.DataFrame:
        sym = symbol.upper()
        if sym not in self._history_by_symbol:
            raise DataFetchError(f"MockProvider: no history for {sym}")
        df = self._history_by_symbol[sym].copy()
        if df.empty:
            return df
        if isinstance(df.index, pd.DatetimeIndex) and (start is not None or end is not None):
            idx = df.index
            if start is not None:
                df = df.loc[idx >= pd.Timestamp(start)]
            if end is not None:
                df = df.loc[df.index < pd.Timestamp(end)]
        return df

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:  # noqa: ARG002
        sym = symbol.upper()
        history = self._history_by_symbol.get(sym, pd.DataFrame()).copy()
        last_price = None
        if not history.empty and "Close" in history.columns:
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if not close.empty:
                last_price = float(close.iloc[-1])
        if sym in self._quote_by_symbol:
            last_price = float(self._quote_by_symbol[sym])
        return UnderlyingData(symbol=sym, last_price=last_price, history=history)

    def get_quote(self, symbol: str) -> float | None:
        sym = symbol.upper()
        if sym in self._quote_by_symbol:
            return float(self._quote_by_symbol[sym])
        history = self._history_by_symbol.get(sym)
        if history is not None and not history.empty and "Close" in history.columns:
            close = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1])
        return None

    def list_option_expiries(self, symbol: str) -> list[date]:
        sym = symbol.upper()
        explicit = self._expiries_by_symbol.get(sym)
        if explicit is not None:
            return sorted(set(explicit))
        inferred = sorted({exp for (s, exp) in self._chains_by_symbol_expiry.keys() if s == sym})
        return inferred

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
        key = (symbol.upper(), expiry)
        chain = self._chains_by_symbol_expiry.get(key)
        if chain is None:
            raise DataFetchError(f"MockProvider: no chain for {symbol.upper()} {expiry.isoformat()}")
        return chain

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict[str, Any]:
        key = (symbol.upper(), expiry)
        raw = self._raw_by_symbol_expiry.get(key)
        if raw is None:
            raise DataFetchError(f"MockProvider: no raw chain for {symbol.upper()} {expiry.isoformat()}")
        return raw

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:  # noqa: ARG002
        sym = symbol.upper()
        ev = self._earnings_by_symbol.get(sym)
        if ev is not None:
            return ev
        return EarningsEvent(symbol=sym, next_date=None, source="mock")
