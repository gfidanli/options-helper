from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf


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


class YFinanceClient:
    def __init__(self) -> None:
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def ticker(self, symbol: str) -> yf.Ticker:
        symbol = symbol.upper()
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:
        try:
            ticker = self.ticker(symbol)
            history = ticker.history(period=period, interval=interval, auto_adjust=False)
            last_price = None
            if not history.empty and "Close" in history.columns:
                last_price = float(history["Close"].iloc[-1])
            return UnderlyingData(symbol=symbol.upper(), last_price=last_price, history=history)
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch underlying data for {symbol}") from exc

    def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:
        expiry_str = expiry.isoformat()
        ticker = self.ticker(symbol)
        try:
            available = set(ticker.options or [])
            if available and expiry_str not in available:
                raise DataFetchError(
                    f"{symbol.upper()} has no options expiry {expiry_str} (available: {sorted(available)[:5]}...)"
                )
            chain = ticker.option_chain(expiry_str)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            return OptionsChain(symbol=symbol.upper(), expiry=expiry, calls=calls, puts=puts)
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch options chain for {symbol} {expiry_str}") from exc

    def get_options_chain_raw(self, symbol: str, expiry: date) -> dict[str, Any]:
        """
        Fetch the raw Yahoo options payload for an expiry.

        Notes:
        - `yfinance` converts the raw payload into a fixed-column DataFrame, dropping
          extra fields Yahoo returns. This method keeps the full payload so we can
          snapshot everything Yahoo provides.
        """
        expiry_str = expiry.isoformat()
        ticker = self.ticker(symbol)
        try:
            # Ensure expirations are populated.
            available = list(ticker.options or [])
            if available and expiry_str not in set(available):
                raise DataFetchError(
                    f"{symbol.upper()} has no options expiry {expiry_str} (available: {available[:5]}...)"
                )

            # `ticker._expirations` maps YYYY-MM-DD -> unix seconds used by the Yahoo endpoint.
            ts = getattr(ticker, "_expirations", {}).get(expiry_str)
            if ts is None:
                raise DataFetchError(f"Missing expiry mapping for {symbol.upper()} {expiry_str}")

            raw = ticker._download_options(ts)  # type: ignore[attr-defined]
            if not raw:
                raise DataFetchError(f"Empty options payload for {symbol.upper()} {expiry_str}")
            return raw
        except DataFetchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch raw options payload for {symbol} {expiry_str}") from exc


def _row_value(row: Any, key: str) -> float | int | None:
    if key not in row or pd.isna(row[key]):
        return None
    val = row[key]
    if isinstance(val, (int, float)):
        return val
    try:
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def contract_row_by_strike(df: pd.DataFrame, strike: float) -> pd.Series | None:
    if df.empty or "strike" not in df.columns:
        return None
    strike_series = df["strike"].astype(float)
    mask = (strike_series - float(strike)).abs() < 1e-6
    if mask.any():
        return df.loc[mask].iloc[0]
    # Fallback: closest strike (useful when floats differ by tiny amount)
    idx = (strike_series - float(strike)).abs().idxmin()
    if pd.isna(idx):
        return None
    return df.loc[idx]
