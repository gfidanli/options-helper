from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf


class CandleCacheError(RuntimeError):
    pass


HistoryFetcher = Callable[[str, date | None, date | None], pd.DataFrame]


def _parse_period_to_start(period: str, *, today: date) -> date | None:
    period = period.strip().lower()
    if period == "max":
        return None
    if period == "ytd":
        return date(today.year, 1, 1)

    units = {"d": 1, "wk": 7, "mo": 30, "y": 365}

    # yfinance periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    if period.endswith("wk"):
        n = int(period[:-2])
        days = n * units["wk"]
        return today - timedelta(days=days)
    for suffix in ("d", "mo", "y"):
        if period.endswith(suffix):
            n = int(period[: -len(suffix)])
            days = n * units[suffix]
            return today - timedelta(days=days)

    raise ValueError(f"Unsupported period format: {period}")


def _default_fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    if start is None and end is None:
        return ticker.history(period="max", interval="1d", auto_adjust=False)

    kwargs = {"interval": "1d", "auto_adjust": False}
    if start is not None:
        kwargs["start"] = start.isoformat()
    if end is not None:
        kwargs["end"] = end.isoformat()
    return ticker.history(**kwargs)


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as exc:  # noqa: BLE001
            raise CandleCacheError("Unable to normalize candle index to datetime") from exc
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


@dataclass(frozen=True)
class CandleStore:
    root_dir: Path
    fetcher: HistoryFetcher = _default_fetcher
    backfill_days: int = 5

    def _path_for_symbol(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_")
        return self.root_dir / f"{safe}.csv"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path_for_symbol(symbol)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return _normalize_history(df)

    def save(self, symbol: str, history: pd.DataFrame) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for_symbol(symbol)
        history.to_csv(path)

    def get_daily_history(self, symbol: str, *, period: str = "2y", today: date | None = None) -> pd.DataFrame:
        """
        Returns a locally cached daily OHLCV DataFrame, updating it incrementally from yfinance.

        Behavior:
        - Ensures the cache covers at least the requested `period` (best-effort).
        - Always refreshes the most recent candles by re-fetching a small tail window.
        """
        today = today or date.today()
        desired_start = _parse_period_to_start(period, today=today)

        cached = self.load(symbol)
        merged = cached

        # Backfill earlier history if required.
        if merged.empty:
            if desired_start is None:
                merged = _normalize_history(self.fetcher(symbol, None, None))
            else:
                merged = _normalize_history(self.fetcher(symbol, desired_start, None))
        elif desired_start is not None:
            first_cached = merged.index.min().date()
            if desired_start < first_cached:
                fetched = self.fetcher(symbol, desired_start, first_cached)
                fetched = _normalize_history(fetched)
                merged = pd.concat([fetched, merged]).sort_index()

        # Refresh the recent tail (captures late revisions / newly available data).
        if not merged.empty and self.backfill_days > 0:
            last_cached_dt = merged.index.max()
            refresh_start = (last_cached_dt.date() - timedelta(days=self.backfill_days))
            fetched = self.fetcher(symbol, refresh_start, None)
            fetched = _normalize_history(fetched)
            if not fetched.empty:
                merged = pd.concat([merged, fetched])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        if merged is None:
            merged = pd.DataFrame()

        # Persist only if we have something; avoid writing empty files for invalid symbols.
        if not merged.empty:
            self.save(symbol, merged)

        return merged


def last_close(history: pd.DataFrame) -> float | None:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    val = history["Close"].dropna()
    if val.empty:
        return None
    return float(val.iloc[-1])
