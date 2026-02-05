from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from options_helper.data.market_types import DataFetchError, EarningsEvent, OptionsChain, UnderlyingData


class YFinanceClient:
    def __init__(self) -> None:
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def ticker(self, symbol: str) -> yf.Ticker:
        symbol = symbol.upper()
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def get_history(
        self,
        symbol: str,
        *,
        start: date | None,
        end: date | None,
        interval: str,
        auto_adjust: bool,
        back_adjust: bool,
    ) -> pd.DataFrame:
        try:
            ticker = self.ticker(symbol)
            if start is None and end is None:
                return ticker.history(
                    period="max",
                    interval=interval,
                    auto_adjust=auto_adjust,
                    back_adjust=back_adjust,
                )

            kwargs: dict[str, object] = {
                "interval": interval,
                "auto_adjust": auto_adjust,
                "back_adjust": back_adjust,
            }
            if start is not None:
                kwargs["start"] = start.isoformat()
            if end is not None:
                kwargs["end"] = end.isoformat()
            return ticker.history(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch history for {symbol}") from exc

    def list_option_expiries(self, symbol: str) -> list[date]:
        try:
            ticker = self.ticker(symbol)
            expiry_strs = list(ticker.options or [])
            expiries: list[date] = []
            for exp_str in expiry_strs:
                try:
                    expiries.append(date.fromisoformat(exp_str))
                except ValueError:
                    continue
            return sorted(set(expiries))
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch option expiries for {symbol}") from exc

    def get_quote(self, symbol: str) -> float | None:
        sym = symbol.upper()
        try:
            ticker = self.ticker(sym)
            fast_info = getattr(ticker, "fast_info", None)
            if isinstance(fast_info, dict):
                for key in ("last_price", "lastPrice", "regularMarketPrice", "last"):
                    val = fast_info.get(key)
                    if val is not None:
                        return float(val)

            info = getattr(ticker, "info", None)
            if isinstance(info, dict):
                for key in ("regularMarketPrice", "previousClose", "lastPrice"):
                    val = info.get(key)
                    if val is not None:
                        return float(val)

            history = ticker.history(period="5d", interval="1d", auto_adjust=False, back_adjust=False)
            if not history.empty and "Close" in history.columns:
                return float(history["Close"].iloc[-1])
            return None
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch quote for {sym}") from exc

    def get_underlying(self, symbol: str, *, period: str = "6mo", interval: str = "1d") -> UnderlyingData:
        try:
            ticker = self.ticker(symbol)
            history = ticker.history(period=period, interval=interval, auto_adjust=True, back_adjust=False)
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

    def get_next_earnings_event(self, symbol: str, *, today: date | None = None) -> EarningsEvent:
        """
        Best-effort fetch of the next earnings date.

        Yahoo/yfinance earnings fields are inconsistent:
        - may be missing entirely
        - may be a range/window (start/end)
        - may be stale or timezone-shifted
        """
        sym = symbol.upper()
        today_val = today or date.today()
        ticker = self.ticker(sym)

        # 1) Prefer the historical-ish earnings dates table when present.
        try:
            get_dates = getattr(ticker, "get_earnings_dates", None)
            if callable(get_dates):
                df = get_dates(limit=12)
                dt_candidates: list[date] = []
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # yfinance commonly returns a DatetimeIndex here.
                    if isinstance(df.index, pd.DatetimeIndex):
                        for ts in df.index:
                            try:
                                dt_candidates.append(ts.to_pydatetime().date())
                            except Exception:  # noqa: BLE001
                                continue
                    # Some variants include an explicit date column.
                    for col in ("Earnings Date", "EarningsDate", "earningsDate"):
                        if col in df.columns:
                            ser = df[col].dropna()
                            for v in ser.tolist():
                                d = _coerce_to_date(v)
                                if d is not None:
                                    dt_candidates.append(d)

                future = sorted({d for d in dt_candidates if d >= today_val})
                next_date = future[0] if future else None
                if next_date is not None:
                    return EarningsEvent(
                        symbol=sym,
                        next_date=next_date,
                        source="yfinance.get_earnings_dates",
                        raw={"rows": int(len(df)) if isinstance(df, pd.DataFrame) else None},
                    )
        except Exception:  # noqa: BLE001
            # Fall through to calendar parsing.
            pass

        # 2) Fall back to the calendar field (often a single date or a 2-date window).
        try:
            cal = getattr(ticker, "calendar", None)
            start = end = None

            if isinstance(cal, pd.DataFrame) and not cal.empty:
                # Typical yfinance layout: index contains keys (e.g. "Earnings Date") and a single column.
                if "Earnings Date" in cal.index:
                    row = cal.loc["Earnings Date"]
                    # Row might be a Series across columns.
                    values = row.tolist() if hasattr(row, "tolist") else [row]
                    start, end = _extract_date_window(values)
                elif "Earnings Date" in cal.columns:
                    values = cal["Earnings Date"].dropna().tolist()
                    start, end = _extract_date_window(values)

            elif isinstance(cal, dict):
                # Some variants return a dict-like payload.
                if "Earnings Date" in cal:
                    start, end = _extract_date_window([cal["Earnings Date"]])

            next_date = None
            if start is not None and start >= today_val:
                next_date = start
            elif end is not None and end >= today_val:
                next_date = end

            return EarningsEvent(
                symbol=sym,
                next_date=next_date,
                window_start=start,
                window_end=end,
                source="yfinance.calendar",
                raw=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise DataFetchError(f"Failed to fetch earnings calendar for {sym}") from exc


def _coerce_to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    # pandas Timestamp is a datetime subclass
    if isinstance(value, datetime):
        return value.date()
    try:
        # Try pandas Timestamp conversion
        if hasattr(value, "to_pydatetime"):
            dt = value.to_pydatetime()
            if isinstance(dt, datetime):
                return dt.date()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, str):
        s = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:  # noqa: BLE001
                continue
    return None


def _extract_date_window(values: list[Any]) -> tuple[date | None, date | None]:
    """
    Parse a possible earnings date window from a list of raw values.

    Yahoo may provide:
    - a single date
    - two dates (start/end of an earnings window)
    - nested lists/tuples
    """
    flat: list[Any] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            flat.extend(list(v))
        else:
            flat.append(v)

    dates = [_coerce_to_date(v) for v in flat]
    dates = [d for d in dates if d is not None]
    if not dates:
        return None, None
    dates_sorted = sorted(set(dates))
    if len(dates_sorted) == 1:
        return dates_sorted[0], None
    return dates_sorted[0], dates_sorted[1]


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
