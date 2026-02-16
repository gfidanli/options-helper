from __future__ import annotations

import pandas as pd

from options_helper.data.market_types import DataFetchError

from .payload_helpers import ensure_required_columns


def _normalize_stock_bars(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        if "symbol" in out.index.names:
            level = out.index.names.index("symbol")
            symbols = out.index.get_level_values(level)
            out = out.xs(symbol if symbol in symbols else symbols[0], level="symbol")
        else:
            out = out.droplevel(0)

    if "symbol" in out.columns:
        sym_col = out["symbol"].astype(str).str.upper()
        target = symbol.upper()
        filtered = out[sym_col == target]
        out = filtered if not filtered.empty else out
        out = out.drop(columns=["symbol"])

    if not isinstance(out.index, pd.DatetimeIndex):
        for col in ("timestamp", "time", "t"):
            if col in out.columns:
                out = out.set_index(col)
                break

    rename_map = {
        "o": "Open",
        "open": "Open",
        "h": "High",
        "high": "High",
        "l": "Low",
        "low": "Low",
        "c": "Close",
        "close": "Close",
        "v": "Volume",
        "volume": "Volume",
        "vw": "VWAP",
        "vwap": "VWAP",
        "n": "Trade Count",
        "trade_count": "Trade Count",
        "tradecount": "Trade Count",
    }
    for col in list(out.columns):
        key = col.lower()
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})

    ensure_required_columns(out, {"Open", "High", "Low", "Close", "Volume"}, "Alpaca bars")

    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    if not isinstance(idx, pd.DatetimeIndex):
        raise DataFetchError("Unable to normalize Alpaca bar index to datetime")
    mask = ~idx.isna()
    out = out.loc[mask].copy()
    out.index = idx[mask].tz_convert(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _normalize_intraday_stock_bars(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = _normalize_intraday_layout(df, symbol_column="symbol", symbol_value=symbol, symbol_target="symbol")
    if "timestamp" not in out.columns:
        raise DataFetchError("Alpaca intraday bars missing timestamp column.")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"]).copy()
    ensure_required_columns(out, {"open", "high", "low", "close", "volume"}, "Alpaca intraday bars")

    for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.NA

    out = out.sort_values("timestamp", na_position="last")
    return out[columns].reset_index(drop=True)


def _normalize_intraday_option_bars(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "contractSymbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = _normalize_intraday_layout(
        df,
        symbol_column="contractSymbol",
        symbol_value=None,
        symbol_target="contractSymbol",
    )
    if "contractSymbol" not in out.columns:
        raise DataFetchError("Alpaca intraday option bars missing contract symbol column.")
    if "timestamp" not in out.columns:
        raise DataFetchError("Alpaca intraday option bars missing timestamp column.")

    out["contractSymbol"] = out["contractSymbol"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["contractSymbol", "timestamp"]).copy()
    ensure_required_columns(out, {"open", "high", "low", "close", "volume"}, "Alpaca intraday option bars")

    for col in ("open", "high", "low", "close", "volume", "trade_count", "vwap"):
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.NA

    out = out.sort_values(["contractSymbol", "timestamp"], na_position="last")
    return out[columns].reset_index(drop=True)


def _normalize_option_bars_daily_full(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_intraday_option_bars(df)
    columns = ["contractSymbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
    if out is None or out.empty:
        return pd.DataFrame(columns=columns)
    out = out.rename(columns={"timestamp": "ts"})
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].reset_index(drop=True)


def _normalize_intraday_layout(
    df: pd.DataFrame,
    *,
    symbol_column: str,
    symbol_value: str | None,
    symbol_target: str,
) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})

    rename_map = {
        "symbol": symbol_target,
        "option_symbol": symbol_target,
        "contract_symbol": symbol_target,
        "contractsymbol": symbol_target,
        "t": "timestamp",
        "time": "timestamp",
        "timestamp": "timestamp",
        "o": "open",
        "open": "open",
        "h": "high",
        "high": "high",
        "l": "low",
        "low": "low",
        "c": "close",
        "close": "close",
        "v": "volume",
        "volume": "volume",
        "n": "trade_count",
        "trade_count": "trade_count",
        "tradecount": "trade_count",
        "vw": "vwap",
        "vwap": "vwap",
    }
    for col in list(out.columns):
        key = str(col).lower()
        if key in rename_map:
            out = out.rename(columns={col: rename_map[key]})

    if symbol_value is not None and symbol_column in out.columns:
        sym_col = out[symbol_column].astype(str).str.upper()
        out = out[sym_col == symbol_value.upper()].copy()
        out = out.drop(columns=[symbol_column])

    return out
