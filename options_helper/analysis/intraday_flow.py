from __future__ import annotations

from collections import Counter
import math
from typing import Final

import pandas as pd

from options_helper.schemas.research_metrics_contracts import (
    INTRADAY_FLOW_CONTRACT_FIELDS,
    INTRADAY_FLOW_TIME_BUCKET_FIELDS,
    classify_delta_bucket,
)


DEFAULT_CONTRACT_MULTIPLIER: Final[float] = 100.0
SUPPORTED_TIME_BUCKET_MINUTES: Final[tuple[int, ...]] = (5, 15)

_CLASSIFIED_TRADE_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "market_date",
    "timestamp",
    "contract_symbol",
    "expiry",
    "option_type",
    "strike",
    "bs_delta",
    "delta_bucket",
    "price",
    "size",
    "bid",
    "ask",
    "has_valid_quote",
    "direction",
    "direction_reason",
    "notional",
    "counted",
    "warning_codes",
)

_CONTRACT_TERM_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "market_date",
    "source",
    "expiry",
    "option_type",
    "strike",
    "delta_bucket",
    "buy_volume",
    "sell_volume",
    "unknown_volume",
    "buy_notional",
    "sell_notional",
    "net_notional",
    "trade_count",
    "unknown_trade_share",
    "quote_coverage_pct",
    "warnings",
)

_DIRECTION_BUY = "buy"
_DIRECTION_SELL = "sell"
_DIRECTION_UNKNOWN = "unknown"
_DIRECTION_DROPPED = "dropped"


def classify_intraday_trades(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    multiplier: float = DEFAULT_CONTRACT_MULTIPLIER,
    quote_tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """
    Classify intraday option trades with best-effort direction labels.

    Steps:
    - normalize input columns/aliases;
    - sort + de-duplicate timestamps deterministically before ``merge_asof``;
    - align each trade to the latest quote by ``contract_symbol``;
    - classify ``buy`` / ``sell`` / ``unknown`` using bid/ask bands;
    - mark malformed trades as ``dropped`` while preserving warning counters.
    """
    normalized_trades = _normalize_trades(trades)
    if normalized_trades.empty:
        return _empty_classified_trades()

    normalized_trades = normalized_trades.dropna(subset=["contract_symbol", "timestamp"]).copy()
    if normalized_trades.empty:
        return _empty_classified_trades()

    normalized_trades = _dedupe_trades(normalized_trades)

    normalized_quotes = _normalize_quotes(quotes)
    normalized_quotes = normalized_quotes.dropna(subset=["contract_symbol", "timestamp"]).copy()
    normalized_quotes = _dedupe_quotes(normalized_quotes)

    merged = _merge_trades_to_quotes(
        trades=normalized_trades,
        quotes=normalized_quotes,
        quote_tolerance=quote_tolerance,
    )
    classified = _classify_merged_trades(merged, multiplier=multiplier)

    return (
        classified.sort_values(["timestamp", "contract_symbol", "price", "size"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, _CLASSIFIED_TRADE_FIELDS]
    )


def summarize_intraday_contract_flow(
    classified_trades: pd.DataFrame,
    *,
    source: str = "unknown",
) -> pd.DataFrame:
    """
    Summarize trade-level classifications into per-contract/day flow rows.
    """
    if classified_trades is None or classified_trades.empty:
        return _empty_contract_flow()

    source_value = _clean_text(source) or "unknown"
    work = _prepare_classified_for_aggregation(classified_trades)
    if work.empty:
        return _empty_contract_flow()

    rows: list[dict[str, object]] = []
    grouped = work.groupby(["symbol", "market_date", "contract_symbol"], sort=True, dropna=False)
    for (symbol, market_date, contract_symbol), group in grouped:
        eligible = group[group["counted"]]
        trade_count = int(eligible.shape[0])
        buy_mask = eligible["direction"] == _DIRECTION_BUY
        sell_mask = eligible["direction"] == _DIRECTION_SELL
        unknown_mask = eligible["direction"] == _DIRECTION_UNKNOWN

        buy_volume = float(eligible.loc[buy_mask, "size"].sum()) if trade_count else 0.0
        sell_volume = float(eligible.loc[sell_mask, "size"].sum()) if trade_count else 0.0
        unknown_volume = float(eligible.loc[unknown_mask, "size"].sum()) if trade_count else 0.0

        buy_notional = float(eligible.loc[buy_mask, "notional"].sum()) if trade_count else 0.0
        sell_notional = float(eligible.loc[sell_mask, "notional"].sum()) if trade_count else 0.0

        unknown_trade_count = int(unknown_mask.sum()) if trade_count else 0
        covered_trade_count = int(eligible["has_valid_quote"].sum()) if trade_count else 0

        warnings = _format_warning_counter(_warning_counter_from_series(group["warning_codes"]))
        if trade_count == 0:
            warnings = _merge_warning_tokens(warnings, ["all_trades_dropped"])

        rows.append(
            {
                "symbol": str(symbol) if symbol is not None else "",
                "market_date": _format_market_date(market_date),
                "source": source_value,
                "contract_symbol": str(contract_symbol) if contract_symbol is not None else "",
                "expiry": _first_non_null(group["expiry"]),
                "option_type": _first_non_null(group["option_type"]),
                "strike": _first_non_null(group["strike"]),
                "delta_bucket": _first_non_null(group["delta_bucket"]),
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "unknown_volume": unknown_volume,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "net_notional": buy_notional - sell_notional,
                "trade_count": trade_count,
                "unknown_trade_share": (unknown_trade_count / trade_count) if trade_count else 0.0,
                "quote_coverage_pct": (covered_trade_count / trade_count) if trade_count else 0.0,
                "warnings": warnings,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_contract_flow()
    out = out.sort_values(
        ["symbol", "market_date", "expiry", "strike", "option_type", "contract_symbol"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return out.loc[:, INTRADAY_FLOW_CONTRACT_FIELDS]


def _resolve_aggregate_source(values: list[object]) -> str:
    source_values = sorted({text for text in (_clean_text(value) for value in values) if text is not None})
    if len(source_values) == 1:
        return source_values[0]
    if source_values:
        return "mixed"
    return "unknown"


def aggregate_intraday_flow_by_contract_terms(contract_flow: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate contract-day rows by expiry/strike/option_type/delta_bucket.
    """
    if contract_flow is None or contract_flow.empty:
        return _empty_contract_term_flow()

    work = contract_flow.copy()
    for col in (
        "buy_volume",
        "sell_volume",
        "unknown_volume",
        "buy_notional",
        "sell_notional",
        "net_notional",
        "trade_count",
        "unknown_trade_share",
        "quote_coverage_pct",
        "strike",
    ):
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in ("symbol", "market_date", "source", "expiry", "option_type", "delta_bucket"):
        if col not in work.columns:
            work[col] = None

    rows: list[dict[str, object]] = []
    grouped = work.groupby(
        ["symbol", "market_date", "expiry", "strike", "option_type", "delta_bucket"],
        sort=True,
        dropna=False,
    )
    for keys, group in grouped:
        symbol, market_date, expiry, strike, option_type, delta_bucket = keys
        trade_count = int(group["trade_count"].fillna(0.0).sum())
        unknown_trade_count = float((group["unknown_trade_share"] * group["trade_count"]).fillna(0.0).sum())
        covered_trade_count = float((group["quote_coverage_pct"] * group["trade_count"]).fillna(0.0).sum())
        source_value = _resolve_aggregate_source(group["source"].tolist())
        warnings = _format_warning_counter(_warning_counter_from_series(group.get("warnings")))

        buy_notional = float(group["buy_notional"].fillna(0.0).sum())
        sell_notional = float(group["sell_notional"].fillna(0.0).sum())
        rows.append(
            {
                "symbol": str(symbol) if symbol is not None else "",
                "market_date": _format_market_date(market_date),
                "source": source_value,
                "expiry": expiry,
                "option_type": option_type,
                "strike": strike,
                "delta_bucket": delta_bucket,
                "buy_volume": float(group["buy_volume"].fillna(0.0).sum()),
                "sell_volume": float(group["sell_volume"].fillna(0.0).sum()),
                "unknown_volume": float(group["unknown_volume"].fillna(0.0).sum()),
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "net_notional": buy_notional - sell_notional,
                "trade_count": trade_count,
                "unknown_trade_share": (unknown_trade_count / trade_count) if trade_count else 0.0,
                "quote_coverage_pct": (covered_trade_count / trade_count) if trade_count else 0.0,
                "warnings": warnings,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_contract_term_flow()
    out = out.sort_values(
        ["symbol", "market_date", "expiry", "strike", "option_type", "delta_bucket"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return out.loc[:, _CONTRACT_TERM_FIELDS]


def summarize_intraday_time_buckets(
    classified_trades: pd.DataFrame,
    *,
    bucket_minutes: int = 5,
) -> pd.DataFrame:
    """
    Summarize classified trades in fixed UTC time buckets (5m or 15m).
    """
    if bucket_minutes not in SUPPORTED_TIME_BUCKET_MINUTES:
        allowed = ", ".join(str(value) for value in SUPPORTED_TIME_BUCKET_MINUTES)
        raise ValueError(f"bucket_minutes must be one of: {allowed}")

    if classified_trades is None or classified_trades.empty:
        return _empty_time_bucket_flow()

    work = _prepare_classified_for_aggregation(classified_trades)
    eligible = work[work["counted"]].copy()
    if eligible.empty:
        return _empty_time_bucket_flow()

    eligible["bucket_start_utc"] = eligible["timestamp"].dt.floor(f"{bucket_minutes}min")

    rows: list[dict[str, object]] = []
    group_cols = [
        "symbol",
        "market_date",
        "bucket_start_utc",
        "contract_symbol",
        "expiry",
        "option_type",
        "strike",
        "delta_bucket",
    ]
    grouped = eligible.groupby(group_cols, sort=True, dropna=False)
    for keys, group in grouped:
        (
            symbol,
            market_date,
            bucket_start_utc,
            contract_symbol,
            expiry,
            option_type,
            strike,
            delta_bucket,
        ) = keys
        trade_count = int(group.shape[0])
        buy_notional = float(group.loc[group["direction"] == _DIRECTION_BUY, "notional"].sum())
        sell_notional = float(group.loc[group["direction"] == _DIRECTION_SELL, "notional"].sum())
        unknown_trade_count = int((group["direction"] == _DIRECTION_UNKNOWN).sum())

        rows.append(
            {
                "symbol": str(symbol) if symbol is not None else "",
                "market_date": _format_market_date(market_date),
                "bucket_start_utc": bucket_start_utc,
                "bucket_minutes": bucket_minutes,
                "contract_symbol": str(contract_symbol) if contract_symbol is not None else "",
                "expiry": expiry,
                "option_type": option_type,
                "strike": strike,
                "delta_bucket": delta_bucket,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "net_notional": buy_notional - sell_notional,
                "trade_count": trade_count,
                "unknown_trade_share": (unknown_trade_count / trade_count) if trade_count else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_time_bucket_flow()
    out = out.sort_values(
        ["symbol", "market_date", "bucket_start_utc", "expiry", "strike", "option_type", "contract_symbol"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    return out.loc[:, INTRADAY_FLOW_TIME_BUCKET_FIELDS]


def _normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "contract_symbol",
                "expiry",
                "option_type",
                "strike",
                "bs_delta",
                "price",
                "size",
            ]
        )

    df = trades.copy()
    out = pd.DataFrame(index=df.index)
    out["symbol"] = _normalize_symbol_series(
        _first_present(df, ("underlying", "root", "symbol", "underlying_symbol"))
    )
    out["timestamp"] = _to_utc_datetime_series(_first_present(df, ("timestamp", "ts", "time", "datetime")))
    out["contract_symbol"] = _clean_text_series(
        _first_present(df, ("contract_symbol", "contractSymbol", "option_symbol", "symbol"))
    )
    out["expiry"] = _normalize_expiry_series(_first_present(df, ("expiry", "expiration", "expiration_date")))
    out["option_type"] = _normalize_option_type_series(_first_present(df, ("option_type", "optionType", "type")))
    out["strike"] = pd.to_numeric(_first_present(df, ("strike",)), errors="coerce")
    out["bs_delta"] = pd.to_numeric(_first_present(df, ("bs_delta", "delta")), errors="coerce")
    out["price"] = pd.to_numeric(_first_present(df, ("price", "trade_price", "lastPrice")), errors="coerce")
    out["size"] = pd.to_numeric(_first_present(df, ("size", "qty", "quantity", "volume")), errors="coerce")
    return out


def _normalize_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes is None or quotes.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp",
                "contract_symbol",
                "expiry",
                "option_type",
                "strike",
                "bs_delta",
                "bid",
                "ask",
            ]
        )

    df = quotes.copy()
    out = pd.DataFrame(index=df.index)
    out["symbol"] = _normalize_symbol_series(
        _first_present(df, ("underlying", "root", "symbol", "underlying_symbol"))
    )
    out["timestamp"] = _to_utc_datetime_series(_first_present(df, ("timestamp", "ts", "time", "datetime")))
    out["contract_symbol"] = _clean_text_series(
        _first_present(df, ("contract_symbol", "contractSymbol", "option_symbol", "symbol"))
    )
    out["expiry"] = _normalize_expiry_series(_first_present(df, ("expiry", "expiration", "expiration_date")))
    out["option_type"] = _normalize_option_type_series(_first_present(df, ("option_type", "optionType", "type")))
    out["strike"] = pd.to_numeric(_first_present(df, ("strike",)), errors="coerce")
    out["bs_delta"] = pd.to_numeric(_first_present(df, ("bs_delta", "delta")), errors="coerce")
    out["bid"] = pd.to_numeric(_first_present(df, ("bid",)), errors="coerce")
    out["ask"] = pd.to_numeric(_first_present(df, ("ask",)), errors="coerce")
    return out


def _dedupe_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    out = trades.sort_values(
        ["contract_symbol", "timestamp", "price", "size", "expiry", "option_type", "strike"],
        kind="mergesort",
        na_position="last",
    )
    return out.drop_duplicates(subset=["contract_symbol", "timestamp", "price", "size"], keep="last")


def _dedupe_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return quotes.copy()

    out = quotes.copy()
    out["_valid_quote"] = ((out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])).astype(int)
    out["_sort_bid"] = out["bid"].fillna(-1.0)
    out["_sort_ask"] = out["ask"].fillna(-1.0)
    out = out.sort_values(
        ["contract_symbol", "timestamp", "_valid_quote", "_sort_bid", "_sort_ask"],
        kind="mergesort",
        na_position="last",
    )
    out = out.drop_duplicates(subset=["contract_symbol", "timestamp"], keep="last")
    return out.drop(columns=["_valid_quote", "_sort_bid", "_sort_ask"])


def _merge_trades_to_quotes(
    *,
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    quote_tolerance: pd.Timedelta | None,
) -> pd.DataFrame:
    left = trades.sort_values(["contract_symbol", "timestamp"], kind="mergesort")
    if quotes.empty:
        merged = left.copy()
        merged["bid"] = float("nan")
        merged["ask"] = float("nan")
        return merged

    right = quotes.sort_values(["contract_symbol", "timestamp"], kind="mergesort")
    right = right[
        ["contract_symbol", "timestamp", "bid", "ask", "symbol", "expiry", "option_type", "strike", "bs_delta"]
    ]
    merged_parts: list[pd.DataFrame] = []
    for contract_symbol, left_part in left.groupby("contract_symbol", dropna=False, sort=False):
        left_part = left_part.sort_values(["timestamp"], kind="mergesort")
        if pd.isna(contract_symbol):
            part = left_part.copy()
            part["bid"] = float("nan")
            part["ask"] = float("nan")
            merged_parts.append(part)
            continue

        right_part = right[right["contract_symbol"] == contract_symbol].copy()
        if right_part.empty:
            part = left_part.copy()
            part["bid"] = float("nan")
            part["ask"] = float("nan")
            merged_parts.append(part)
            continue

        right_part = right_part.sort_values(["timestamp"], kind="mergesort")
        part = pd.merge_asof(
            left_part,
            right_part,
            on="timestamp",
            direction="backward",
            suffixes=("", "_quote"),
            tolerance=quote_tolerance,
        )
        merged_parts.append(part)

    if not merged_parts:
        merged = left.copy()
        merged["bid"] = float("nan")
        merged["ask"] = float("nan")
        return merged

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values(["timestamp", "contract_symbol"], kind="mergesort")

    for col in ("symbol", "expiry", "option_type", "strike", "bs_delta"):
        quote_col = f"{col}_quote"
        if quote_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[quote_col])
            merged = merged.drop(columns=[quote_col])

    return merged


def _classify_merged_trades(merged: pd.DataFrame, *, multiplier: float) -> pd.DataFrame:
    out = merged.copy()
    out["price"] = pd.to_numeric(out.get("price"), errors="coerce")
    out["size"] = pd.to_numeric(out.get("size"), errors="coerce")
    out["bid"] = pd.to_numeric(out.get("bid"), errors="coerce")
    out["ask"] = pd.to_numeric(out.get("ask"), errors="coerce")
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    out["bs_delta"] = pd.to_numeric(out.get("bs_delta"), errors="coerce")

    size_missing = out["size"].isna()
    size_invalid = out["size"] < 0
    out["size"] = out["size"].where(~(size_missing | size_invalid), 0.0)

    invalid_price = out["price"].isna() | (out["price"] <= 0.0)
    zero_size = out["size"] <= 0.0
    counted = ~(invalid_price | zero_size)

    missing_quote = out["bid"].isna() | out["ask"].isna()
    has_valid_quote = (~missing_quote) & (out["bid"] > 0.0) & (out["ask"] > 0.0) & (out["ask"] >= out["bid"])
    invalid_quote = (~missing_quote) & ~has_valid_quote

    buy_mask = counted & has_valid_quote & (out["price"] >= out["ask"])
    sell_mask = counted & has_valid_quote & (out["price"] <= out["bid"])
    unknown_mask = counted & ~(buy_mask | sell_mask)

    direction = pd.Series(_DIRECTION_DROPPED, index=out.index, dtype="object")
    direction_reason = pd.Series("dropped", index=out.index, dtype="object")

    direction.loc[buy_mask] = _DIRECTION_BUY
    direction_reason.loc[buy_mask] = "at_or_above_ask"

    direction.loc[sell_mask] = _DIRECTION_SELL
    direction_reason.loc[sell_mask] = "at_or_below_bid"

    direction.loc[unknown_mask] = _DIRECTION_UNKNOWN
    direction_reason.loc[unknown_mask & missing_quote] = "missing_quote"
    direction_reason.loc[unknown_mask & invalid_quote] = "invalid_quote"
    direction_reason.loc[unknown_mask & has_valid_quote] = "inside_spread"
    direction_reason.loc[(~counted) & invalid_price] = "invalid_price"
    direction_reason.loc[(~counted) & (~invalid_price) & zero_size] = "zero_size"

    out["has_valid_quote"] = has_valid_quote
    out["direction"] = direction
    out["direction_reason"] = direction_reason
    out["counted"] = counted
    out["notional"] = (out["size"] * out["price"] * float(multiplier)).where(counted, 0.0)
    out["market_date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["delta_bucket"] = out["bs_delta"].map(classify_delta_bucket)

    warning_codes: list[list[str]] = []
    for idx in out.index:
        codes: list[str] = []
        if bool(size_missing.loc[idx]):
            codes.append("missing_size_coerced_zero")
        if bool(size_invalid.loc[idx]):
            codes.append("invalid_size_coerced_zero")
        if bool(invalid_price.loc[idx]):
            codes.append("dropped_invalid_price")
        if bool(zero_size.loc[idx]):
            codes.append("dropped_zero_size")
        if out.at[idx, "direction"] == _DIRECTION_UNKNOWN:
            reason = out.at[idx, "direction_reason"]
            if reason == "missing_quote":
                codes.append("unknown_missing_quote")
            elif reason == "invalid_quote":
                codes.append("unknown_invalid_quote")
            else:
                codes.append("unknown_inside_spread")
        if bool(counted.loc[idx]) and out.at[idx, "delta_bucket"] is None:
            codes.append("missing_delta_bucket")
        warning_codes.append(codes)

    out["warning_codes"] = warning_codes
    out["symbol"] = _normalize_symbol_series(out.get("symbol"))
    out["expiry"] = _normalize_expiry_series(out.get("expiry"))
    out["option_type"] = _normalize_option_type_series(out.get("option_type"))
    return out


def _prepare_classified_for_aggregation(classified_trades: pd.DataFrame) -> pd.DataFrame:
    if classified_trades is None or classified_trades.empty:
        return _empty_classified_trades()

    out = classified_trades.copy()
    for col in _CLASSIFIED_TRADE_FIELDS:
        if col not in out.columns:
            out[col] = None

    out["symbol"] = _normalize_symbol_series(out["symbol"])
    out["contract_symbol"] = _clean_text_series(out["contract_symbol"])
    out["timestamp"] = _to_utc_datetime_series(out["timestamp"])
    out["market_date"] = out["market_date"].where(out["market_date"].notna(), out["timestamp"].dt.strftime("%Y-%m-%d"))
    out["market_date"] = out["market_date"].map(_format_market_date)
    out["expiry"] = _normalize_expiry_series(out["expiry"])
    out["option_type"] = _normalize_option_type_series(out["option_type"])
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["size"] = pd.to_numeric(out["size"], errors="coerce").fillna(0.0)
    out["notional"] = pd.to_numeric(out["notional"], errors="coerce").fillna(0.0)
    out["bs_delta"] = pd.to_numeric(out["bs_delta"], errors="coerce")

    if "has_valid_quote" in out.columns:
        out["has_valid_quote"] = out["has_valid_quote"].fillna(False).astype(bool)
    else:
        out["has_valid_quote"] = False

    if "counted" in out.columns:
        out["counted"] = out["counted"].fillna(False).astype(bool)
    else:
        out["counted"] = out["direction"].isin({_DIRECTION_BUY, _DIRECTION_SELL, _DIRECTION_UNKNOWN})

    out["warning_codes"] = out["warning_codes"].map(_coerce_warning_tokens)
    out["delta_bucket"] = out["delta_bucket"].where(out["delta_bucket"].notna(), out["bs_delta"].map(classify_delta_bucket))

    out = out.dropna(subset=["contract_symbol", "timestamp"]).copy()
    return out


def _first_present(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        return None
    text = str(value).strip()
    return text or None


def _clean_text_series(values: pd.Series | object) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    return series.map(_clean_text)


def _normalize_symbol_series(values: pd.Series | object) -> pd.Series:
    cleaned = _clean_text_series(values)
    return cleaned.map(lambda value: value.upper() if value else None)


def _normalize_option_type_series(values: pd.Series | object) -> pd.Series:
    cleaned = _clean_text_series(values)
    return cleaned.map(_normalize_option_type)


def _normalize_option_type(value: str | None) -> str | None:
    if value is None:
        return None
    lower = value.lower()
    if lower in {"call", "c"}:
        return "call"
    if lower in {"put", "p"}:
        return "put"
    return lower


def _normalize_expiry_series(values: pd.Series | object) -> pd.Series:
    cleaned = _clean_text_series(values)
    parsed = pd.to_datetime(cleaned, errors="coerce", utc=True)
    parsed_text = parsed.dt.strftime("%Y-%m-%d")
    return parsed_text.where(parsed.notna(), cleaned)


def _to_utc_datetime_series(values: pd.Series | object) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    return pd.to_datetime(series, errors="coerce", utc=True)


def _first_non_null(values: pd.Series) -> object:
    if values is None or values.empty:
        return None
    for value in values.tolist():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:  # noqa: BLE001
            pass
        return value
    return None


def _format_market_date(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _coerce_warning_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [token for token in (str(item).strip() for item in value) if token]
    if isinstance(value, tuple | set):
        return [token for token in (str(item).strip() for item in value) if token]
    text = _clean_text(value)
    return [text] if text else []


def _warning_counter_from_series(values: pd.Series | None) -> Counter[str]:
    counter: Counter[str] = Counter()
    if values is None:
        return counter
    for item in values.tolist():
        for token in _coerce_warning_tokens(item):
            code, count = _decode_warning_token(token)
            counter[code] += count
    return counter


def _decode_warning_token(token: str) -> tuple[str, int]:
    if ":" not in token:
        return token, 1
    name, count_text = token.rsplit(":", 1)
    try:
        count = int(count_text)
    except ValueError:
        return token, 1
    if not name or count <= 0:
        return token, 1
    return name, count


def _format_warning_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return []
    formatted: list[str] = []
    for code in sorted(counter):
        count = int(counter[code])
        if count <= 1:
            formatted.append(code)
        else:
            formatted.append(f"{code}:{count}")
    return formatted


def _merge_warning_tokens(base: list[str], extra: list[str]) -> list[str]:
    merged = _warning_counter_from_series(pd.Series([base, extra], dtype="object"))
    return _format_warning_counter(merged)


def _empty_classified_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=_CLASSIFIED_TRADE_FIELDS)


def _empty_contract_flow() -> pd.DataFrame:
    return pd.DataFrame(columns=INTRADAY_FLOW_CONTRACT_FIELDS)


def _empty_contract_term_flow() -> pd.DataFrame:
    return pd.DataFrame(columns=_CONTRACT_TERM_FIELDS)


def _empty_time_bucket_flow() -> pd.DataFrame:
    return pd.DataFrame(columns=INTRADAY_FLOW_TIME_BUCKET_FIELDS)


__all__ = [
    "DEFAULT_CONTRACT_MULTIPLIER",
    "SUPPORTED_TIME_BUCKET_MINUTES",
    "aggregate_intraday_flow_by_contract_terms",
    "classify_intraday_trades",
    "summarize_intraday_contract_flow",
    "summarize_intraday_time_buckets",
]
