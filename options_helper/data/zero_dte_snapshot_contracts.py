from __future__ import annotations

import pandas as pd

from options_helper.analysis.osi import infer_settlement_style, parse_contract_symbol
from options_helper.data.zero_dte_dataset_helpers import first_present as _first_present


def normalize_snapshot_contracts(
    option_snapshot: pd.DataFrame | None,
    *,
    anchor_utc: pd.Timestamp,
) -> pd.DataFrame:
    if option_snapshot is None or option_snapshot.empty:
        return pd.DataFrame()

    frame = option_snapshot.copy()
    frame.columns = [str(col) for col in frame.columns]
    contract_col = _first_present(frame.columns, "contract_symbol", "contractSymbol", "osi")
    if contract_col is None:
        return pd.DataFrame()

    frame["contract_symbol"] = frame[contract_col].astype(str).str.strip().str.upper()
    frame = frame.loc[frame["contract_symbol"] != ""].copy()
    if frame.empty:
        return pd.DataFrame()

    parsed = frame["contract_symbol"].map(parse_contract_symbol)
    frame["option_type"] = _normalize_snapshot_option_type(frame, parsed)
    frame["expiry"] = _normalize_snapshot_expiry(frame, parsed)
    frame["strike"] = _normalize_snapshot_strike(frame, parsed)
    frame["settlement"] = _normalize_snapshot_settlement(frame, parsed)
    frame["underlying_norm"] = _normalize_snapshot_underlying(frame, parsed)
    frame["quote_ts"] = _normalize_snapshot_quote_ts(frame)
    frame["bid"] = _normalize_snapshot_numeric(frame, "bid")
    frame["ask"] = _normalize_snapshot_numeric(frame, "ask")
    return _dedupe_snapshot_contract_rows(frame, anchor_utc=anchor_utc)


def _normalize_snapshot_option_type(frame: pd.DataFrame, parsed: pd.Series) -> pd.Series:
    option_type_col = _first_present(frame.columns, "option_type", "optionType")
    if option_type_col is None:
        return parsed.map(lambda item: item.option_type if item is not None else None)
    option_type = frame[option_type_col].map(_normalize_option_type)
    fallback = option_type.isna()
    if fallback.any():
        option_type.loc[fallback] = parsed[fallback].map(lambda item: item.option_type if item is not None else None)
    return option_type


def _normalize_snapshot_expiry(frame: pd.DataFrame, parsed: pd.Series) -> pd.Series:
    expiry_col = _first_present(frame.columns, "expiry", "expiration", "expirationDate")
    if expiry_col is None:
        return parsed.map(lambda item: item.expiry if item is not None else None)
    expiry = pd.to_datetime(frame[expiry_col], errors="coerce").dt.date
    missing_expiry = expiry.isna()
    if missing_expiry.any():
        expiry.loc[missing_expiry] = parsed[missing_expiry].map(lambda item: item.expiry if item is not None else None)
    return expiry


def _normalize_snapshot_strike(frame: pd.DataFrame, parsed: pd.Series) -> pd.Series:
    strike_col = _first_present(frame.columns, "strike", "strike_price")
    if strike_col is None:
        return parsed.map(lambda item: item.strike if item is not None else float("nan"))
    strike = pd.to_numeric(frame[strike_col], errors="coerce")
    missing_strike = strike.isna()
    if missing_strike.any():
        strike.loc[missing_strike] = parsed[missing_strike].map(
            lambda item: item.strike if item is not None else float("nan")
        )
    return strike


def _normalize_snapshot_settlement(frame: pd.DataFrame, parsed: pd.Series) -> pd.Series:
    settlement_col = _first_present(
        frame.columns,
        "settlement",
        "settlement_style",
        "settlementType",
        "settleType",
    )
    if settlement_col is None:
        return parsed.map(infer_settlement_style)
    settlement = frame[settlement_col].map(_normalize_settlement)
    missing_settlement = settlement.isna()
    if missing_settlement.any():
        settlement.loc[missing_settlement] = parsed[missing_settlement].map(infer_settlement_style)
    return settlement


def _normalize_snapshot_underlying(frame: pd.DataFrame, parsed: pd.Series) -> pd.Series:
    underlying_col = _first_present(frame.columns, "underlying", "symbol", "root")
    if underlying_col is None:
        return parsed.map(lambda item: item.underlying_norm if item is not None else None)
    underlying = frame[underlying_col].astype(str).str.strip().str.upper()
    missing_underlying = underlying.isin({"", "NAN", "NONE"})
    if missing_underlying.any():
        underlying.loc[missing_underlying] = parsed[missing_underlying].map(
            lambda item: item.underlying_norm if item is not None else None
        )
    return underlying


def _normalize_snapshot_quote_ts(frame: pd.DataFrame) -> pd.Series:
    quote_ts_col = _first_present(
        frame.columns,
        "quote_timestamp",
        "quote_ts",
        "timestamp",
        "updated_at",
        "lastTradeDate",
    )
    if quote_ts_col is None:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[quote_ts_col], errors="coerce", utc=True)


def _normalize_snapshot_numeric(frame: pd.DataFrame, column_name: str) -> pd.Series:
    source_col = _first_present(frame.columns, column_name)
    if source_col is None:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[source_col], errors="coerce")


def _dedupe_snapshot_contract_rows(frame: pd.DataFrame, *, anchor_utc: pd.Timestamp) -> pd.DataFrame:
    if not frame["quote_ts"].isna().all():
        sorted_frame = _sort_snapshot_with_quote_priority(frame, anchor_utc=anchor_utc)
    else:
        sorted_frame = frame.sort_values(by=["contract_symbol"], kind="mergesort")
    deduped = sorted_frame.drop_duplicates(subset=["contract_symbol"], keep="first").copy()
    return deduped.reset_index(drop=True)


def _sort_snapshot_with_quote_priority(frame: pd.DataFrame, *, anchor_utc: pd.Timestamp) -> pd.DataFrame:
    prioritized = frame.copy()
    after_anchor = prioritized["quote_ts"] >= anchor_utc
    prioritized["quote_priority"] = 1
    prioritized.loc[after_anchor, "quote_priority"] = 0
    prioritized.loc[prioritized["quote_ts"].isna(), "quote_priority"] = 2
    prioritized["quote_delta_seconds"] = (prioritized["quote_ts"] - anchor_utc).abs().dt.total_seconds()
    prioritized["quote_delta_seconds"] = prioritized["quote_delta_seconds"].fillna(float("inf"))
    return prioritized.sort_values(
        by=["contract_symbol", "quote_priority", "quote_delta_seconds", "quote_ts"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )


def _normalize_option_type(raw: object) -> str | None:
    text = str(raw or "").strip().lower()
    if text in {"put", "p"}:
        return "put"
    if text in {"call", "c"}:
        return "call"
    return None


def _normalize_settlement(raw: object) -> str | None:
    text = str(raw or "").strip().lower()
    if text in {"pm", "p.m.", "p.m", "p"}:
        return "pm"
    if text in {"am", "a.m.", "a.m", "a"}:
        return "am"
    return None

