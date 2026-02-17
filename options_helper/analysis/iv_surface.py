from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import pandas as pd

from options_helper.schemas.research_metrics_contracts import (
    DELTA_BUCKET_ORDER,
    IV_SURFACE_DELTA_BUCKET_FIELDS,
    IV_SURFACE_TENOR_FIELDS,
    IV_SURFACE_TENOR_TARGETS_DTE,
    classify_delta_bucket,
)


IV_SURFACE_TENOR_CHANGE_FIELDS: tuple[str, ...] = (
    "symbol",
    "as_of",
    "tenor_target_dte",
    "expiry",
    "dte",
    "atm_iv_change_pp",
    "atm_mark_change",
    "straddle_mark_change",
    "expected_move_pct_change_pp",
    "skew_25d_pp_change",
    "skew_10d_pp_change",
    "contracts_used_change",
    "warnings",
)

IV_SURFACE_DELTA_BUCKET_CHANGE_FIELDS: tuple[str, ...] = (
    "symbol",
    "as_of",
    "tenor_target_dte",
    "expiry",
    "option_type",
    "delta_bucket",
    "avg_iv_change_pp",
    "median_iv_change_pp",
    "n_contracts_change",
    "warnings",
)


@dataclass(frozen=True)
class IvSurfaceResult:
    tenor: pd.DataFrame
    delta_buckets: pd.DataFrame
    tenor_changes: pd.DataFrame
    delta_bucket_changes: pd.DataFrame
    warnings: tuple[str, ...]


def compute_iv_surface(
    snapshot: pd.DataFrame,
    *,
    symbol: str,
    as_of: date | str,
    spot: float,
    tenor_targets_dte: Sequence[int] = IV_SURFACE_TENOR_TARGETS_DTE,
    include_skew_10d: bool = True,
    previous_tenor: pd.DataFrame | None = None,
    previous_delta_buckets: pd.DataFrame | None = None,
) -> IvSurfaceResult:
    """
    Build deterministic IV surface tables from a single chain snapshot.

    Inputs are treated as best-effort. Missing/invalid fields degrade gracefully to
    null metrics with warnings rather than hard failures.
    """
    as_of_date = _parse_as_of_date(as_of)
    as_of_str = as_of_date.isoformat()
    symbol_norm = str(symbol).upper().strip()
    spot_value = _coerce_number(spot)

    targets = _normalize_targets(tenor_targets_dte)
    chain, base_warnings = _prepare_snapshot(snapshot)
    expiry_dte_pairs = _available_expiry_dtes(chain, as_of_date)

    tenor_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []

    for target in targets:
        selected = _select_expiry_for_target(expiry_dte_pairs, target)
        tenor_row, tenor_warnings = _compute_tenor_row(
            chain,
            symbol=symbol_norm,
            as_of=as_of_str,
            target_dte=target,
            selected=selected,
            spot=spot_value,
            base_warnings=base_warnings,
            include_skew_10d=include_skew_10d,
        )
        tenor_rows.append(tenor_row)

        delta_rows.extend(
            _compute_delta_bucket_rows(
                chain,
                symbol=symbol_norm,
                as_of=as_of_str,
                target_dte=target,
                selected=selected,
                base_warnings=base_warnings,
                tenor_warnings=tenor_warnings,
            )
        )

    tenor_df = _rows_to_frame(tenor_rows, IV_SURFACE_TENOR_FIELDS)
    delta_df = _rows_to_frame(delta_rows, IV_SURFACE_DELTA_BUCKET_FIELDS)

    tenor_changes, delta_bucket_changes = compute_iv_surface_changes(
        tenor_df,
        delta_df,
        previous_tenor=previous_tenor,
        previous_delta_buckets=previous_delta_buckets,
    )

    return IvSurfaceResult(
        tenor=tenor_df,
        delta_buckets=delta_df,
        tenor_changes=tenor_changes,
        delta_bucket_changes=delta_bucket_changes,
        warnings=tuple(_dedupe_preserve_order(base_warnings)),
    )


def compute_iv_surface_changes(
    current_tenor: pd.DataFrame,
    current_delta_buckets: pd.DataFrame,
    *,
    previous_tenor: pd.DataFrame | None = None,
    previous_delta_buckets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute day-over-day changes for IV surface tables.

    When previous tables are not supplied, corresponding change frames are empty.
    """
    tenor_changes = _empty_frame(IV_SURFACE_TENOR_CHANGE_FIELDS)
    delta_changes = _empty_frame(IV_SURFACE_DELTA_BUCKET_CHANGE_FIELDS)

    if previous_tenor is not None:
        tenor_changes = _compute_tenor_changes(current_tenor, previous_tenor)
    if previous_delta_buckets is not None:
        delta_changes = _compute_delta_bucket_changes(current_delta_buckets, previous_delta_buckets)

    return tenor_changes, delta_changes


def _compute_tenor_changes(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current_rows = _rows_with_column_order(current, IV_SURFACE_TENOR_FIELDS)
    previous_lookup, previous_warnings = _build_tenor_lookup(previous)

    rows: list[dict[str, object]] = []
    for _, cur in current_rows.iterrows():
        row_warnings = _coerce_warning_list(cur.get("warnings")) + list(previous_warnings)

        target = _coerce_int(cur.get("tenor_target_dte"))
        prev_row = previous_lookup.get(target) if target is not None else None

        if target is None:
            row_warnings.append("invalid_tenor_target_dte")
        if prev_row is None:
            row_warnings.append("missing_previous_row")

        rows.append(
            {
                "symbol": cur.get("symbol"),
                "as_of": cur.get("as_of"),
                "tenor_target_dte": cur.get("tenor_target_dte"),
                "expiry": cur.get("expiry"),
                "dte": cur.get("dte"),
                "atm_iv_change_pp": _scaled_change(cur.get("atm_iv"), _get_value(prev_row, "atm_iv"), 100.0),
                "atm_mark_change": _scaled_change(cur.get("atm_mark"), _get_value(prev_row, "atm_mark"), 1.0),
                "straddle_mark_change": _scaled_change(
                    cur.get("straddle_mark"),
                    _get_value(prev_row, "straddle_mark"),
                    1.0,
                ),
                "expected_move_pct_change_pp": _scaled_change(
                    cur.get("expected_move_pct"),
                    _get_value(prev_row, "expected_move_pct"),
                    100.0,
                ),
                "skew_25d_pp_change": _scaled_change(
                    cur.get("skew_25d_pp"),
                    _get_value(prev_row, "skew_25d_pp"),
                    1.0,
                ),
                "skew_10d_pp_change": _scaled_change(
                    cur.get("skew_10d_pp"),
                    _get_value(prev_row, "skew_10d_pp"),
                    1.0,
                ),
                "contracts_used_change": _count_change(
                    cur.get("contracts_used"),
                    _get_value(prev_row, "contracts_used"),
                ),
                "warnings": _dedupe_preserve_order(row_warnings),
            }
        )

    return _rows_to_frame(rows, IV_SURFACE_TENOR_CHANGE_FIELDS)


def _compute_delta_bucket_changes(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current_rows = _rows_with_column_order(current, IV_SURFACE_DELTA_BUCKET_FIELDS)
    previous_lookup, previous_warnings = _build_delta_lookup(previous)

    rows: list[dict[str, object]] = []
    for _, cur in current_rows.iterrows():
        row_warnings = _coerce_warning_list(cur.get("warnings")) + list(previous_warnings)

        key = (
            _coerce_int(cur.get("tenor_target_dte")),
            _normalize_option_type(cur.get("option_type")),
            _coerce_text(cur.get("delta_bucket")),
        )
        prev_row = previous_lookup.get(key)
        if prev_row is None:
            row_warnings.append("missing_previous_row")

        rows.append(
            {
                "symbol": cur.get("symbol"),
                "as_of": cur.get("as_of"),
                "tenor_target_dte": cur.get("tenor_target_dte"),
                "expiry": cur.get("expiry"),
                "option_type": cur.get("option_type"),
                "delta_bucket": cur.get("delta_bucket"),
                "avg_iv_change_pp": _scaled_change(cur.get("avg_iv"), _get_value(prev_row, "avg_iv"), 100.0),
                "median_iv_change_pp": _scaled_change(
                    cur.get("median_iv"),
                    _get_value(prev_row, "median_iv"),
                    100.0,
                ),
                "n_contracts_change": _count_change(
                    cur.get("n_contracts"),
                    _get_value(prev_row, "n_contracts"),
                ),
                "warnings": _dedupe_preserve_order(row_warnings),
            }
        )

    return _rows_to_frame(rows, IV_SURFACE_DELTA_BUCKET_CHANGE_FIELDS)


def _prepare_snapshot(snapshot: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    if snapshot is None:
        data = pd.DataFrame()
    else:
        data = snapshot.copy()

    warnings: list[str] = []
    if data.empty:
        warnings.append("empty_snapshot")

    index = data.index

    if "expiry" in data.columns:
        parsed = pd.to_datetime(data["expiry"], errors="coerce")
        expiry_date = pd.Series(parsed.dt.date, index=index, dtype="object")
    else:
        warnings.append("missing_expiry")
        expiry_date = pd.Series([None] * len(data), index=index, dtype="object")

    if "optionType" in data.columns:
        option_type = data["optionType"].astype(str).str.lower().str.strip()
        option_type = option_type.where(option_type.isin(["call", "put"]))
    else:
        warnings.append("missing_optionType")
        option_type = pd.Series([None] * len(data), index=index, dtype="object")

    if "strike" in data.columns:
        strike = pd.to_numeric(data["strike"], errors="coerce")
    else:
        warnings.append("missing_strike")
        strike = pd.Series([float("nan")] * len(data), index=index, dtype="float64")

    if "impliedVolatility" in data.columns:
        iv = pd.to_numeric(data["impliedVolatility"], errors="coerce")
        iv = iv.where(iv > 0)
    else:
        warnings.append("missing_impliedVolatility")
        iv = pd.Series([float("nan")] * len(data), index=index, dtype="float64")

    if "bs_delta" in data.columns:
        delta = pd.to_numeric(data["bs_delta"], errors="coerce")
        delta = delta.where((delta >= -1.0) & (delta <= 1.0))
    else:
        warnings.append("missing_bs_delta")
        delta = pd.Series([float("nan")] * len(data), index=index, dtype="float64")

    mark, mark_warning = _compute_mark(data)
    if mark_warning is not None:
        warnings.append(mark_warning)

    out = data.copy()
    out["_expiry_date"] = expiry_date
    out["_option_type"] = option_type
    out["_strike"] = strike
    out["_iv"] = iv
    out["_delta"] = delta
    out["_delta_bucket"] = delta.map(classify_delta_bucket)
    out["_mark"] = mark.where(mark > 0)

    return out, _dedupe_preserve_order(warnings)


def _compute_mark(df: pd.DataFrame) -> tuple[pd.Series, str | None]:
    mark = _col_as_float(df, "mark")
    mark = mark.where(mark > 0)

    bid = _col_as_float(df, "bid")
    ask = _col_as_float(df, "ask")
    last = _col_as_float(df, "lastPrice")

    mid = ((bid + ask) / 2.0).where((bid > 0) & (ask > 0))
    mark = mark.fillna(mid)
    mark = mark.fillna(last.where(last > 0))
    mark = mark.fillna(ask.where(ask > 0))
    mark = mark.fillna(bid.where(bid > 0))

    mark_warning = None
    has_quote_inputs = any(col in df.columns for col in ("bid", "ask", "lastPrice", "mark"))
    if not has_quote_inputs:
        mark_warning = "missing_quotes"

    return mark, mark_warning


def _compute_tenor_atm_metrics(sub: pd.DataFrame, *, atm_strike: float) -> tuple[float | None, float | None, float | None, list[str]]:
    warnings: list[str] = []
    atm_iv = _atm_iv(sub, atm_strike)
    if atm_iv is None:
        warnings.append("missing_atm_iv")

    call_mark = _atm_side_mark(sub, "call", atm_strike)
    put_mark = _atm_side_mark(sub, "put", atm_strike)
    if call_mark is not None or put_mark is not None:
        marks = [x for x in (call_mark, put_mark) if x is not None]
        atm_mark = float(sum(marks) / len(marks))
    else:
        atm_mark = None
        warnings.append("missing_atm_mark")

    straddle_mark = None
    if call_mark is not None and put_mark is not None:
        straddle_mark = float(call_mark + put_mark)
    else:
        warnings.append("missing_atm_straddle")
    return atm_iv, atm_mark, straddle_mark, warnings


def _compute_tenor_row(
    chain: pd.DataFrame,
    *,
    symbol: str,
    as_of: str,
    target_dte: int,
    selected: tuple[date, int] | None,
    spot: float | None,
    base_warnings: list[str],
    include_skew_10d: bool,
) -> tuple[dict[str, object], list[str]]:
    warnings = list(base_warnings)

    expiry_str: str | None = None
    dte: int | None = None
    tenor_gap: int | None = None
    atm_strike = atm_iv = atm_mark = straddle_mark = expected_move_pct = skew_25d_pp = skew_10d_pp = None
    contracts_used = 0

    if selected is None:
        warnings.append("no_valid_expiry_for_tenor")
    else:
        expiry, dte = selected
        expiry_str = expiry.isoformat()
        tenor_gap = int(dte - target_dte)

        sub = chain[chain["_expiry_date"] == expiry]
        contracts_used = int((sub["_iv"].notna() | sub["_mark"].notna()).sum())
        if contracts_used <= 0:
            warnings.append("no_valid_contract_metrics")

        if spot is None:
            warnings.append("invalid_spot")
        else:
            atm_strike = _pick_atm_strike(sub, spot)
            if atm_strike is None:
                warnings.append("missing_atm_strike")

        if atm_strike is not None:
            atm_iv, atm_mark, straddle_mark, atm_warnings = _compute_tenor_atm_metrics(sub, atm_strike=atm_strike)
            warnings.extend(atm_warnings)

        if straddle_mark is not None and spot is not None and spot > 0:
            expected_move_pct = float(straddle_mark / spot)
        elif straddle_mark is not None:
            warnings.append("non_positive_spot_for_expected_move")

        skew_25d_pp = _compute_skew_pp(sub, target_abs_delta=0.25)
        if skew_25d_pp is None:
            warnings.append("missing_skew_25d")

        if include_skew_10d:
            skew_10d_pp = _compute_skew_pp(sub, target_abs_delta=0.10)
            if skew_10d_pp is None:
                warnings.append("missing_skew_10d")

    row = {
        "symbol": symbol,
        "as_of": as_of,
        "tenor_target_dte": int(target_dte),
        "expiry": expiry_str,
        "dte": dte,
        "tenor_gap_dte": tenor_gap,
        "atm_strike": atm_strike,
        "atm_iv": atm_iv,
        "atm_mark": atm_mark,
        "straddle_mark": straddle_mark,
        "expected_move_pct": expected_move_pct,
        "skew_25d_pp": skew_25d_pp,
        "skew_10d_pp": skew_10d_pp,
        "contracts_used": int(contracts_used),
        "warnings": _dedupe_preserve_order(warnings),
    }
    return row, _dedupe_preserve_order(warnings)


def _compute_delta_bucket_rows(
    chain: pd.DataFrame,
    *,
    symbol: str,
    as_of: str,
    target_dte: int,
    selected: tuple[date, int] | None,
    base_warnings: list[str],
    tenor_warnings: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    expiry_str: str | None = None
    sub = chain.iloc[0:0]
    if selected is not None:
        expiry, _dte = selected
        expiry_str = expiry.isoformat()
        sub = chain[chain["_expiry_date"] == expiry]

    for option_type in ("call", "put"):
        for bucket in DELTA_BUCKET_ORDER:
            row_warnings = list(base_warnings)
            avg_iv = median_iv = None
            n_contracts = 0

            if selected is None:
                row_warnings.append("no_valid_expiry_for_tenor")
            else:
                filt = sub
                filt = filt[filt["_option_type"] == option_type]
                filt = filt[filt["_delta_bucket"] == bucket]
                filt = filt[filt["_iv"].notna()]

                n_contracts = int(len(filt))
                if n_contracts > 0:
                    avg_iv = float(filt["_iv"].mean())
                    median_iv = float(filt["_iv"].median())

                if n_contracts == 0:
                    row_warnings.append("empty_bucket")

            row_warnings = _dedupe_preserve_order(row_warnings + tenor_warnings)

            rows.append(
                {
                    "symbol": symbol,
                    "as_of": as_of,
                    "tenor_target_dte": int(target_dte),
                    "expiry": expiry_str,
                    "option_type": option_type,
                    "delta_bucket": bucket,
                    "avg_iv": avg_iv,
                    "median_iv": median_iv,
                    "n_contracts": n_contracts,
                    "warnings": row_warnings,
                }
            )

    return rows


def _build_tenor_lookup(previous: pd.DataFrame) -> tuple[dict[int, pd.Series], list[str]]:
    warnings: list[str] = []
    if previous.empty:
        return {}, warnings
    if "tenor_target_dte" not in previous.columns:
        return {}, ["previous_tenor_missing_tenor_target_dte"]

    prev = previous.copy()
    prev["_target"] = pd.to_numeric(prev["tenor_target_dte"], errors="coerce")
    prev = prev.dropna(subset=["_target"])
    if prev.empty:
        return {}, ["previous_tenor_missing_valid_targets"]

    sort_cols = [col for col in ["_target", "as_of", "expiry"] if col in prev.columns]
    if sort_cols:
        prev = prev.sort_values(sort_cols, ascending=True, kind="mergesort")

    lookup: dict[int, pd.Series] = {}
    for _, row in prev.iterrows():
        target = _coerce_int(row.get("_target"))
        if target is None:
            continue
        lookup[target] = row

    return lookup, warnings


def _build_delta_lookup(previous: pd.DataFrame) -> tuple[dict[tuple[int | None, str | None, str | None], pd.Series], list[str]]:
    warnings: list[str] = []
    if previous.empty:
        return {}, warnings

    required = {"tenor_target_dte", "option_type", "delta_bucket"}
    missing = sorted(required - set(previous.columns))
    if missing:
        return {}, ["previous_delta_missing_columns:" + ",".join(missing)]

    prev = previous.copy()
    prev["_target"] = pd.to_numeric(prev["tenor_target_dte"], errors="coerce")
    prev["_option_type"] = prev["option_type"].astype(str).str.lower().str.strip()
    prev["_delta_bucket"] = prev["delta_bucket"].astype(str).str.strip()

    prev = prev.dropna(subset=["_target"])
    sort_cols = [col for col in ["_target", "_option_type", "_delta_bucket", "as_of", "expiry"] if col in prev.columns]
    if sort_cols:
        prev = prev.sort_values(sort_cols, ascending=True, kind="mergesort")

    lookup: dict[tuple[int | None, str | None, str | None], pd.Series] = {}
    for _, row in prev.iterrows():
        key = (
            _coerce_int(row.get("_target")),
            _normalize_option_type(row.get("_option_type")),
            _coerce_text(row.get("_delta_bucket")),
        )
        lookup[key] = row

    return lookup, warnings


def _available_expiry_dtes(chain: pd.DataFrame, as_of: date) -> list[tuple[date, int]]:
    if chain.empty or "_expiry_date" not in chain.columns:
        return []

    out: list[tuple[date, int]] = []
    for expiry in sorted(set(chain["_expiry_date"].dropna().tolist())):
        if not isinstance(expiry, date):
            continue
        dte = int((expiry - as_of).days)
        if dte <= 0:
            continue
        out.append((expiry, dte))
    return out


def _select_expiry_for_target(expiry_dte_pairs: list[tuple[date, int]], target_dte: int) -> tuple[date, int] | None:
    if not expiry_dte_pairs:
        return None
    return min(expiry_dte_pairs, key=lambda x: (abs(x[1] - int(target_dte)), x[0]))


def _pick_atm_strike(sub: pd.DataFrame, spot: float) -> float | None:
    if sub.empty:
        return None

    strikes = pd.to_numeric(sub["_strike"], errors="coerce").dropna()
    if strikes.empty:
        return None

    unique_strikes = sorted({float(x) for x in strikes.tolist()}, key=lambda strike: (abs(strike - spot), strike))
    if not unique_strikes:
        return None
    return float(unique_strikes[0])


def _atm_iv(sub: pd.DataFrame, strike: float) -> float | None:
    mask = (pd.to_numeric(sub["_strike"], errors="coerce") - float(strike)).abs() <= 1e-9
    iv = pd.to_numeric(sub.loc[mask, "_iv"], errors="coerce").dropna()
    if iv.empty:
        return None
    return float(iv.mean())


def _atm_side_mark(sub: pd.DataFrame, option_type: str, strike: float) -> float | None:
    strike_mask = (pd.to_numeric(sub["_strike"], errors="coerce") - float(strike)).abs() <= 1e-9
    side = sub[strike_mask & (sub["_option_type"] == option_type)]
    marks = pd.to_numeric(side["_mark"], errors="coerce")
    marks = marks[marks > 0]
    if marks.empty:
        return None
    return float(marks.mean())


def _compute_skew_pp(sub: pd.DataFrame, *, target_abs_delta: float) -> float | None:
    call_iv = _pick_iv_near_delta(sub, option_type="call", target_delta=float(target_abs_delta))
    put_iv = _pick_iv_near_delta(sub, option_type="put", target_delta=-float(target_abs_delta))
    if call_iv is None or put_iv is None:
        return None
    return float((put_iv - call_iv) * 100.0)


def _pick_iv_near_delta(sub: pd.DataFrame, *, option_type: str, target_delta: float) -> float | None:
    if sub.empty:
        return None

    side = sub[sub["_option_type"] == option_type].copy()
    if side.empty:
        return None

    side = side.dropna(subset=["_delta", "_iv"])
    if side.empty:
        return None

    side["_dist"] = (pd.to_numeric(side["_delta"], errors="coerce") - float(target_delta)).abs()
    side["_strike_sort"] = pd.to_numeric(side["_strike"], errors="coerce").fillna(float("inf"))
    side = side.sort_values(["_dist", "_strike_sort"], ascending=[True, True], kind="mergesort")
    if side.empty:
        return None

    iv = _coerce_number(side.iloc[0].get("_iv"))
    return iv


def _normalize_targets(targets: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for target in targets:
        val = _coerce_int(target)
        if val is None or val <= 0 or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return tuple(out) if out else IV_SURFACE_TENOR_TARGETS_DTE


def _parse_as_of_date(as_of: date | str) -> date:
    if isinstance(as_of, date):
        return as_of

    text = str(as_of).strip()
    if not text:
        raise ValueError("as_of must be a valid date")

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        ts = pd.to_datetime(text, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"as_of must be a valid date: {as_of}") from None
        return ts.date()


def _rows_with_column_order(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out.loc[:, list(columns)]


def _rows_to_frame(rows: list[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return _empty_frame(columns)
    out = pd.DataFrame(rows)
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out.loc[:, list(columns)]


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def _col_as_float(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def _scaled_change(current: object, previous: object, scale: float) -> float | None:
    cur = _coerce_number(current)
    prev = _coerce_number(previous)
    if cur is None or prev is None:
        return None
    return float((cur - prev) * scale)


def _count_change(current: object, previous: object) -> int | None:
    cur = _coerce_int(current)
    prev = _coerce_int(previous)
    if cur is None or prev is None:
        return None
    return int(cur - prev)


def _coerce_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _coerce_int(value: object) -> int | None:
    number = _coerce_number(value)
    if number is None:
        return None
    return int(number)


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_option_type(value: object) -> str | None:
    text = _coerce_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered not in {"call", "put"}:
        return None
    return lowered


def _coerce_warning_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, tuple | set):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _get_value(row: pd.Series | None, field: str) -> object:
    if row is None:
        return None
    return row.get(field)


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "IV_SURFACE_DELTA_BUCKET_CHANGE_FIELDS",
    "IV_SURFACE_TENOR_CHANGE_FIELDS",
    "IvSurfaceResult",
    "compute_iv_surface",
    "compute_iv_surface_changes",
]
