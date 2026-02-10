from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from options_helper.analysis.osi import ParsedContract, format_osi, format_osi_compact, normalize_underlying
from options_helper.data.alpaca_symbols import to_alpaca_symbol
from options_helper.models import MultiLegPosition, Portfolio, Position

_WIDE_SPREAD_PCT = 0.35
_TIMESTAMP_FIELDS = ("timestamp", "ts", "t", "time", "updated_at")


@dataclass(frozen=True)
class _LiveCaches:
    as_of: datetime | None
    option_quotes: dict[str, dict[str, Any]]
    option_trades: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class _ContractMetrics:
    contract_symbol: str
    live_symbol: str | None
    bid: float | None
    ask: float | None
    last: float | None
    mark: float | None
    spread_pct: float | None
    quote_age_seconds: float | None
    warnings: list[str]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(value_float):
        return None
    return value_float


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_symbol(value: Any) -> str | None:
    if value is None:
        return None
    symbol = str(value).strip().upper()
    return symbol or None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _first_present(row: Mapping[str, Any] | None, names: tuple[str, ...]) -> Any:
    if row is None:
        return None
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _extract_float(row: Mapping[str, Any] | None, names: tuple[str, ...]) -> float | None:
    return _coerce_float(_first_present(row, names))


def _extract_timestamp(row: Mapping[str, Any] | None) -> datetime | None:
    value = _first_present(row, _TIMESTAMP_FIELDS)
    return _coerce_timestamp(value)


def _cache_row(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        row = value.get("row")
        if isinstance(row, Mapping):
            return dict(row)
        return dict(value)
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            row = _cache_row(item)
            if row is not None:
                return row
        return None
    row_attr = getattr(value, "row", None)
    if isinstance(row_attr, Mapping):
        return dict(row_attr)
    return None


def _normalize_cache(cache: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(cache, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw_symbol, raw_row in cache.items():
        row = _cache_row(raw_row)
        if row is None:
            continue

        key = _normalize_symbol(raw_symbol)
        if key is not None:
            out[key] = row

        row_symbol = _normalize_symbol(
            _first_present(row, ("symbol", "S", "contract_symbol", "contractSymbol", "option_symbol"))
        )
        if row_symbol is not None:
            out[row_symbol] = row
    return out


def _resolve_as_of(
    live: Any,
    *,
    option_quotes: dict[str, dict[str, Any]],
    option_trades: dict[str, dict[str, Any]],
) -> datetime | None:
    for name in ("as_of", "snapshot_at", "snapshot_ts", "captured_at", "updated_at", "timestamp"):
        dt = _coerce_timestamp(_field(live, name))
        if dt is not None:
            return dt

    latest: datetime | None = None
    for cache in (option_quotes, option_trades):
        for row in cache.values():
            ts = _extract_timestamp(row)
            if ts is None:
                continue
            if latest is None or ts > latest:
                latest = ts
    return latest


def _build_live_caches(live: Any) -> _LiveCaches:
    option_quotes = _normalize_cache(_field(live, "option_quotes"))
    option_trades = _normalize_cache(_field(live, "option_trades"))
    as_of = _resolve_as_of(live, option_quotes=option_quotes, option_trades=option_trades)
    return _LiveCaches(as_of=as_of, option_quotes=option_quotes, option_trades=option_trades)


def _sanitize_stale_after_seconds(value: float) -> float:
    try:
        return max(0.0, float(value))
    except Exception:  # noqa: BLE001
        return 0.0


def _mark_price(*, bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    if last is not None and last > 0.0:
        return last
    if ask is not None and ask > 0.0:
        return ask
    if bid is not None and bid > 0.0:
        return bid
    return None


def _spread_pct(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0.0:
        return None
    return (ask - bid) / mid


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _lookup_row(
    cache: dict[str, dict[str, Any]],
    candidates: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    for key in candidates:
        row = cache.get(key)
        if row is not None:
            return row, key
    return None, None


def _contract_candidates(parsed: ParsedContract) -> tuple[str, list[str]]:
    compact = format_osi_compact(parsed)
    padded = format_osi(parsed)

    raw_candidates = [
        compact,
        to_alpaca_symbol(compact),
        padded,
        padded.replace(" ", ""),
        to_alpaca_symbol(padded),
        to_alpaca_symbol(padded.replace(" ", "")),
    ]
    candidates: list[str] = []
    for value in raw_candidates:
        normalized = _normalize_symbol(value)
        if normalized is None or normalized in candidates:
            continue
        candidates.append(normalized)
    return compact, candidates


def _contract_metrics(
    *,
    parsed: ParsedContract,
    option_quotes: dict[str, dict[str, Any]],
    option_trades: dict[str, dict[str, Any]],
    as_of: datetime | None,
    stale_after_seconds: float,
) -> _ContractMetrics:
    contract_symbol, candidates = _contract_candidates(parsed)
    quote_row, quote_live_symbol = _lookup_row(option_quotes, candidates)
    trade_row, trade_live_symbol = _lookup_row(option_trades, candidates)

    bid = _extract_float(quote_row, ("bid_price", "bidPrice", "bp", "bid"))
    ask = _extract_float(quote_row, ("ask_price", "askPrice", "ap", "ask"))
    last = _extract_float(trade_row, ("price", "last_price", "lastPrice", "last", "p"))
    if last is None:
        last = _extract_float(quote_row, ("last_price", "lastPrice", "last", "price"))

    mark = _mark_price(bid=bid, ask=ask, last=last)
    spread_pct = _spread_pct(bid, ask)

    quote_ts = _extract_timestamp(quote_row)
    trade_ts = _extract_timestamp(trade_row)
    event_ts = quote_ts if quote_ts is not None else trade_ts

    quote_age_seconds: float | None = None
    if as_of is not None and event_ts is not None:
        quote_age_seconds = max(0.0, float((as_of - event_ts).total_seconds()))

    warnings: list[str] = []
    if quote_row is None:
        warnings.append("missing_quote")
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
        warnings.append("missing_bid_or_ask")
    if mark is None:
        warnings.append("missing_mark")
    if spread_pct is not None:
        if spread_pct < 0.0:
            warnings.append("invalid_spread")
        elif spread_pct > _WIDE_SPREAD_PCT:
            warnings.append("wide_spread")
    if quote_age_seconds is not None and quote_age_seconds > stale_after_seconds:
        warnings.append("quote_stale")

    live_symbol = quote_live_symbol or trade_live_symbol
    return _ContractMetrics(
        contract_symbol=contract_symbol,
        live_symbol=live_symbol,
        bid=bid,
        ask=ask,
        last=last,
        mark=mark,
        spread_pct=spread_pct,
        quote_age_seconds=quote_age_seconds,
        warnings=_dedupe(warnings),
    )


def _position_contract(position: Position) -> ParsedContract:
    normalized_underlying = normalize_underlying(position.symbol)
    return ParsedContract(
        underlying=position.symbol,
        underlying_norm=normalized_underlying,
        expiry=position.expiry,
        option_type=position.option_type,
        strike=float(position.strike),
    )


def compute_live_position_rows(
    portfolio: Portfolio,
    live: Any,
    *,
    stale_after_seconds: float,
) -> pd.DataFrame:
    columns = [
        "id",
        "symbol",
        "structure",
        "option_type",
        "expiry",
        "dte",
        "strike",
        "contracts",
        "cost_basis",
        "contract_symbol",
        "live_symbol",
        "bid",
        "ask",
        "last",
        "mark",
        "spread_pct",
        "quote_age_seconds",
        "pnl_abs",
        "pnl_pct",
        "warnings",
        "as_of",
    ]
    stale_limit = _sanitize_stale_after_seconds(stale_after_seconds)
    caches = _build_live_caches(live)

    rows: list[dict[str, Any]] = []
    for position in portfolio.positions:
        if not isinstance(position, Position):
            continue

        contract = _position_contract(position)
        metrics = _contract_metrics(
            parsed=contract,
            option_quotes=caches.option_quotes,
            option_trades=caches.option_trades,
            as_of=caches.as_of,
            stale_after_seconds=stale_limit,
        )

        dte: int | None = None
        if caches.as_of is not None:
            dte = max(0, (position.expiry - caches.as_of.date()).days)

        pnl_abs = None
        pnl_pct = None
        if metrics.mark is not None:
            pnl_abs = (metrics.mark - float(position.cost_basis)) * 100.0 * float(position.contracts)
            if position.cost_basis > 0:
                pnl_pct = (metrics.mark / float(position.cost_basis)) - 1.0

        rows.append(
            {
                "id": position.id,
                "symbol": normalize_underlying(position.symbol),
                "structure": "single",
                "option_type": position.option_type,
                "expiry": position.expiry.isoformat(),
                "dte": dte,
                "strike": float(position.strike),
                "contracts": int(position.contracts),
                "cost_basis": float(position.cost_basis),
                "contract_symbol": metrics.contract_symbol,
                "live_symbol": metrics.live_symbol,
                "bid": metrics.bid,
                "ask": metrics.ask,
                "last": metrics.last,
                "mark": metrics.mark,
                "spread_pct": metrics.spread_pct,
                "quote_age_seconds": metrics.quote_age_seconds,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                "warnings": metrics.warnings,
                "as_of": None if caches.as_of is None else caches.as_of.isoformat(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns].sort_values(by=["symbol", "id"], kind="stable").reset_index(drop=True)


def compute_live_multileg_rows(
    portfolio: Portfolio,
    live: Any,
    *,
    stale_after_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structure_columns = [
        "id",
        "symbol",
        "structure",
        "legs",
        "net_debit",
        "net_mark",
        "net_pnl_abs",
        "net_pnl_pct",
        "quote_age_seconds_max",
        "missing_legs",
        "warnings",
        "as_of",
    ]
    legs_columns = [
        "position_id",
        "symbol",
        "leg_index",
        "side",
        "option_type",
        "expiry",
        "dte",
        "strike",
        "contracts",
        "signed_contracts",
        "contract_symbol",
        "live_symbol",
        "bid",
        "ask",
        "last",
        "mark",
        "spread_pct",
        "quote_age_seconds",
        "leg_value",
        "warnings",
        "as_of",
    ]
    stale_limit = _sanitize_stale_after_seconds(stale_after_seconds)
    caches = _build_live_caches(live)

    structure_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []

    for position in portfolio.positions:
        if not isinstance(position, MultiLegPosition):
            continue

        net_mark_value = 0.0
        net_mark_ready = True
        missing_legs = 0
        quote_age_max: float | None = None
        structure_warnings: list[str] = []

        for leg_index, leg in enumerate(position.legs, start=1):
            parsed = ParsedContract(
                underlying=position.symbol,
                underlying_norm=normalize_underlying(position.symbol),
                expiry=leg.expiry,
                option_type=leg.option_type,
                strike=float(leg.strike),
            )
            metrics = _contract_metrics(
                parsed=parsed,
                option_quotes=caches.option_quotes,
                option_trades=caches.option_trades,
                as_of=caches.as_of,
                stale_after_seconds=stale_limit,
            )

            dte: int | None = None
            if caches.as_of is not None:
                dte = max(0, (leg.expiry - caches.as_of.date()).days)

            signed_contracts = int(leg.signed_contracts)
            leg_value = None
            if metrics.mark is not None:
                leg_value = metrics.mark * float(signed_contracts) * 100.0
                net_mark_value += leg_value
            else:
                net_mark_ready = False
                missing_legs += 1

            if metrics.quote_age_seconds is not None:
                if quote_age_max is None or metrics.quote_age_seconds > quote_age_max:
                    quote_age_max = metrics.quote_age_seconds

            structure_warnings.extend(metrics.warnings)
            leg_rows.append(
                {
                    "position_id": position.id,
                    "symbol": normalize_underlying(position.symbol),
                    "leg_index": leg_index,
                    "side": leg.side,
                    "option_type": leg.option_type,
                    "expiry": leg.expiry.isoformat(),
                    "dte": dte,
                    "strike": float(leg.strike),
                    "contracts": int(leg.contracts),
                    "signed_contracts": signed_contracts,
                    "contract_symbol": metrics.contract_symbol,
                    "live_symbol": metrics.live_symbol,
                    "bid": metrics.bid,
                    "ask": metrics.ask,
                    "last": metrics.last,
                    "mark": metrics.mark,
                    "spread_pct": metrics.spread_pct,
                    "quote_age_seconds": metrics.quote_age_seconds,
                    "leg_value": leg_value,
                    "warnings": metrics.warnings,
                    "as_of": None if caches.as_of is None else caches.as_of.isoformat(),
                }
            )

        net_mark = net_mark_value if net_mark_ready else None
        net_debit = None if position.net_debit is None else float(position.net_debit)
        net_pnl_abs = None
        if net_mark is not None and net_debit is not None:
            net_pnl_abs = net_mark - net_debit

        net_pnl_pct = None
        if net_pnl_abs is not None and net_debit is not None and abs(net_debit) > 0.0:
            net_pnl_pct = net_pnl_abs / abs(net_debit)

        if missing_legs > 0:
            structure_warnings.insert(0, "missing_legs")
        if net_debit is None:
            structure_warnings.append("missing_net_debit")

        structure_rows.append(
            {
                "id": position.id,
                "symbol": normalize_underlying(position.symbol),
                "structure": "multi-leg",
                "legs": len(position.legs),
                "net_debit": net_debit,
                "net_mark": net_mark,
                "net_pnl_abs": net_pnl_abs,
                "net_pnl_pct": net_pnl_pct,
                "quote_age_seconds_max": quote_age_max,
                "missing_legs": missing_legs,
                "warnings": _dedupe(structure_warnings),
                "as_of": None if caches.as_of is None else caches.as_of.isoformat(),
            }
        )

    if not structure_rows:
        structure_df = pd.DataFrame(columns=structure_columns)
    else:
        structure_df = pd.DataFrame(structure_rows)
        for column in structure_columns:
            if column not in structure_df.columns:
                structure_df[column] = None
        structure_df = (
            structure_df[structure_columns]
            .sort_values(by=["symbol", "id"], kind="stable")
            .reset_index(drop=True)
        )

    if not leg_rows:
        legs_df = pd.DataFrame(columns=legs_columns)
    else:
        legs_df = pd.DataFrame(leg_rows)
        for column in legs_columns:
            if column not in legs_df.columns:
                legs_df[column] = None
        legs_df = (
            legs_df[legs_columns]
            .sort_values(by=["position_id", "leg_index"], kind="stable")
            .reset_index(drop=True)
        )

    return structure_df, legs_df


__all__ = [
    "compute_live_multileg_rows",
    "compute_live_position_rows",
]
