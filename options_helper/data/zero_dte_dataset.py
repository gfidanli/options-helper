from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from options_helper.analysis.osi import infer_settlement_style, parse_contract_symbol
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.zero_dte_dataset_helpers import (
    coerce_decision_timestamp as _coerce_decision_timestamp,
    duckdb_table_exists as _duckdb_table_exists,
    first_present as _first_present,
    normalize_symbol as _normalize_symbol,
    us_equity_half_days as _us_equity_half_days,
    us_equity_holidays as _us_equity_holidays,
)
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.zero_dte_put_study import QuoteQualityStatus, SkipReason


DEFAULT_PROXY_UNDERLYING = "SPY"
DEFAULT_MARKET_TZ = "America/New_York"
_DEFAULT_STOCK_SPEC = ("stocks", "bars", "1Min")
_DEFAULT_OPTION_SPEC = ("options", "bars", "1Min")

_UNDERLYING_COLUMNS = (
    "timestamp",
    "timestamp_market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
)

_STATE_COLUMNS = (
    "session_date",
    "decision_ts",
    "decision_ts_market",
    "bar_ts",
    "bar_ts_market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "bar_age_seconds",
    "status",
    "is_half_day",
)

_STRIKE_PREMIUM_COLUMNS = (
    "session_date",
    "entry_anchor_ts",
    "target_strike_return",
    "target_strike_price",
    "contract_symbol",
    "strike_price",
    "strike_distance",
    "expiry",
    "settlement",
    "entry_premium",
    "entry_premium_source",
    "entry_quote_ts",
    "quote_age_seconds",
    "spread",
    "spread_pct",
    "quote_quality_status",
    "skip_reason",
)


@dataclass(frozen=True)
class ContractEligibilityRules:
    option_type: str = "put"
    require_same_day_expiry: bool = True
    require_pm_settlement: bool = True
    allowed_underlyings: tuple[str, ...] = ("SPXW", "SPY")


@dataclass(frozen=True)
class QuoteQualityRules:
    max_quote_age_seconds: float = 180.0
    max_spread_pct: float = 0.35


@dataclass(frozen=True)
class SessionWindow:
    session_date: date
    is_trading_day: bool
    is_half_day: bool
    market_open: datetime | None
    market_close: datetime | None
    close_reason: str | None = None


@dataclass(frozen=True)
class IntradayStateDataset:
    underlying_symbol: str
    proxy_symbol: str
    session: SessionWindow
    underlying_bars: pd.DataFrame
    state_rows: pd.DataFrame
    option_snapshot: pd.DataFrame
    option_bars: pd.DataFrame
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ZeroDTEIntradayDatasetLoader:
    intraday_store: IntradayStore
    market_tz_name: str = DEFAULT_MARKET_TZ
    options_snapshot_store: Any | None = None
    warehouse: DuckDBWarehouse | None = None

    @property
    def market_tz(self) -> ZoneInfo:
        return ZoneInfo(self.market_tz_name)

    def load_day(
        self,
        session_date: date,
        *,
        underlying_symbol: str = DEFAULT_PROXY_UNDERLYING,
        decision_times: Sequence[str | time | datetime] | None = None,
        option_contract_symbols: Sequence[str] | None = None,
        include_option_snapshot: bool = False,
        include_option_bars: bool = False,
        stock_timeframe: str = "1Min",
        option_timeframe: str = "1Min",
    ) -> IntradayStateDataset:
        symbol = _normalize_symbol(underlying_symbol, default=DEFAULT_PROXY_UNDERLYING)
        session = build_us_equity_session(session_date, market_tz=self.market_tz)
        notes: list[str] = []

        underlying_bars = self.load_underlying_bars(
            session_date,
            underlying_symbol=symbol,
            timeframe=stock_timeframe,
            session=session,
        )
        if underlying_bars.empty:
            notes.append(f"No underlying bars found for {symbol} on {session_date.isoformat()}.")

        state_rows = build_state_rows(
            session=session,
            underlying_bars=underlying_bars,
            market_tz=self.market_tz,
            decision_times=decision_times,
        )

        option_snapshot = pd.DataFrame()
        if include_option_snapshot:
            option_snapshot = self._load_option_snapshot(symbol, session_date, notes)

        option_bars = pd.DataFrame()
        if include_option_bars and option_contract_symbols:
            option_bars = self._load_option_bars(
                session_date,
                option_contract_symbols,
                timeframe=option_timeframe,
                notes=notes,
            )

        return IntradayStateDataset(
            underlying_symbol=symbol,
            proxy_symbol=DEFAULT_PROXY_UNDERLYING,
            session=session,
            underlying_bars=underlying_bars,
            state_rows=state_rows,
            option_snapshot=option_snapshot,
            option_bars=option_bars,
            notes=tuple(notes),
        )

    def load_strike_premium_snapshot(
        self,
        session_date: date,
        *,
        previous_close: float,
        strike_returns: Sequence[float],
        entry_anchor_ts: str | datetime | pd.Timestamp,
        underlying_symbol: str = DEFAULT_PROXY_UNDERLYING,
        include_option_bars: bool = True,
        option_timeframe: str = "1Min",
        eligibility_rules: ContractEligibilityRules | None = None,
        quote_quality_rules: QuoteQualityRules | None = None,
    ) -> pd.DataFrame:
        symbol = _normalize_symbol(underlying_symbol, default=DEFAULT_PROXY_UNDERLYING)
        notes: list[str] = []
        option_snapshot = self._load_option_snapshot(symbol, session_date, notes)
        contracts = extract_option_contract_symbols(option_snapshot)

        option_bars = pd.DataFrame()
        if include_option_bars and contracts:
            option_bars = self._load_option_bars(
                session_date,
                contracts,
                timeframe=option_timeframe,
                notes=notes,
            )

        out = resolve_strike_premium_snapshot(
            option_snapshot=option_snapshot,
            option_bars=option_bars,
            session_date=session_date,
            previous_close=previous_close,
            strike_returns=strike_returns,
            entry_anchor_ts=entry_anchor_ts,
            eligibility_rules=eligibility_rules,
            quote_quality_rules=quote_quality_rules,
        )
        if notes:
            out = out.copy()
            out["notes"] = "; ".join(notes)
        return out

    def load_underlying_bars(
        self,
        session_date: date,
        *,
        underlying_symbol: str = DEFAULT_PROXY_UNDERLYING,
        timeframe: str = "1Min",
        session: SessionWindow | None = None,
    ) -> pd.DataFrame:
        resolved_session = session or build_us_equity_session(session_date, market_tz=self.market_tz)
        if not resolved_session.is_trading_day:
            return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

        raw = self.intraday_store.load_partition(
            _DEFAULT_STOCK_SPEC[0],
            _DEFAULT_STOCK_SPEC[1],
            timeframe,
            underlying_symbol,
            session_date,
        )
        normalized = normalize_intraday_bars(raw, market_tz=self.market_tz)
        if normalized.empty:
            return normalized

        open_ts = pd.Timestamp(resolved_session.market_open)
        close_ts = pd.Timestamp(resolved_session.market_close)
        in_session = (normalized["timestamp_market"] >= open_ts) & (
            normalized["timestamp_market"] <= close_ts
        )
        out = normalized.loc[in_session].copy()
        if out.empty:
            return pd.DataFrame(columns=_UNDERLYING_COLUMNS)
        return out.reset_index(drop=True)

    def _load_option_snapshot(self, underlying_symbol: str, session_date: date, notes: list[str]) -> pd.DataFrame:
        store = self.options_snapshot_store
        if store is None or not hasattr(store, "load_day"):
            notes.append("Option snapshot store unavailable; returning empty snapshot frame.")
            return pd.DataFrame()
        try:
            loaded = store.load_day(underlying_symbol, session_date)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Option snapshot load failed for {underlying_symbol}: {exc}")
            return pd.DataFrame()
        if loaded is None:
            return pd.DataFrame()
        return loaded.copy()

    def _load_option_bars(
        self,
        session_date: date,
        contract_symbols: Sequence[str],
        *,
        timeframe: str,
        notes: list[str],
    ) -> pd.DataFrame:
        normalized_contracts = [_normalize_symbol(contract) for contract in contract_symbols if str(contract).strip()]
        frames: list[pd.DataFrame] = []

        for contract in normalized_contracts:
            part = self.intraday_store.load_partition(
                _DEFAULT_OPTION_SPEC[0],
                _DEFAULT_OPTION_SPEC[1],
                timeframe,
                contract,
                session_date,
            )
            if part is None or part.empty:
                continue
            normalized = normalize_intraday_bars(part, market_tz=self.market_tz)
            if normalized.empty:
                continue
            normalized["contract_symbol"] = contract
            frames.append(normalized)

        if frames:
            return pd.concat(frames, ignore_index=True)

        warehouse = self.warehouse
        if warehouse is None:
            return pd.DataFrame()
        if not _duckdb_table_exists(warehouse, "option_bars"):
            notes.append("DuckDB table missing: option_bars")
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in normalized_contracts)
        if not placeholders:
            return pd.DataFrame()

        start_local = datetime.combine(session_date, time.min, tzinfo=self.market_tz)
        end_local = datetime.combine(session_date + timedelta(days=1), time.min, tzinfo=self.market_tz)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

        try:
            loaded = warehouse.fetch_df(
                f"""
                SELECT
                  contract_symbol,
                  ts AS timestamp,
                  open,
                  high,
                  low,
                  close,
                  volume,
                  vwap,
                  trade_count
                FROM option_bars
                WHERE interval = ?
                  AND contract_symbol IN ({placeholders})
                  AND ts >= ?
                  AND ts < ?
                ORDER BY contract_symbol ASC, ts ASC
                """,
                [timeframe.lower(), *normalized_contracts, start_utc, end_utc],
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"DuckDB option bars query failed: {exc}")
            return pd.DataFrame()

        if loaded is None or loaded.empty:
            return pd.DataFrame()
        if "contract_symbol" not in loaded.columns:
            return normalize_intraday_bars(loaded, market_tz=self.market_tz)

        frames = []
        for contract_symbol, sub in loaded.groupby("contract_symbol", sort=False):
            normalized = normalize_intraday_bars(sub, market_tz=self.market_tz)
            if normalized.empty:
                continue
            normalized["contract_symbol"] = str(contract_symbol)
            frames.append(normalized)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def extract_option_contract_symbols(option_snapshot: pd.DataFrame | None) -> list[str]:
    if option_snapshot is None or option_snapshot.empty:
        return []
    frame = option_snapshot.copy()
    frame.columns = [str(col) for col in frame.columns]
    contract_col = _first_present(frame.columns, "contract_symbol", "contractSymbol", "osi")
    if contract_col is None:
        return []
    out: set[str] = set()
    for raw in frame[contract_col]:
        symbol = str(raw or "").strip().upper()
        if symbol:
            out.add(symbol)
    return sorted(out)


def resolve_strike_premium_snapshot(
    *,
    option_snapshot: pd.DataFrame | None,
    option_bars: pd.DataFrame | None,
    session_date: date,
    previous_close: float,
    strike_returns: Sequence[float],
    entry_anchor_ts: str | datetime | pd.Timestamp,
    eligibility_rules: ContractEligibilityRules | None = None,
    quote_quality_rules: QuoteQualityRules | None = None,
) -> pd.DataFrame:
    if not math.isfinite(float(previous_close)) or float(previous_close) <= 0.0:
        raise ValueError("previous_close must be a positive finite value")

    anchor_utc = _coerce_utc_timestamp(entry_anchor_ts)
    rules = eligibility_rules or ContractEligibilityRules()
    quote_rules = quote_quality_rules or QuoteQualityRules()

    targets = sorted({float(raw) for raw in strike_returns})
    if not targets:
        return pd.DataFrame(columns=_STRIKE_PREMIUM_COLUMNS)

    contracts = _normalize_snapshot_contracts(option_snapshot, anchor_utc=anchor_utc)
    eligible = _apply_contract_eligibility(contracts, session_date=session_date, rules=rules)
    bars = _normalize_option_bars_frame(option_bars)

    rows: list[dict[str, object]] = []
    for target_return in targets:
        target_price = float(previous_close) * (1.0 + target_return)
        base_row: dict[str, object] = {
            "session_date": session_date.isoformat(),
            "entry_anchor_ts": anchor_utc,
            "target_strike_return": target_return,
            "target_strike_price": target_price,
            "contract_symbol": None,
            "strike_price": float("nan"),
            "strike_distance": float("nan"),
            "expiry": None,
            "settlement": None,
            "entry_premium": float("nan"),
            "entry_premium_source": None,
            "entry_quote_ts": pd.NaT,
            "quote_age_seconds": float("nan"),
            "spread": float("nan"),
            "spread_pct": float("nan"),
            "quote_quality_status": QuoteQualityStatus.MISSING.value,
            "skip_reason": SkipReason.NO_ELIGIBLE_CONTRACTS.value,
        }

        selected = _choose_contract_for_target(eligible, target_price=target_price)
        if selected is None:
            rows.append(base_row)
            continue

        premium = _resolve_entry_premium(
            selected=selected,
            option_bars=bars,
            anchor_utc=anchor_utc,
            quote_rules=quote_rules,
        )
        base_row.update(
            {
                "contract_symbol": selected.get("contract_symbol"),
                "strike_price": selected.get("strike"),
                "strike_distance": abs(float(selected.get("strike")) - target_price),
                "expiry": selected.get("expiry"),
                "settlement": selected.get("settlement"),
                "entry_premium": premium["entry_premium"],
                "entry_premium_source": premium["entry_premium_source"],
                "entry_quote_ts": premium["entry_quote_ts"],
                "quote_age_seconds": premium["quote_age_seconds"],
                "spread": premium["spread"],
                "spread_pct": premium["spread_pct"],
                "quote_quality_status": premium["quote_quality_status"],
                "skip_reason": premium["skip_reason"],
            }
        )
        rows.append(base_row)

    return pd.DataFrame(rows, columns=_STRIKE_PREMIUM_COLUMNS)


def _normalize_snapshot_contracts(
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

    option_type_col = _first_present(frame.columns, "option_type", "optionType")
    if option_type_col is None:
        frame["option_type"] = parsed.map(
            lambda item: item.option_type if item is not None else None
        )
    else:
        frame["option_type"] = frame[option_type_col].map(_normalize_option_type)
        fallback = frame["option_type"].isna()
        frame.loc[fallback, "option_type"] = parsed[fallback].map(
            lambda item: item.option_type if item is not None else None
        )

    expiry_col = _first_present(frame.columns, "expiry", "expiration", "expirationDate")
    if expiry_col is None:
        frame["expiry"] = parsed.map(lambda item: item.expiry if item is not None else None)
    else:
        parsed_expiry = pd.to_datetime(frame[expiry_col], errors="coerce").dt.date
        frame["expiry"] = parsed_expiry
        missing_expiry = frame["expiry"].isna()
        frame.loc[missing_expiry, "expiry"] = parsed[missing_expiry].map(
            lambda item: item.expiry if item is not None else None
        )

    strike_col = _first_present(frame.columns, "strike", "strike_price")
    if strike_col is None:
        frame["strike"] = parsed.map(lambda item: item.strike if item is not None else float("nan"))
    else:
        frame["strike"] = pd.to_numeric(frame[strike_col], errors="coerce")
        missing_strike = frame["strike"].isna()
        frame.loc[missing_strike, "strike"] = parsed[missing_strike].map(
            lambda item: item.strike if item is not None else float("nan")
        )

    settlement_col = _first_present(
        frame.columns,
        "settlement",
        "settlement_style",
        "settlementType",
        "settleType",
    )
    if settlement_col is None:
        frame["settlement"] = parsed.map(infer_settlement_style)
    else:
        frame["settlement"] = frame[settlement_col].map(_normalize_settlement)
        missing_settlement = frame["settlement"].isna()
        frame.loc[missing_settlement, "settlement"] = parsed[missing_settlement].map(
            infer_settlement_style
        )

    underlying_col = _first_present(frame.columns, "underlying", "symbol", "root")
    if underlying_col is None:
        frame["underlying_norm"] = parsed.map(
            lambda item: item.underlying_norm if item is not None else None
        )
    else:
        frame["underlying_norm"] = frame[underlying_col].astype(str).str.strip().str.upper()
        missing_underlying = frame["underlying_norm"].isin({"", "NAN", "NONE"})
        frame.loc[missing_underlying, "underlying_norm"] = parsed[missing_underlying].map(
            lambda item: item.underlying_norm if item is not None else None
        )

    quote_ts_col = _first_present(
        frame.columns,
        "quote_timestamp",
        "quote_ts",
        "timestamp",
        "updated_at",
        "lastTradeDate",
    )
    if quote_ts_col is None:
        frame["quote_ts"] = pd.NaT
    else:
        frame["quote_ts"] = pd.to_datetime(frame[quote_ts_col], errors="coerce", utc=True)

    bid_col = _first_present(frame.columns, "bid")
    ask_col = _first_present(frame.columns, "ask")
    frame["bid"] = pd.to_numeric(frame[bid_col], errors="coerce") if bid_col is not None else float("nan")
    frame["ask"] = pd.to_numeric(frame[ask_col], errors="coerce") if ask_col is not None else float("nan")

    if not frame["quote_ts"].isna().all():
        after_anchor = frame["quote_ts"] >= anchor_utc
        frame["quote_priority"] = 1
        frame.loc[after_anchor, "quote_priority"] = 0
        frame.loc[frame["quote_ts"].isna(), "quote_priority"] = 2
        frame["quote_delta_seconds"] = (frame["quote_ts"] - anchor_utc).abs().dt.total_seconds()
        frame["quote_delta_seconds"] = frame["quote_delta_seconds"].fillna(float("inf"))
        frame = frame.sort_values(
            by=[
                "contract_symbol",
                "quote_priority",
                "quote_delta_seconds",
                "quote_ts",
            ],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
    else:
        frame = frame.sort_values(by=["contract_symbol"], kind="mergesort")

    deduped = frame.drop_duplicates(subset=["contract_symbol"], keep="first").copy()
    return deduped.reset_index(drop=True)


def _apply_contract_eligibility(
    contracts: pd.DataFrame,
    *,
    session_date: date,
    rules: ContractEligibilityRules,
) -> pd.DataFrame:
    if contracts is None or contracts.empty:
        return pd.DataFrame()

    eligible = contracts.copy()
    option_type = str(rules.option_type or "").strip().lower()
    if option_type:
        eligible = eligible.loc[
            eligible["option_type"].astype(str).str.lower() == option_type
        ]
    if eligible.empty:
        return eligible

    if rules.require_same_day_expiry:
        eligible = eligible.loc[eligible["expiry"] == session_date]
    if eligible.empty:
        return eligible

    if rules.require_pm_settlement:
        eligible = eligible.loc[eligible["settlement"] == "pm"]
    if eligible.empty:
        return eligible

    if rules.allowed_underlyings:
        allowed = {str(item).strip().upper() for item in rules.allowed_underlyings if str(item).strip()}
        eligible = eligible.loc[eligible["underlying_norm"].astype(str).str.upper().isin(allowed)]

    return eligible.reset_index(drop=True)


def _choose_contract_for_target(eligible: pd.DataFrame, *, target_price: float) -> pd.Series | None:
    if eligible is None or eligible.empty:
        return None
    if "strike" not in eligible.columns:
        return None

    table = eligible.copy()
    table["strike"] = pd.to_numeric(table["strike"], errors="coerce")
    table = table.loc[~table["strike"].isna()].copy()
    if table.empty:
        return None

    table["strike_distance"] = (table["strike"].astype(float) - float(target_price)).abs()
    table["bid_rank"] = -pd.to_numeric(table.get("bid"), errors="coerce").fillna(0.0)
    table = table.sort_values(
        by=["strike_distance", "bid_rank", "contract_symbol"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    return table.iloc[0]


def _resolve_entry_premium(
    *,
    selected: pd.Series,
    option_bars: pd.DataFrame,
    anchor_utc: pd.Timestamp,
    quote_rules: QuoteQualityRules,
) -> dict[str, object]:
    quality = _evaluate_quote_quality(selected, anchor_utc=anchor_utc, rules=quote_rules)
    status = quality["quote_quality_status"]
    premium = quality["quote_bid"]
    source = None
    if (
        _quote_status_is_usable(status)
        and math.isfinite(premium)
        and premium > 0.0
    ):
        source = "quote_bid"
    quote_ts = selected.get("quote_ts")
    age_seconds = quality["quote_age_seconds"]

    if source is None:
        bar_premium, bar_ts, bar_age = _lookup_option_bar_premium(
            option_bars,
            contract_symbol=str(selected.get("contract_symbol") or ""),
            anchor_utc=anchor_utc,
        )
        if math.isfinite(bar_premium) and bar_premium > 0.0:
            premium = bar_premium
            source = "bar_close"
            quote_ts = bar_ts
            age_seconds = bar_age
            if status == QuoteQualityStatus.MISSING.value:
                status = QuoteQualityStatus.UNKNOWN.value

    skip_reason: str | None = None
    if source is None:
        skip_reason = SkipReason.BAD_QUOTE_QUALITY.value
        premium = float("nan")

    return {
        "entry_premium": premium,
        "entry_premium_source": source,
        "entry_quote_ts": quote_ts,
        "quote_age_seconds": age_seconds,
        "spread": quality["spread"],
        "spread_pct": quality["spread_pct"],
        "quote_quality_status": status,
        "skip_reason": skip_reason,
    }


def _quote_status_is_usable(status: object) -> bool:
    return str(status or "").strip().lower() == QuoteQualityStatus.GOOD.value


def _lookup_option_bar_premium(
    option_bars: pd.DataFrame,
    *,
    contract_symbol: str,
    anchor_utc: pd.Timestamp,
) -> tuple[float, pd.Timestamp | pd.NaT, float]:
    if option_bars is None or option_bars.empty or not contract_symbol:
        return float("nan"), pd.NaT, float("nan")

    frame = option_bars.copy()
    frame.columns = [str(col) for col in frame.columns]
    symbol_col = _first_present(frame.columns, "contract_symbol", "contractSymbol")
    if symbol_col is None:
        return float("nan"), pd.NaT, float("nan")

    sub = frame.loc[frame[symbol_col].astype(str).str.upper() == contract_symbol.upper()].copy()
    if sub.empty:
        return float("nan"), pd.NaT, float("nan")

    ts_col = _first_present(sub.columns, "timestamp", "ts", "time")
    if ts_col is None:
        return float("nan"), pd.NaT, float("nan")
    sub["timestamp"] = pd.to_datetime(sub[ts_col], errors="coerce", utc=True)
    sub = sub.loc[~sub["timestamp"].isna()].copy()
    if sub.empty:
        return float("nan"), pd.NaT, float("nan")

    sub["after_anchor"] = sub["timestamp"] >= anchor_utc
    sub["delta_seconds"] = (sub["timestamp"] - anchor_utc).abs().dt.total_seconds()
    sub["priority"] = 1
    sub.loc[sub["after_anchor"], "priority"] = 0
    sub = sub.sort_values(
        by=["priority", "delta_seconds", "timestamp"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    winner = sub.iloc[0]
    price_col = _first_present(sub.columns, "close", "open", "high", "low")
    if price_col is None:
        return float("nan"), pd.NaT, float("nan")
    price = float(pd.to_numeric(winner[price_col], errors="coerce"))
    ts_value = pd.Timestamp(winner["timestamp"])
    age_seconds = float(abs((ts_value - anchor_utc).total_seconds()))
    return price, ts_value, age_seconds


def _normalize_option_bars_frame(option_bars: pd.DataFrame | None) -> pd.DataFrame:
    if option_bars is None or option_bars.empty:
        return pd.DataFrame()
    frame = option_bars.copy()
    frame.columns = [str(col) for col in frame.columns]
    return frame


def _evaluate_quote_quality(
    selected: pd.Series,
    *,
    anchor_utc: pd.Timestamp,
    rules: QuoteQualityRules,
) -> dict[str, float | str]:
    bid = float(pd.to_numeric(selected.get("bid"), errors="coerce"))
    ask = float(pd.to_numeric(selected.get("ask"), errors="coerce"))
    spread = float("nan")
    spread_pct = float("nan")
    age_seconds = float("nan")

    quote_ts = selected.get("quote_ts")
    quote_timestamp: pd.Timestamp | None = None
    if isinstance(quote_ts, pd.Timestamp) and not pd.isna(quote_ts):
        quote_timestamp = quote_ts.tz_convert("UTC") if quote_ts.tzinfo is not None else quote_ts.tz_localize("UTC")
        age_seconds = abs(float((quote_timestamp - anchor_utc).total_seconds()))

    if not math.isfinite(bid) or not math.isfinite(ask):
        status = QuoteQualityStatus.MISSING.value
        return {
            "quote_bid": float("nan"),
            "spread": spread,
            "spread_pct": spread_pct,
            "quote_age_seconds": age_seconds,
            "quote_quality_status": status,
        }

    if bid <= 0.0 or ask <= 0.0:
        status = QuoteQualityStatus.ZERO_BID.value
        return {
            "quote_bid": bid,
            "spread": spread,
            "spread_pct": spread_pct,
            "quote_age_seconds": age_seconds,
            "quote_quality_status": status,
        }

    spread = ask - bid
    mid = (ask + bid) / 2.0
    spread_pct = spread / mid if mid > 0.0 else float("nan")

    if bid > ask:
        status = QuoteQualityStatus.CROSSED.value
    elif math.isfinite(spread_pct) and spread_pct > float(rules.max_spread_pct):
        status = QuoteQualityStatus.WIDE.value
    elif math.isfinite(age_seconds) and age_seconds > float(rules.max_quote_age_seconds):
        status = QuoteQualityStatus.STALE.value
    else:
        status = QuoteQualityStatus.GOOD.value

    return {
        "quote_bid": bid,
        "spread": spread,
        "spread_pct": spread_pct,
        "quote_age_seconds": age_seconds,
        "quote_quality_status": status,
    }


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


def _coerce_utc_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError("entry_anchor_ts must be parseable as datetime")
    return pd.Timestamp(parsed)


def normalize_intraday_bars(df: pd.DataFrame | None, *, market_tz: ZoneInfo) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

    table = df.copy()
    table.columns = [str(col) for col in table.columns]
    ts_col = _first_present(table.columns, "timestamp", "ts", "time")
    if ts_col is None:
        return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

    table["timestamp"] = pd.to_datetime(table[ts_col], errors="coerce", utc=True)
    table = table.loc[~table["timestamp"].isna()].copy()
    if table.empty:
        return pd.DataFrame(columns=_UNDERLYING_COLUMNS)

    table["timestamp_market"] = table["timestamp"].dt.tz_convert(market_tz)

    for col in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
        else:
            table[col] = pd.NA

    table = table.sort_values("timestamp", kind="mergesort")
    table = table.drop_duplicates(subset=["timestamp"], keep="last")

    out = table.loc[:, list(_UNDERLYING_COLUMNS)].copy()
    return out.reset_index(drop=True)


def build_state_rows(
    *,
    session: SessionWindow,
    underlying_bars: pd.DataFrame,
    market_tz: ZoneInfo,
    decision_times: Sequence[str | time | datetime] | None = None,
) -> pd.DataFrame:
    if decision_times is None:
        decisions = [pd.Timestamp(ts) for ts in underlying_bars.get("timestamp_market", pd.Series(dtype=object)).tolist()]
    else:
        decisions = [
            _coerce_decision_timestamp(item, session_date=session.session_date, market_tz=market_tz)
            for item in decision_times
        ]

    rows: list[dict[str, object]] = []
    for decision_ts_market in decisions:
        rows.append(
            _build_state_row(
                session=session,
                underlying_bars=underlying_bars,
                decision_ts_market=decision_ts_market,
            )
        )

    if not rows:
        return pd.DataFrame(columns=_STATE_COLUMNS)
    return pd.DataFrame(rows, columns=_STATE_COLUMNS)


def _build_state_row(
    *,
    session: SessionWindow,
    underlying_bars: pd.DataFrame,
    decision_ts_market: pd.Timestamp,
) -> dict[str, object]:
    decision_utc = decision_ts_market.tz_convert("UTC")
    base: dict[str, object] = {
        "session_date": session.session_date.isoformat(),
        "decision_ts": decision_utc,
        "decision_ts_market": decision_ts_market,
        "bar_ts": pd.NaT,
        "bar_ts_market": pd.NaT,
        "open": pd.NA,
        "high": pd.NA,
        "low": pd.NA,
        "close": pd.NA,
        "volume": pd.NA,
        "vwap": pd.NA,
        "trade_count": pd.NA,
        "bar_age_seconds": pd.NA,
        "status": "no_underlying_data",
        "is_half_day": session.is_half_day,
    }

    if not session.is_trading_day or session.market_open is None or session.market_close is None:
        base["status"] = "market_closed"
        return base

    open_ts = pd.Timestamp(session.market_open)
    close_ts = pd.Timestamp(session.market_close)
    if decision_ts_market < open_ts or decision_ts_market > close_ts:
        base["status"] = "outside_session"
        return base

    if underlying_bars.empty:
        return base

    eligible = underlying_bars.loc[underlying_bars["timestamp_market"] <= decision_ts_market]
    if eligible.empty:
        base["status"] = "no_prior_bar"
        return base

    row = eligible.iloc[-1]
    bar_market_ts = pd.Timestamp(row["timestamp_market"])
    bar_utc_ts = pd.Timestamp(row["timestamp"])
    base.update(
        {
            "bar_ts": bar_utc_ts,
            "bar_ts_market": bar_market_ts,
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
            "vwap": row.get("vwap"),
            "trade_count": row.get("trade_count"),
            "bar_age_seconds": float((decision_ts_market - bar_market_ts).total_seconds()),
            "status": "ok",
        }
    )
    return base


def build_us_equity_session(session_date: date, *, market_tz: ZoneInfo) -> SessionWindow:
    if session_date.weekday() >= 5:
        return SessionWindow(
            session_date=session_date,
            is_trading_day=False,
            is_half_day=False,
            market_open=None,
            market_close=None,
            close_reason="weekend",
        )

    holidays = _us_equity_holidays(session_date.year)
    if session_date in holidays:
        return SessionWindow(
            session_date=session_date,
            is_trading_day=False,
            is_half_day=False,
            market_open=None,
            market_close=None,
            close_reason="holiday",
        )

    half_days = _us_equity_half_days(session_date.year)
    is_half_day = session_date in half_days
    close_time = time(13, 0) if is_half_day else time(16, 0)

    return SessionWindow(
        session_date=session_date,
        is_trading_day=True,
        is_half_day=is_half_day,
        market_open=datetime.combine(session_date, time(9, 30), tzinfo=market_tz),
        market_close=datetime.combine(session_date, close_time, tzinfo=market_tz),
        close_reason="half_day" if is_half_day else "regular",
    )


__all__ = ["DEFAULT_MARKET_TZ", "DEFAULT_PROXY_UNDERLYING", "IntradayStateDataset", "SessionWindow", "ZeroDTEIntradayDatasetLoader", "build_state_rows", "build_us_equity_session", "normalize_intraday_bars"]
