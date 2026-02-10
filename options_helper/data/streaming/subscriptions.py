from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from options_helper.analysis.osi import ParsedContract, format_osi_compact, normalize_underlying
from options_helper.data.alpaca_symbols import to_alpaca_symbol
from options_helper.models import MultiLegPosition, OptionType, Portfolio, Position


@dataclass(frozen=True)
class SubscriptionPlan:
    stocks: list[str] = field(default_factory=list)
    option_contracts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False
    truncated_count: int = 0


def _append_unique(values: list[str], seen: set[str], value: str | None) -> None:
    token = (value or "").strip()
    if not token or token in seen:
        return
    seen.add(token)
    values.append(token)


def _to_alpaca_option_contract(
    *,
    symbol: str,
    expiry: date,
    option_type: OptionType,
    strike: float,
) -> str:
    parsed = ParsedContract(
        underlying=symbol,
        underlying_norm=normalize_underlying(symbol),
        expiry=expiry,
        option_type=option_type,
        strike=float(strike),
    )
    return to_alpaca_symbol(format_osi_compact(parsed))


def build_subscription_plan(
    portfolio: Portfolio,
    *,
    stream_stocks: bool,
    stream_options: bool,
    max_option_contracts: int,
) -> SubscriptionPlan:
    stocks: list[str] = []
    option_contracts: list[str] = []
    warnings: list[str] = []

    stock_seen: set[str] = set()
    option_seen: set[str] = set()

    if stream_stocks:
        for position in portfolio.positions:
            _append_unique(stocks, stock_seen, to_alpaca_symbol(position.symbol))

    if stream_options:
        for position in portfolio.positions:
            if isinstance(position, Position):
                try:
                    contract = _to_alpaca_option_contract(
                        symbol=position.symbol,
                        expiry=position.expiry,
                        option_type=position.option_type,
                        strike=position.strike,
                    )
                except ValueError as exc:
                    warnings.append(f"Skipped option position {position.id}: {exc}")
                    continue
                _append_unique(option_contracts, option_seen, contract)
                continue

            if isinstance(position, MultiLegPosition):
                for leg_index, leg in enumerate(position.legs, start=1):
                    try:
                        contract = _to_alpaca_option_contract(
                            symbol=position.symbol,
                            expiry=leg.expiry,
                            option_type=leg.option_type,
                            strike=leg.strike,
                        )
                    except ValueError as exc:
                        warnings.append(
                            f"Skipped multi-leg position {position.id} leg {leg_index}: {exc}"
                        )
                        continue
                    _append_unique(option_contracts, option_seen, contract)

    truncated = False
    truncated_count = 0
    max_contracts = int(max_option_contracts)

    if stream_options and max_contracts < 0:
        warnings.append(
            f"max_option_contracts={max_option_contracts} is negative; treating as 0."
        )
        max_contracts = 0

    if stream_options and len(option_contracts) > max_contracts:
        original_count = len(option_contracts)
        truncated_count = original_count - max_contracts
        option_contracts = option_contracts[:max_contracts]
        truncated = True
        warnings.append(
            "Option subscriptions truncated to "
            f"max_option_contracts={max_contracts}; "
            f"dropped {truncated_count} contract(s) from {original_count}."
        )

    return SubscriptionPlan(
        stocks=stocks,
        option_contracts=option_contracts,
        warnings=warnings,
        truncated=truncated,
        truncated_count=truncated_count,
    )


__all__ = ["SubscriptionPlan", "build_subscription_plan"]
