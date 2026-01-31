from __future__ import annotations

from options_helper.technicals_backtesting.indicators.provider_base import IndicatorProvider
from options_helper.technicals_backtesting.indicators.provider_ta import TaIndicatorProvider


def get_provider(name: str) -> IndicatorProvider:
    provider = name.lower().strip()
    if provider == "ta":
        return TaIndicatorProvider()
    raise ValueError(f"Unsupported indicator provider: {name}")

