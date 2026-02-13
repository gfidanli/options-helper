from __future__ import annotations

import sys
import types
from typing import Any

from options_helper.data import alpaca_client_legacy as _legacy

# Common direct imports used across commands/providers/streaming/tests.
AlpacaClient = _legacy.AlpacaClient
contracts_to_df = _legacy.contracts_to_df
option_chain_to_rows = _legacy.option_chain_to_rows
_load_market_tz = _legacy._load_market_tz
_maybe_load_alpaca_env = _legacy._maybe_load_alpaca_env


def __getattr__(name: str) -> Any:
    return getattr(_legacy, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_legacy)))


class _LegacyPassthroughModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        super().__setattr__(name, value)
        if name in {
            "__class__",
            "__dict__",
            "__doc__",
            "__name__",
            "__package__",
            "__loader__",
            "__spec__",
            "__file__",
            "__cached__",
            "__builtins__",
        }:
            return
        if hasattr(_legacy, name):
            setattr(_legacy, name, value)


sys.modules[__name__].__class__ = _LegacyPassthroughModule


__all__ = [
    "AlpacaClient",
    "contracts_to_df",
    "option_chain_to_rows",
    "_load_market_tz",
    "_maybe_load_alpaca_env",
]
