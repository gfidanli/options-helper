from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Protocol

if TYPE_CHECKING:
    import pandas as pd


class OptionBarsStoreError(RuntimeError):
    pass


class OptionBarsStore(Protocol):
    def upsert_bars(
        self,
        df: pd.DataFrame,
        *,
        interval: str = "1d",
        provider: str,
        updated_at: datetime | None = None,
    ) -> None:
        ...

    def mark_meta_success(
        self,
        contract_symbols: Iterable[str],
        *,
        interval: str = "1d",
        provider: str,
        status: str = "ok",
        rows: int | dict[str, int] | None = None,
        start_ts: datetime | dict[str, datetime] | None = None,
        end_ts: datetime | dict[str, datetime] | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        ...

    def mark_meta_error(
        self,
        contract_symbols: Iterable[str],
        *,
        interval: str = "1d",
        provider: str,
        error: str | Exception,
        status: str = "error",
        updated_at: datetime | None = None,
    ) -> None:
        ...

    def coverage(
        self,
        contract_symbol: str,
        *,
        interval: str = "1d",
        provider: str,
    ) -> dict[str, Any] | None:
        ...
