from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from options_helper.data.candles import CandleStore, HistoryFetcher
    from options_helper.data.derived import DerivedStore
    from options_helper.data.earnings import EarningsStore
    from options_helper.data.journal import JournalStore
    from options_helper.data.option_bars import OptionBarsStore
    from options_helper.data.option_contracts import OptionContractsStore
    from options_helper.data.options_snapshots import OptionsSnapshotStore
    from options_helper.data.providers.base import MarketDataProvider


def build_provider(name: str | None = None) -> MarketDataProvider:
    from options_helper.data.providers import get_provider

    return get_provider(name)


def build_candle_store(
    cache_dir: Path,
    *,
    provider: MarketDataProvider | None = None,
    fetcher: HistoryFetcher | None = None,
    auto_adjust: bool | None = None,
    back_adjust: bool | None = None,
) -> CandleStore:
    from options_helper.data.store_factory import get_candle_store

    kwargs: dict[str, object] = {}
    if provider is not None:
        kwargs["provider"] = provider
    if fetcher is not None:
        kwargs["fetcher"] = fetcher
    if auto_adjust is not None:
        kwargs["auto_adjust"] = bool(auto_adjust)
    if back_adjust is not None:
        kwargs["back_adjust"] = bool(back_adjust)
    return get_candle_store(cache_dir, **kwargs)


def build_snapshot_store(cache_dir: Path) -> OptionsSnapshotStore:
    from options_helper.data.store_factory import get_options_snapshot_store

    return get_options_snapshot_store(cache_dir)


def build_derived_store(derived_dir: Path) -> DerivedStore:
    from options_helper.data.store_factory import get_derived_store

    return get_derived_store(derived_dir)


def build_earnings_store(cache_dir: Path) -> EarningsStore:
    from options_helper.data.earnings import EarningsStore

    return EarningsStore(cache_dir)


def build_journal_store(journal_dir: Path) -> JournalStore:
    from options_helper.data.store_factory import get_journal_store

    return get_journal_store(journal_dir)


def build_option_contracts_store(contracts_dir: Path) -> OptionContractsStore:
    from options_helper.data.store_factory import get_option_contracts_store

    return get_option_contracts_store(contracts_dir)


def build_option_bars_store(bars_dir: Path) -> OptionBarsStore:
    from options_helper.data.store_factory import get_option_bars_store

    return get_option_bars_store(bars_dir)
