from __future__ import annotations

from options_helper.data.alpaca_symbols import to_alpaca_symbol, to_repo_symbol


def test_to_alpaca_symbol_maps_class_shares() -> None:
    assert to_alpaca_symbol("BRK-B") == "BRK.B"
    assert to_alpaca_symbol("brk-b") == "BRK.B"
    assert to_alpaca_symbol("BRK.B") == "BRK.B"


def test_to_alpaca_symbol_passes_regular_symbols() -> None:
    assert to_alpaca_symbol("SPY") == "SPY"
    assert to_alpaca_symbol(" spy ") == "SPY"


def test_to_repo_symbol_maps_class_shares() -> None:
    assert to_repo_symbol("BRK.B") == "BRK-B"
    assert to_repo_symbol("brk.b") == "BRK-B"
    assert to_repo_symbol("BRK-B") == "BRK-B"


def test_to_repo_symbol_passes_regular_symbols() -> None:
    assert to_repo_symbol("SPY") == "SPY"
    assert to_repo_symbol(" spy ") == "SPY"
