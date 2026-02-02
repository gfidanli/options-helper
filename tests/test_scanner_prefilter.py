from __future__ import annotations

from options_helper.data.scanner import prefilter_symbols


def test_prefilter_default_drops_special_and_hyphen_suffixes() -> None:
    symbols = ["AAA", "$AAPL", "BRK-B", "AAM-WT", "TEST/ABC", "XYZ^"]
    filtered, dropped = prefilter_symbols(symbols, mode="default", exclude={"AAA"})
    assert "AAA" not in filtered
    assert "BRK-B" in filtered
    assert "$AAPL" not in filtered
    assert "AAM-WT" not in filtered
    assert dropped["special_char"] >= 2
    assert dropped["hyphen_suffix"] >= 1


def test_prefilter_respects_scanned_and_none_mode() -> None:
    symbols = ["AAA", "$AAPL", "BRK-B"]
    filtered, dropped = prefilter_symbols(symbols, mode="none", scanned={"AAA"})
    assert "AAA" not in filtered
    assert "$AAPL" in filtered
    assert "BRK-B" in filtered
    assert dropped.get("scanned", 0) == 1
