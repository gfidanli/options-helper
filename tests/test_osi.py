from datetime import date

import pytest

from options_helper.analysis.osi import ParsedContract, format_osi, normalize_underlying, parse_contract_symbol


def test_parse_and_format_round_trip() -> None:
    raw = "AAPL260116C00150000"
    parsed = parse_contract_symbol(raw)
    assert parsed is not None
    assert parsed.underlying == "AAPL"
    assert parsed.underlying_norm == "AAPL"
    assert parsed.option_type == "call"
    assert parsed.expiry == date(2026, 1, 16)
    assert parsed.strike == pytest.approx(150.0)

    canonical = format_osi(parsed)
    assert canonical == "AAPL  260116C00150000"

    parsed_again = parse_contract_symbol(canonical)
    assert parsed_again is not None
    assert format_osi(parsed_again) == canonical


def test_parse_normalizes_dotted_underlying() -> None:
    raw = "BRK.B260116P00300000"
    parsed = parse_contract_symbol(raw)
    assert parsed is not None
    assert parsed.underlying == "BRK.B"
    assert parsed.underlying_norm == "BRK-B"
    assert parsed.option_type == "put"
    assert format_osi(parsed) == "BRK-B 260116P00300000"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "AAPL260116C00150",
        "AAPL260132C00150000",
        "AAPL260116X00150000",
    ],
)
def test_parse_invalid_contract_symbol_returns_none(raw) -> None:
    assert parse_contract_symbol(raw) is None


def test_format_osi_strike_rounding() -> None:
    parsed = ParsedContract(
        underlying="TEST",
        underlying_norm="TEST",
        expiry=date(2026, 1, 16),
        option_type="call",
        strike=150.125,
    )
    assert format_osi(parsed) == "TEST  260116C00150125"


def test_normalize_underlying() -> None:
    assert normalize_underlying(" brk.b ") == "BRK-B"
    assert normalize_underlying("BRK-B") == "BRK-B"
    assert normalize_underlying("aapl") == "AAPL"
