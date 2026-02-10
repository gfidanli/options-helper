from datetime import date

from options_helper.analysis.osi import ParsedContract, format_osi_compact, parse_contract_symbol


def test_format_osi_compact_has_no_padding_spaces() -> None:
    parsed = ParsedContract(
        underlying="AAPL",
        underlying_norm="AAPL",
        expiry=date(2024, 1, 19),
        option_type="call",
        strike=150.0,
    )
    assert format_osi_compact(parsed) == "AAPL240119C00150000"


def test_format_osi_compact_roundtrip_parse() -> None:
    parsed = ParsedContract(
        underlying="SPY",
        underlying_norm="SPY",
        expiry=date(2024, 1, 19),
        option_type="put",
        strike=430.0,
    )
    compact = format_osi_compact(parsed)

    assert compact == "SPY240119P00430000"
    assert " " not in compact

    parsed_roundtrip = parse_contract_symbol(compact)
    assert parsed_roundtrip is not None
    assert format_osi_compact(parsed_roundtrip) == compact
