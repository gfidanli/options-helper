from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.quality_checks import (
    QualityCheckResult,
    evaluate_candle_checks_for_symbol,
    evaluate_derived_duplicate_guard_from_frame,
    evaluate_flow_pk_null_guard_from_frame,
    evaluate_options_bars_checks_from_frame,
    evaluate_snapshot_parseable_contract_symbol_check,
    persist_quality_checks,
    run_candle_quality_checks,
)


def test_candle_checks_detect_duplicates_out_of_order_negative_and_gaps() -> None:
    history = pd.DataFrame(
        {
            "Open": [10.0, -9.0, 8.0],
            "High": [10.5, -8.0, 8.5],
            "Low": [9.5, -9.5, 7.5],
            "Close": [10.1, -8.7, 8.2],
        },
        index=pd.to_datetime(["2026-01-08", "2026-01-06", "2026-01-06"]),
    )

    checks = evaluate_candle_checks_for_symbol("aaa", history)
    by_name = {check.check_name: check for check in checks}

    assert by_name["candles_unique_symbol_date"].status == "fail"
    assert by_name["candles_unique_symbol_date"].metrics["duplicate_date_rows"] == 2

    assert by_name["candles_monotonic_date"].status == "fail"
    assert by_name["candles_monotonic_date"].metrics["out_of_order_pairs"] == 1

    assert by_name["candles_no_negative_prices"].status == "fail"
    assert by_name["candles_no_negative_prices"].metrics["negative_rows"] == 1

    assert by_name["candles_gap_days_last_30"].status == "fail"
    assert by_name["candles_gap_days_last_30"].metrics["gap_days"] > 0

    skipped = run_candle_quality_checks(candle_store=None, symbols=[], skip_reason="no_symbols")
    assert len(skipped) == 4
    assert {item.status for item in skipped} == {"skip"}


def test_options_snapshot_flow_and_derived_checks_detect_failures() -> None:
    bars_frame = pd.DataFrame(
        [
            {
                "contract_symbol": "AAA260117C00100000",
                "interval": "1d",
                "provider": "alpaca",
                "ts": "2026-01-03",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
            },
            {
                "contract_symbol": "AAA260117C00100000",
                "interval": "1d",
                "provider": "alpaca",
                "ts": "2026-01-02",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
            },
            {
                "contract_symbol": "AAA260117C00100000",
                "interval": "1d",
                "provider": "alpaca",
                "ts": "2026-01-02",
                "open": -0.5,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
            },
        ]
    )
    bars_checks = evaluate_options_bars_checks_from_frame(bars_frame)
    bars_by_name = {check.check_name: check for check in bars_checks}

    assert bars_by_name["options_bars_monotonic_ts"].status == "fail"
    assert bars_by_name["options_bars_no_negative_prices"].status == "fail"
    assert bars_by_name["options_bars_duplicate_pk"].status == "fail"

    snapshot_check = evaluate_snapshot_parseable_contract_symbol_check(
        "AAA",
        date(2026, 1, 8),
        pd.DataFrame(
            {
                "contractSymbol": ["AAA260117C00100000", "BAD_SYMBOL"],
                "openInterest": [10, 5],
            }
        ),
    )
    assert snapshot_check.status == "fail"
    assert snapshot_check.metrics["unparseable_contracts"] == 1

    flow_check = evaluate_flow_pk_null_guard_from_frame(
        pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "from_date": "2026-01-01",
                    "to_date": "2026-01-02",
                    "window_size": 1,
                    "group_by": "contract",
                    "row_key": "AAA|2026-01-02",
                },
                {
                    "symbol": "AAA",
                    "from_date": "2026-01-01",
                    "to_date": "2026-01-02",
                    "window_size": 1,
                    "group_by": "contract",
                    "row_key": None,
                },
            ]
        )
    )
    assert flow_check.status == "fail"
    assert flow_check.metrics["null_pk_rows"] == 1

    derived_check = evaluate_derived_duplicate_guard_from_frame(
        "AAA",
        pd.DataFrame(
            {
                "date": ["2026-01-08", "2026-01-08"],
                "spot": [100.0, 101.0],
            }
        ),
    )
    assert derived_check.status == "fail"
    assert derived_check.metrics["duplicate_rows"] == 2


class _CaptureRunLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def log_check(self, **kwargs: object) -> str:
        self.rows.append(dict(kwargs))
        return str(len(self.rows))


def test_persist_quality_checks_maps_scope_to_partition_key() -> None:
    logger = _CaptureRunLogger()
    checks = [
        QualityCheckResult(
            asset_key="candles_daily",
            check_name="candles_no_negative_prices",
            severity="error",
            status="fail",
            scope_key="AAA",
            metrics={"negative_rows": 1},
            message="negative price",
        )
    ]

    persist_quality_checks(run_logger=logger, checks=checks)

    assert len(logger.rows) == 1
    row = logger.rows[0]
    assert row["asset_key"] == "candles_daily"
    assert row["check_name"] == "candles_no_negative_prices"
    assert row["partition_key"] == "AAA"
    assert row["status"] == "fail"
