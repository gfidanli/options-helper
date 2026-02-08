from __future__ import annotations

from datetime import datetime, timezone

from options_helper.analysis.strategy_modeling import (
    StrategySegmentationConfig,
    StrategySegmentationResult,
    aggregate_strategy_segmentation,
)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _trade(
    *,
    trade_id: str,
    event_id: str,
    symbol: str,
    direction: str,
    realized_r: float | None,
    status: str = "closed",
    reject_code: str | None = None,
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "event_id": event_id,
        "strategy": "sfp",
        "symbol": symbol,
        "direction": direction,
        "signal_ts": _ts("2026-01-01T21:00:00Z"),
        "signal_confirmed_ts": _ts("2026-01-01T21:00:00Z"),
        "entry_ts": _ts("2026-01-02T14:30:00Z"),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 101.0,
        "exit_ts": _ts("2026-01-02T15:00:00Z") if status == "closed" else None,
        "exit_price": 100.0 if status == "closed" else None,
        "status": status,
        "exit_reason": "target_hit" if status == "closed" else None,
        "reject_code": reject_code,
        "initial_risk": 1.0,
        "realized_r": realized_r,
        "mae_r": realized_r,
        "mfe_r": realized_r,
        "holding_bars": 1,
        "gap_fill_applied": False,
    }


def _find_segment_row(
    result: StrategySegmentationResult,
    *,
    dimension: str,
    value: str,
):
    for row in result.segments:
        if row.segment_dimension == dimension and row.segment_value == value:
            return row
    raise AssertionError(f"missing segment row: {dimension}={value}")


def test_aggregate_strategy_segmentation_reconciles_slice_counts_to_base_trades() -> None:
    trades = [
        _trade(
            trade_id="tr-1",
            event_id="evt-1",
            symbol="SPY",
            direction="long",
            realized_r=1.0,
        ),
        _trade(
            trade_id="tr-2",
            event_id="evt-2",
            symbol="SPY",
            direction="short",
            realized_r=-1.0,
        ),
        _trade(
            trade_id="tr-3",
            event_id="evt-3",
            symbol="QQQ",
            direction="long",
            realized_r=0.5,
        ),
        _trade(
            trade_id="tr-4",
            event_id="evt-4",
            symbol="QQQ",
            direction="long",
            realized_r=None,
            status="rejected",
            reject_code="invalid_signal",
        ),
    ]
    segment_context = [
        {
            "event_id": "evt-1",
            "ticker": "SPY",
            "extension_bucket": "high",
            "rsi_regime": "overbought",
            "rsi_divergence": "bearish",
            "volatility_regime": "high",
            "bars_since_swing_bucket": "0-2",
        },
        {
            "event_id": "evt-2",
            "ticker": "SPY",
            "extension_bucket": "low",
            "rsi_regime": "oversold",
            "rsi_divergence": "bullish",
            "volatility_regime": "low",
            "bars_since_swing_bucket": "3-9",
        },
    ]

    result = aggregate_strategy_segmentation(
        trades,
        segment_context=segment_context,
        config=StrategySegmentationConfig(min_sample_threshold=2),
    )

    assert result.base_trade_count == 3
    assert result.overall.trade_count == 3
    assert result.overall.sample_size == 3

    for row in result.reconciliation:
        assert row.base_trade_count == 3
        assert row.slice_trade_count == 3
        assert row.is_reconciled is True
    assert result.all_dimensions_reconciled is True

    extension_high = _find_segment_row(result, dimension="extension_bucket", value="high")
    extension_low = _find_segment_row(result, dimension="extension_bucket", value="low")
    extension_unknown = _find_segment_row(result, dimension="extension_bucket", value="unknown")
    assert extension_high.sample_size == 1
    assert extension_low.sample_size == 1
    assert extension_unknown.sample_size == 1

    rsi_unknown = _find_segment_row(result, dimension="rsi_regime", value="unknown")
    assert rsi_unknown.sample_size == 1


def test_aggregate_strategy_segmentation_applies_reliability_gating_consistently() -> None:
    trades = [
        _trade(
            trade_id="tr-1",
            event_id="evt-1",
            symbol="SPY",
            direction="long",
            realized_r=1.0,
        ),
        _trade(
            trade_id="tr-2",
            event_id="evt-2",
            symbol="SPY",
            direction="short",
            realized_r=0.5,
        ),
        _trade(
            trade_id="tr-3",
            event_id="evt-3",
            symbol="SPY",
            direction="long",
            realized_r=-1.0,
        ),
        _trade(
            trade_id="tr-4",
            event_id="evt-4",
            symbol="QQQ",
            direction="long",
            realized_r=-0.5,
        ),
    ]

    result = aggregate_strategy_segmentation(
        trades,
        config=StrategySegmentationConfig(
            min_sample_threshold=3,
            include_confidence_intervals=True,
        ),
    )

    spy = _find_segment_row(result, dimension="symbol", value="SPY")
    qqq = _find_segment_row(result, dimension="symbol", value="QQQ")

    assert result.overall.sample_size == 4
    assert result.overall.min_sample_threshold == 3
    assert result.overall.is_reliable is True

    assert spy.sample_size == 3
    assert spy.min_sample_threshold == 3
    assert spy.is_reliable is True

    assert qqq.sample_size == 1
    assert qqq.min_sample_threshold == 3
    assert qqq.is_reliable is False

    assert result.overall.confidence_interval_low is not None
    assert result.overall.confidence_interval_high is not None
    assert result.overall.confidence_interval_label is not None

    without_ci = aggregate_strategy_segmentation(
        trades,
        config=StrategySegmentationConfig(
            min_sample_threshold=3,
            include_confidence_intervals=False,
        ),
    )
    for row in (without_ci.overall, *without_ci.segments):
        assert row.confidence_interval_low is None
        assert row.confidence_interval_high is None
        assert row.confidence_interval_label is None
        assert row.min_sample_threshold == 3

