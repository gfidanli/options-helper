from __future__ import annotations

from datetime import date, datetime, timezone
import json
import time
import tracemalloc
from types import SimpleNamespace

import pandas as pd

from options_helper.analysis.strategy_signals import build_strategy_signal_events
from options_helper.data.strategy_modeling_artifacts import write_strategy_modeling_artifacts

_SYMBOL_COUNT = 300
_TRADES_PER_SYMBOL = 12
_EXPECTED_TRADE_COUNT = _SYMBOL_COUNT * _TRADES_PER_SYMBOL
_RUNTIME_THRESHOLD_SECONDS = 1.5
_PEAK_MEMORY_THRESHOLD_MIB = 32.0
_GENERATED_AT = datetime(2026, 2, 8, 12, 30, tzinfo=timezone.utc)
_SIGNAL_SYMBOL_COUNT = 80
_SIGNAL_BAR_COUNT = 420
_SIGNAL_RUNTIME_THRESHOLD_SECONDS = 4.0


def _build_universe_scale_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    symbols = tuple(f"SYM{idx:03d}" for idx in range(_SYMBOL_COUNT))

    trade_rows: list[dict[str, object]] = []
    event_ids: list[str] = []
    accepted_trade_ids: list[str] = []

    for symbol_idx, symbol in enumerate(symbols):
        for trade_idx in range(_TRADES_PER_SYMBOL):
            trade_id = f"{symbol}-{trade_idx:02d}"
            event_id = f"evt-{trade_id}"
            event_ids.append(event_id)
            accepted_trade_ids.append(trade_id)

            realized_r = round(((trade_idx % 9) - 4) * 0.3, 4)
            gap_fill_applied = trade_idx % 6 == 0
            exit_reason = "stop_gap" if gap_fill_applied else "time_stop"

            trade_rows.append(
                {
                    "trade_id": trade_id,
                    "event_id": event_id,
                    "symbol": symbol,
                    "direction": "long" if trade_idx % 2 == 0 else "short",
                    "status": "closed",
                    "signal_confirmed_ts": "2026-01-05T20:55:00+00:00",
                    "entry_ts": "2026-01-06T14:30:00+00:00",
                    "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
                    "entry_price": 100.0 + symbol_idx,
                    "stop_price": 98.0 + symbol_idx,
                    "target_price": 102.0 + symbol_idx,
                    "exit_ts": "2026-01-06T14:35:00+00:00",
                    "exit_price": 100.0 + symbol_idx + realized_r,
                    "exit_reason": exit_reason,
                    "initial_risk": 2.0,
                    "realized_r": realized_r,
                    "gap_fill_applied": gap_fill_applied,
                    "reject_code": None,
                }
            )

    request = SimpleNamespace(
        strategy="sfp",
        symbols=symbols,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        intraday_timeframe="5Min",
        intraday_source="alpaca",
        signal_confirmation_lag_bars=2,
        policy={
            "require_intraday_bars": True,
            "risk_per_trade_pct": 1.0,
            "gap_fill_policy": "fill_at_open",
            "sizing_rule": "risk_pct_of_equity",
            "entry_ts_anchor_policy": "first_tradable_bar_open_after_signal_confirmed_ts",
        },
    )

    run_result = SimpleNamespace(
        strategy="sfp",
        requested_symbols=symbols,
        modeled_symbols=symbols,
        signal_events=tuple(event_ids),
        accepted_trade_ids=tuple(accepted_trade_ids),
        skipped_trade_ids=(),
        intraday_preflight=SimpleNamespace(
            require_intraday_bars=True,
            blocked_symbols=[],
            coverage_by_symbol={},
            notes=[],
        ),
        portfolio_metrics={
            "starting_capital": 100_000.0,
            "ending_capital": 108_200.0,
            "total_return_pct": 8.2,
            "trade_count": _EXPECTED_TRADE_COUNT,
            "avg_realized_r": 0.08,
        },
        target_hit_rates=tuple(
            {
                "target_label": f"{tenths / 10:.1f}R",
                "target_r": tenths / 10,
                "trade_count": _EXPECTED_TRADE_COUNT,
                "hit_count": int(_EXPECTED_TRADE_COUNT * 0.4),
                "hit_rate": 40.0,
                "avg_bars_to_hit": 3.0,
                "median_bars_to_hit": 3.0,
                "expectancy_r": 0.08,
            }
            for tenths in range(10, 21)
        ),
        segment_records=tuple(
            {
                "segment_dimension": "symbol",
                "segment_value": symbol,
                "trade_count": _TRADES_PER_SYMBOL,
                "win_rate": 55.0,
                "avg_realized_r": 0.08,
                "expectancy_r": 0.08,
                "profit_factor": 1.2,
                "sharpe_ratio": 0.9,
                "max_drawdown_pct": -4.2,
            }
            for symbol in symbols
        ),
        trade_simulations=tuple(trade_rows),
    )
    return request, run_result


def _build_signal_generation_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=_SIGNAL_BAR_COUNT, freq="B")
    close = [
        100.0 + (((bar_idx % 60) - 30) * 0.25) + ((bar_idx // 60) * 0.05)
        for bar_idx in range(_SIGNAL_BAR_COUNT)
    ]
    open_ = [value + (((bar_idx % 3) - 1) * 0.05) for bar_idx, value in enumerate(close)]
    high = [max(o, c) + 0.25 for o, c in zip(open_, close, strict=True)]
    low = [min(o, c) - 0.25 for o, c in zip(open_, close, strict=True)]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_strategy_modeling_universe_scale_smoke_runtime_and_memory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    request, run_result = _build_universe_scale_fixture()

    # Warm-up pass to reduce one-time import/filesystem noise before measurement.
    write_strategy_modeling_artifacts(
        out_dir=tmp_path / "warmup",
        strategy="sfp",
        request=request,
        run_result=run_result,
        generated_at=_GENERATED_AT,
    )

    tracemalloc.start()
    start = time.perf_counter()
    artifact_paths = write_strategy_modeling_artifacts(
        out_dir=tmp_path / "benchmark",
        strategy="sfp",
        request=request,
        run_result=run_result,
        generated_at=_GENERATED_AT,
    )
    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mib = peak_bytes / (1024 * 1024)

    assert elapsed <= _RUNTIME_THRESHOLD_SECONDS, (
        f"Universe-scale artifact write exceeded runtime gate: "
        f"{elapsed:.3f}s > {_RUNTIME_THRESHOLD_SECONDS:.2f}s"
    )
    assert peak_mib <= _PEAK_MEMORY_THRESHOLD_MIB, (
        f"Universe-scale artifact write exceeded memory gate: "
        f"{peak_mib:.2f} MiB > {_PEAK_MEMORY_THRESHOLD_MIB:.2f} MiB"
    )

    payload = json.loads(artifact_paths.summary_json.read_text(encoding="utf-8"))
    required_top_level_keys = {
        "summary",
        "policy_metadata",
        "filter_metadata",
        "filter_summary",
        "directional_metrics",
        "trade_log",
    }
    assert required_top_level_keys.issubset(payload.keys())

    summary = payload.get("summary")
    assert isinstance(summary, dict)
    required_summary_keys = {
        "trade_count",
        "accepted_trade_count",
        "losses_below_minus_one_r",
    }
    assert required_summary_keys.issubset(summary.keys())

    assert payload["summary"]["trade_count"] == _EXPECTED_TRADE_COUNT
    assert payload["summary"]["accepted_trade_count"] == _EXPECTED_TRADE_COUNT
    assert payload["summary"]["losses_below_minus_one_r"] > 0


def test_strategy_signal_generation_runtime_for_ma_and_trend_strategies() -> None:
    ohlc = _build_signal_generation_ohlc()
    symbols = [f"SYM{idx:03d}" for idx in range(_SIGNAL_SYMBOL_COUNT)]

    # Warm-up once per strategy to reduce one-time interpreter overhead.
    build_strategy_signal_events(
        "ma_crossover",
        ohlc,
        symbol=symbols[0],
        timeframe="1d",
        fast_window=20,
        slow_window=50,
        fast_type="sma",
        slow_type="sma",
        atr_window=14,
        atr_stop_multiple=2.0,
    )
    build_strategy_signal_events(
        "trend_following",
        ohlc,
        symbol=symbols[0],
        timeframe="1d",
        trend_window=200,
        trend_type="sma",
        fast_window=20,
        fast_type="sma",
        slope_lookback_bars=3,
        atr_window=14,
        atr_stop_multiple=2.0,
    )

    start = time.perf_counter()
    event_count = 0
    for symbol in symbols:
        event_count += len(
            build_strategy_signal_events(
                "ma_crossover",
                ohlc,
                symbol=symbol,
                timeframe="1d",
                fast_window=20,
                slow_window=50,
                fast_type="sma",
                slow_type="sma",
                atr_window=14,
                atr_stop_multiple=2.0,
            )
        )
    for symbol in symbols:
        event_count += len(
            build_strategy_signal_events(
                "trend_following",
                ohlc,
                symbol=symbol,
                timeframe="1d",
                trend_window=200,
                trend_type="sma",
                fast_window=20,
                fast_type="sma",
                slope_lookback_bars=3,
                atr_window=14,
                atr_stop_multiple=2.0,
            )
        )
    elapsed = time.perf_counter() - start

    assert elapsed <= _SIGNAL_RUNTIME_THRESHOLD_SECONDS, (
        "Signal-generation runtime exceeded budget for ma_crossover/trend_following: "
        f"{elapsed:.3f}s > {_SIGNAL_RUNTIME_THRESHOLD_SECONDS:.2f}s"
    )
    assert event_count > 0
