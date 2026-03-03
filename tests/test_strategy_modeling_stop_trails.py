from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from options_helper.analysis import strategy_modeling
from options_helper.analysis.strategy_modeling import StrategyModelingRequest
from options_helper.analysis.strategy_simulator import build_r_target_ladder
from options_helper.data.strategy_modeling_artifacts import write_strategy_modeling_artifacts
from options_helper.data.strategy_modeling_io import (
    IntradayCoverageBySymbol,
    StrategyModelingDailyLoadResult,
    StrategyModelingIntradayLoadResult,
    StrategyModelingIntradayPreflightResult,
    StrategyModelingUniverseLoadResult,
)
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent


def _daily_ohlc_stage_fixture() -> pd.DataFrame:
    dates = pd.date_range("2025-11-29", "2026-01-07", freq="D", tz="UTC")
    ramp = np.clip(np.arange(dates.size, dtype="float64") - 22.0, 0.0, None)
    close = 99.0 + (0.2 * ramp)
    return pd.DataFrame(
        {
            "ts": dates,
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(dates.size, 1_000.0),
            "vwap": close,
            "trade_count": np.full(dates.size, 1.0),
        }
    )


def _intraday_stop_trail_fixture() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            ("2026-01-05 14:30:00+00:00", "2026-01-05", 100.0, 100.9, 99.8, 100.8),
            ("2026-01-05 14:31:00+00:00", "2026-01-05", 100.8, 100.9, 100.6, 100.7),
            ("2026-01-06 14:30:00+00:00", "2026-01-06", 101.2, 101.9, 101.1, 101.7),
            ("2026-01-06 14:31:00+00:00", "2026-01-06", 101.7, 101.8, 101.4, 101.6),
            ("2026-01-07 14:30:00+00:00", "2026-01-07", 103.5, 103.6, 102.8, 103.0),
        ],
        columns=["timestamp", "session_date", "open", "high", "low", "close"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
    return frame


def _stub_universe(*, database_path: str | Path | None = None) -> StrategyModelingUniverseLoadResult:
    path = Path(database_path) if database_path is not None else Path("/tmp/strategy-modeling-stop-trails.duckdb")
    return StrategyModelingUniverseLoadResult(
        symbols=["SPY"],
        notes=[],
        database_path=path,
        database_exists=True,
    )


def _stub_daily_loader(
    symbols: list[str],
    *,
    database_path: str | Path | None = None,
    **_: object,
) -> StrategyModelingDailyLoadResult:
    normalized = [str(symbol).upper() for symbol in symbols]
    candles_by_symbol = {"SPY": _daily_ohlc_stage_fixture()} if "SPY" in normalized else {}
    source_by_symbol = {"SPY": "adjusted"} if "SPY" in normalized else {}
    missing_symbols = [symbol for symbol in normalized if symbol != "SPY"]
    return StrategyModelingDailyLoadResult(
        candles_by_symbol=candles_by_symbol,
        source_by_symbol=source_by_symbol,
        skipped_symbols=[],
        missing_symbols=missing_symbols,
        notes=[],
    )


def _stub_required_sessions(
    _candles_by_symbol: dict[str, pd.DataFrame],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, list[date]]:
    return {
        "SPY": [
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
        ]
    }


def _stub_intraday_loader(
    required_sessions_by_symbol: dict[str, tuple[date, ...]],
    *,
    require_intraday_bars: bool = True,
    **_: object,
) -> StrategyModelingIntradayLoadResult:
    coverage_by_symbol: dict[str, IntradayCoverageBySymbol] = {}
    for symbol, days in required_sessions_by_symbol.items():
        required_days = tuple(sorted(set(days)))
        coverage_by_symbol[symbol] = IntradayCoverageBySymbol(
            symbol=symbol,
            required_days=required_days,
            covered_days=required_days,
            missing_days=(),
        )

    preflight = StrategyModelingIntradayPreflightResult(
        require_intraday_bars=bool(require_intraday_bars),
        coverage_by_symbol=coverage_by_symbol,
        blocked_symbols=[],
        notes=[],
    )
    return StrategyModelingIntradayLoadResult(
        bars_by_symbol={"SPY": _intraday_stop_trail_fixture()},
        preflight=preflight,
        notes=[],
    )


def _stub_signal_builder(
    *_args: object,
    symbol: object = None,
    timeframe: str | None = "1d",
    **_kwargs: object,
) -> list[StrategySignalEvent]:
    return [
        StrategySignalEvent(
            event_id="evt-stop-trail-integration",
            strategy="sfp",
            symbol=str(symbol),
            timeframe=str(timeframe or "1d"),
            direction="long",
            signal_ts=pd.Timestamp("2026-01-02T21:00:00Z").to_pydatetime(),
            signal_confirmed_ts=pd.Timestamp("2026-01-02T21:00:00Z").to_pydatetime(),
            entry_ts=pd.Timestamp("2026-01-05T00:00:00Z").to_pydatetime(),
            entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
            stop_price=99.0,
            notes=[],
        )
    ]


def test_strategy_modeling_stop_trails_capture_stage_transition_and_policy_metadata(tmp_path: Path) -> None:
    service = strategy_modeling.build_strategy_modeling_service(
        list_universe_loader=_stub_universe,
        daily_loader=_stub_daily_loader,
        required_sessions_builder=_stub_required_sessions,
        intraday_loader=_stub_intraday_loader,
        feature_computer=lambda *_args, **_kwargs: pd.DataFrame(),
        signal_builder=_stub_signal_builder,
    )
    request = StrategyModelingRequest(
        strategy="sfp",
        symbols=("SPY",),
        max_hold_bars=5,
        target_ladder=build_r_target_ladder(min_target_tenths=40, max_target_tenths=40),
        policy={
            "stop_trail_rules": [
                {"start_r": 1.5, "ema_span": 9, "buffer_atr_multiple": 0.25},
                {"start_r": 0.5, "ema_span": 21},
            ]
        },
    )

    run_result = service.run(request)

    assert len(run_result.trade_simulations) == 1
    trade = run_result.trade_simulations[0]
    assert [update.reason for update in trade.stop_updates] == [
        "stop_trail_tightened",
        "stop_trail_tightened",
    ]
    assert [update.stage for update in trade.stop_updates] == [
        "start_0.5R_ema21",
        "start_1.5R_ema9",
    ]

    paths = write_strategy_modeling_artifacts(
        out_dir=tmp_path,
        strategy="sfp",
        request=request,
        run_result=run_result,
        generated_at=datetime(2026, 2, 8, 12, 30, tzinfo=timezone.utc),
    )
    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["policy_metadata"]["stop_trail_rules"] == [
        {"start_r": 0.5, "ema_span": 21},
        {"start_r": 1.5, "ema_span": 9, "buffer_atr_multiple": 0.25},
    ]
