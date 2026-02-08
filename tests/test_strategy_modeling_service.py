from __future__ import annotations

from pathlib import Path

import pandas as pd

import options_helper.cli_deps as cli_deps
from options_helper.analysis import strategy_modeling
from options_helper.analysis.strategy_modeling import StrategyModelingRequest
from options_helper.analysis.strategy_simulator import build_r_target_ladder
from options_helper.data.strategy_modeling_io import (
    IntradayCoverageBySymbol,
    StrategyModelingDailyLoadResult,
    StrategyModelingIntradayLoadResult,
    StrategyModelingIntradayPreflightResult,
    StrategyModelingUniverseLoadResult,
)


def _sample_daily_candles() -> pd.DataFrame:
    ts = pd.date_range("2026-01-05", periods=7, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": [1.0, 5.2, 4.4, 2.9, 4.6, 4.2, 4.0],
            "high": [1.2, 5.5, 4.0, 3.0, 6.0, 4.5, 4.3],
            "low": [0.8, 4.2, 3.8, 2.5, 4.1, 3.8, 3.7],
            "close": [1.0, 5.1, 3.9, 2.8, 4.7, 4.1, 4.0],
            "volume": [100, 100, 100, 100, 100, 100, 100],
            "vwap": [1.0, 5.1, 3.9, 2.8, 4.7, 4.1, 4.0],
            "trade_count": [1, 1, 1, 1, 1, 1, 1],
        }
    )


def _sample_intraday_bars() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            ("2026-01-12T14:30:00Z", 4.5, 4.6, 4.4, 4.5),
            ("2026-01-12T14:31:00Z", 4.5, 4.7, 4.3, 4.5),
        ],
        columns=["timestamp", "open", "high", "low", "close"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _stub_list_universe(*, database_path: str | Path | None = None) -> StrategyModelingUniverseLoadResult:
    path = Path(database_path) if database_path is not None else Path("/tmp/strategy-modeling-test.duckdb")
    return StrategyModelingUniverseLoadResult(
        symbols=["SPY"],
        notes=["stub:universe"],
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
    candles_by_symbol = {"SPY": _sample_daily_candles()} if "SPY" in normalized else {}
    source_by_symbol = {"SPY": "adjusted"} if "SPY" in normalized else {}
    missing_symbols = [symbol for symbol in normalized if symbol != "SPY"]
    return StrategyModelingDailyLoadResult(
        candles_by_symbol=candles_by_symbol,
        source_by_symbol=source_by_symbol,
        skipped_symbols=[],
        missing_symbols=missing_symbols,
        notes=["stub:daily"],
    )


def _stub_intraday_loader(
    required_sessions_by_symbol: dict[str, tuple],
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
        notes=["stub:preflight"],
    )
    return StrategyModelingIntradayLoadResult(
        bars_by_symbol={"SPY": _sample_intraday_bars()},
        preflight=preflight,
        notes=["stub:intraday"],
    )


def test_strategy_modeling_service_parity_cli_vs_streamlit_callers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(strategy_modeling, "list_strategy_modeling_universe", _stub_list_universe)
    monkeypatch.setattr(strategy_modeling, "load_daily_ohlc_history", _stub_daily_loader)
    monkeypatch.setattr(strategy_modeling, "load_required_intraday_bars", _stub_intraday_loader)

    request = StrategyModelingRequest(
        strategy="sfp",
        symbols=("SPY",),
        database_path=tmp_path / "warehouse.duckdb",
        intraday_dir=tmp_path / "intraday",
        signal_kwargs={
            "swing_left_bars": 1,
            "swing_right_bars": 2,
            "min_swing_distance_bars": 1,
        },
        target_ladder=build_r_target_ladder(min_target_tenths=10, max_target_tenths=10),
        starting_capital=10_000.0,
        max_hold_bars=2,
    )

    cli_service = cli_deps.build_strategy_modeling_service()
    streamlit_service = strategy_modeling.build_strategy_modeling_service()

    cli_result = cli_service.run(request)
    streamlit_result = streamlit_service.run(request)

    assert cli_result == streamlit_result
    assert len(cli_result.signal_events) == 1
    assert len(cli_result.trade_simulations) == 1
    assert cli_result.trade_simulations[0].status == "closed"
    assert cli_result.portfolio_metrics.trade_count == 1
    assert cli_result.is_intraday_blocked is False
