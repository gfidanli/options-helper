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
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent


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


def test_strategy_modeling_service_enters_at_regular_open_after_daily_confirmation() -> None:
    """Regression: avoid after-hours rows being selected as daily next-open entry anchors."""

    def _stub_universe(*, database_path=None):  # noqa: ANN001,ANN202
        return StrategyModelingUniverseLoadResult(
            symbols=["SPY"],
            notes=["stub:universe"],
            database_path=Path("/tmp/strategy-modeling-test.duckdb"),
            database_exists=True,
        )

    def _stub_daily_loader(symbols, **_):  # noqa: ANN001,ANN202
        candles = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-14T00:00:00Z", "2026-01-15T00:00:00Z"], utc=True),
                "open": [690.0, 694.0],
                "high": [692.0, 696.0],
                "low": [688.0, 691.0],
                "close": [691.0, 692.0],
                "volume": [1.0, 1.0],
                "vwap": [690.5, 693.5],
                "trade_count": [1.0, 1.0],
            }
        )
        return StrategyModelingDailyLoadResult(
            candles_by_symbol={"SPY": candles},
            source_by_symbol={"SPY": "adjusted"},
            skipped_symbols=[],
            missing_symbols=[],
            notes=["stub:daily"],
        )

    def _stub_required_sessions(*_, **__):  # noqa: ANN001,ANN202
        return {"SPY": [pd.Timestamp("2026-01-15").date()]}

    def _stub_intraday_loader(required_sessions_by_symbol, **_):  # noqa: ANN001,ANN202
        bars = pd.DataFrame(
            [
                # Cross-session after-hours row that must NOT be selected as the next-day entry.
                ("2026-01-15T00:00:00Z", "2026-01-14", 689.55, 689.60, 689.40, 689.45),
                # Premarket row.
                ("2026-01-15T09:00:00Z", "2026-01-15", 691.10, 691.63, 690.65, 691.63),
                # Regular-session market open (09:30 ET).
                ("2026-01-15T14:30:00Z", "2026-01-15", 694.57, 694.69, 693.97, 694.07),
                ("2026-01-15T14:31:00Z", "2026-01-15", 694.07, 694.20, 693.90, 694.00),
            ],
            columns=["timestamp", "session_date", "open", "high", "low", "close"],
        )
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars["session_date"] = pd.to_datetime(bars["session_date"], errors="coerce")

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
            require_intraday_bars=True,
            coverage_by_symbol=coverage_by_symbol,
            blocked_symbols=[],
            notes=["stub:preflight"],
        )
        return StrategyModelingIntradayLoadResult(
            bars_by_symbol={"SPY": bars},
            preflight=preflight,
            notes=["stub:intraday"],
        )

    def _stub_signal_builder(*_, **__):  # noqa: ANN001,ANN202
        return [
            StrategySignalEvent(
                event_id="sfp:SPY:1d:2026-01-14T00:00:00:long",
                strategy="sfp",
                symbol="SPY",
                timeframe="1d",
                direction="long",
                signal_ts=pd.Timestamp("2026-01-14T00:00:00Z").to_pydatetime(),
                signal_confirmed_ts=pd.Timestamp("2026-01-14T00:00:00Z").to_pydatetime(),
                entry_ts=pd.Timestamp("2026-01-15T00:00:00Z").to_pydatetime(),
                entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
                stop_price=693.0,
                notes=[],
            )
        ]

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
        intraday_timeframe="1Min",
        target_ladder=build_r_target_ladder(min_target_tenths=10, max_target_tenths=10),
        max_hold_bars=2,
    )

    result = service.run(request)
    assert len(result.trade_simulations) == 1
    trade = result.trade_simulations[0]
    assert trade.entry_ts == pd.Timestamp("2026-01-15T14:30:00Z").to_pydatetime()
    assert float(trade.entry_price) == 694.57
