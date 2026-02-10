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
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent, StrategyTradeSimulation


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


def _sample_ma_trend_daily_candles() -> pd.DataFrame:
    ts = pd.date_range("2026-01-05", periods=8, freq="B", tz="UTC")
    close = [10.0, 9.0, 8.0, 7.0, 9.0, 11.0, 12.0, 13.0]
    open_ = [10.2, 9.2, 8.2, 7.2, 8.8, 10.8, 11.8, 12.8]
    high = [value + 0.4 for value in close]
    low = [value - 0.4 for value in close]
    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100] * len(ts),
            "vwap": close,
            "trade_count": [1] * len(ts),
        }
    )


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


def _stub_daily_loader_ma_trend(
    symbols: list[str],
    *,
    database_path: str | Path | None = None,
    **_: object,
) -> StrategyModelingDailyLoadResult:
    normalized = [str(symbol).upper() for symbol in symbols]
    candles_by_symbol = {"SPY": _sample_ma_trend_daily_candles()} if "SPY" in normalized else {}
    source_by_symbol = {"SPY": "adjusted"} if "SPY" in normalized else {}
    missing_symbols = [symbol for symbol in normalized if symbol != "SPY"]
    return StrategyModelingDailyLoadResult(
        candles_by_symbol=candles_by_symbol,
        source_by_symbol=source_by_symbol,
        skipped_symbols=[],
        missing_symbols=missing_symbols,
        notes=["stub:daily"],
    )


def _stub_intraday_loader_ma_trend(
    required_sessions_by_symbol: dict[str, tuple],
    *,
    require_intraday_bars: bool = True,
    **_: object,
) -> StrategyModelingIntradayLoadResult:
    coverage_by_symbol: dict[str, IntradayCoverageBySymbol] = {}
    bars_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol, days in required_sessions_by_symbol.items():
        required_days = tuple(sorted(set(days)))
        coverage_by_symbol[symbol] = IntradayCoverageBySymbol(
            symbol=symbol,
            required_days=required_days,
            covered_days=required_days,
            missing_days=(),
        )

        rows: list[tuple[str, float, float, float, float, str]] = []
        for session_day in required_days:
            session_ts = pd.Timestamp(session_day)
            first_ts = (session_ts + pd.Timedelta(hours=14, minutes=30)).tz_localize("UTC")
            second_ts = first_ts + pd.Timedelta(minutes=1)
            rows.append((first_ts.isoformat(), 11.0, 11.2, 10.8, 11.1, str(session_day)))
            rows.append((second_ts.isoformat(), 11.1, 11.3, 10.9, 11.0, str(session_day)))

        frame = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "session_date"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce")
        bars_by_symbol[symbol] = frame

    preflight = StrategyModelingIntradayPreflightResult(
        require_intraday_bars=bool(require_intraday_bars),
        coverage_by_symbol=coverage_by_symbol,
        blocked_symbols=[],
        notes=["stub:preflight"],
    )
    return StrategyModelingIntradayLoadResult(
        bars_by_symbol=bars_by_symbol,
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


def test_strategy_modeling_service_runs_ma_crossover_and_trend_following_strategies(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(strategy_modeling, "list_strategy_modeling_universe", _stub_list_universe)
    monkeypatch.setattr(strategy_modeling, "load_daily_ohlc_history", _stub_daily_loader_ma_trend)
    monkeypatch.setattr(strategy_modeling, "load_required_intraday_bars", _stub_intraday_loader_ma_trend)

    service = strategy_modeling.build_strategy_modeling_service()
    strategy_cases = [
        (
            "ma_crossover",
            {
                "fast_window": 2,
                "slow_window": 3,
                "fast_type": "sma",
                "slow_type": "sma",
                "atr_window": 2,
                "atr_stop_multiple": 1.5,
            },
        ),
        (
            "trend_following",
            {
                "trend_window": 3,
                "trend_type": "sma",
                "fast_window": 2,
                "fast_type": "sma",
                "slope_lookback_bars": 1,
                "atr_window": 2,
                "atr_stop_multiple": 1.5,
            },
        ),
    ]

    for strategy_name, signal_kwargs in strategy_cases:
        request = StrategyModelingRequest(
            strategy=strategy_name,
            symbols=("SPY",),
            signal_kwargs=signal_kwargs,
            target_ladder=build_r_target_ladder(min_target_tenths=10, max_target_tenths=10),
            starting_capital=10_000.0,
            max_hold_bars=2,
        )
        result = service.run(request)

        assert result.strategy == strategy_name
        assert result.is_intraday_blocked is False
        assert len(result.signal_events) >= 1
        assert all(event.strategy == strategy_name for event in result.signal_events)
        assert len(result.trade_simulations) >= 1
        assert any(trade.status == "closed" for trade in result.trade_simulations)


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


def test_strategy_modeling_service_directional_counterfactuals_use_portfolio_target_subset() -> None:
    """Portfolio metrics/ledger/segmentation use one target subset while ladder stats stay global."""

    def _stub_universe(*, database_path=None):  # noqa: ANN001,ANN202
        return StrategyModelingUniverseLoadResult(
            symbols=["SPY", "QQQ"],
            notes=["stub:universe"],
            database_path=Path("/tmp/strategy-modeling-test.duckdb"),
            database_exists=True,
        )

    def _stub_daily_loader(symbols, **_):  # noqa: ANN001,ANN202
        requested = [str(symbol).upper() for symbol in symbols]
        candles_by_symbol = {symbol: _sample_daily_candles() for symbol in requested if symbol in {"SPY", "QQQ"}}
        source_by_symbol = {symbol: "adjusted" for symbol in candles_by_symbol}
        missing = [symbol for symbol in requested if symbol not in candles_by_symbol]
        return StrategyModelingDailyLoadResult(
            candles_by_symbol=candles_by_symbol,
            source_by_symbol=source_by_symbol,
            skipped_symbols=[],
            missing_symbols=missing,
            notes=["stub:daily"],
        )

    def _stub_intraday_loader(required_sessions_by_symbol, **_):  # noqa: ANN001,ANN202
        coverage_by_symbol: dict[str, IntradayCoverageBySymbol] = {}
        bars_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol, days in required_sessions_by_symbol.items():
            required_days = tuple(sorted(set(days)))
            coverage_by_symbol[symbol] = IntradayCoverageBySymbol(
                symbol=symbol,
                required_days=required_days,
                covered_days=required_days,
                missing_days=(),
            )
            bars_by_symbol[symbol] = _sample_intraday_bars()

        preflight = StrategyModelingIntradayPreflightResult(
            require_intraday_bars=True,
            coverage_by_symbol=coverage_by_symbol,
            blocked_symbols=[],
            notes=["stub:preflight"],
        )
        return StrategyModelingIntradayLoadResult(
            bars_by_symbol=bars_by_symbol,
            preflight=preflight,
            notes=["stub:intraday"],
        )

    def _stub_signal_builder(*_args, symbol=None, timeframe="1d", **_kwargs):  # noqa: ANN001,ANN202
        normalized = str(symbol or "").upper()
        direction = "long" if normalized == "SPY" else "short"
        stop_price = 9.0 if direction == "long" else 21.0
        base_ts = pd.Timestamp("2026-01-12T14:30:00Z") if normalized == "SPY" else pd.Timestamp("2026-07-13T14:30:00Z")
        return [
            StrategySignalEvent(
                event_id=f"evt-{normalized.lower()}",
                strategy="sfp",
                symbol=normalized,
                timeframe=str(timeframe or "1d"),
                direction=direction,  # type: ignore[arg-type]
                signal_ts=base_ts.to_pydatetime(),
                signal_confirmed_ts=base_ts.to_pydatetime(),
                entry_ts=base_ts.to_pydatetime(),
                entry_price_source="first_tradable_bar_open_after_signal_confirmed_ts",
                stop_price=stop_price,
                notes=[],
            )
        ]

    def _stub_trade_simulator(events, _bars_by_symbol, *, target_ladder=None, **_):  # noqa: ANN001,ANN202
        ladder = tuple(target_ladder or build_r_target_ladder(min_target_tenths=10, max_target_tenths=20, step_tenths=10))
        out: list[StrategyTradeSimulation] = []
        for event in events:
            direction = str(event.direction).strip().lower()
            entry_price = 10.0 if direction == "long" else 20.0
            initial_risk = 1.0
            stop_price = entry_price - initial_risk if direction == "long" else entry_price + initial_risk
            entry_ts = pd.Timestamp(event.entry_ts)
            exit_ts = (entry_ts + pd.Timedelta(days=30)).to_pydatetime()

            for target in ladder:
                target_r = float(target.target_r)
                target_price = entry_price + target_r if direction == "long" else entry_price - target_r

                if direction == "long" and target.label == "1.0R":
                    exit_price = target_price
                    exit_reason = "target_hit"
                    realized_r = 1.0
                elif direction == "long":
                    exit_price = stop_price
                    exit_reason = "stop_hit"
                    realized_r = -1.0
                elif target.label == "1.0R":
                    exit_price = stop_price
                    exit_reason = "stop_hit"
                    realized_r = -1.0
                else:
                    exit_price = target_price
                    exit_reason = "target_hit"
                    realized_r = target_r

                out.append(
                    StrategyTradeSimulation(
                        trade_id=f"{event.event_id}:{target.label}",
                        event_id=event.event_id,
                        strategy=event.strategy,
                        symbol=event.symbol,
                        direction=event.direction,
                        signal_ts=event.signal_ts,
                        signal_confirmed_ts=event.signal_confirmed_ts,
                        entry_ts=entry_ts.to_pydatetime(),
                        entry_price_source=event.entry_price_source,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        exit_ts=exit_ts,
                        exit_price=exit_price,
                        status="closed",
                        exit_reason=exit_reason,  # type: ignore[arg-type]
                        reject_code=None,
                        initial_risk=initial_risk,
                        realized_r=realized_r,
                        mae_r=min(0.0, realized_r),
                        mfe_r=max(0.0, realized_r),
                        holding_bars=1,
                        gap_fill_applied=False,
                    )
                )
        return out

    service = strategy_modeling.build_strategy_modeling_service(
        list_universe_loader=_stub_universe,
        daily_loader=_stub_daily_loader,
        intraday_loader=_stub_intraday_loader,
        feature_computer=lambda *_args, **_kwargs: pd.DataFrame(),
        signal_builder=_stub_signal_builder,
        trade_simulator=_stub_trade_simulator,
    )

    request = StrategyModelingRequest(
        strategy="sfp",
        symbols=("SPY", "QQQ"),
        target_ladder=build_r_target_ladder(min_target_tenths=10, max_target_tenths=20, step_tenths=10),
        max_hold_bars=2,
    )

    result = service.run(request)

    assert len(result.trade_simulations) == 4
    assert result.portfolio_metrics.trade_count == 2
    assert set(result.accepted_trade_ids) == {"evt-spy:1.0R", "evt-qqq:1.0R"}
    assert result.segmentation.base_trade_count == 2

    ladder_by_label = {row.target_label: row for row in result.target_hit_rates}
    assert set(ladder_by_label) == {"1.0R", "2.0R"}
    assert ladder_by_label["1.0R"].trade_count == 2
    assert ladder_by_label["2.0R"].trade_count == 2
    assert ladder_by_label["1.0R"].hit_count == 1
    assert ladder_by_label["2.0R"].hit_count == 1

    directional = result.directional_metrics
    assert {"counts", "portfolio_target", "combined", "long_only", "short_only"}.issubset(set(directional))
    assert directional["counts"]["all_simulated_trade_count"] == 4
    assert directional["counts"]["portfolio_subset_trade_count"] == 2
    assert directional["portfolio_target"]["target_label"] == "1.0R"
    assert directional["portfolio_target"]["selection_source"] == "preferred_target_label"

    assert directional["combined"]["trade_count"] == 2
    assert directional["long_only"]["trade_count"] == 1
    assert directional["short_only"]["trade_count"] == 1
    assert directional["combined"]["trade_count"] == result.portfolio_metrics.trade_count

    long_return = directional["long_only"]["portfolio_metrics"]["total_return_pct"]
    short_return = directional["short_only"]["portfolio_metrics"]["total_return_pct"]
    combined_return = directional["combined"]["portfolio_metrics"]["total_return_pct"]
    assert long_return > 0.0
    assert short_return < 0.0
    assert long_return != combined_return
