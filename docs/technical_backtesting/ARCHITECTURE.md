# Architecture — Technical Indicators + Backtesting/Optimization

## 1) Chosen Libraries

### Backtesting framework
- `backtesting.py`
  - Expects OHLC DataFrame with `Open`, `High`, `Low`, `Close` (+ optional `Volume`)
  - Supports `Backtest.run()` and `Backtest.optimize()` with constraints and two methods (`grid`, `sambo`)
  - Designed around optimizing entry/exit decisions on a single asset

### Indicators
- `ta` (Technical Analysis Library in Python)
  - pandas/numpy based feature engineering library
  - Provides ATR, BollingerBands, RSI, etc.
- Optional acceleration:
  - `TA-Lib` (if installed) as an alternative indicator provider for speed/standardization

## 2) Data Flow

Existing repo code:
    yfinance OHLC fetch  ──>  raw DF (maybe extra cols)

This feature:
    (1) CandleFrame adapter
        └─ ensures required OHLC columns + datetime index

    (2) Indicator engine
        └─ adds derived indicator columns (ATR, BBands, RSI, zscore, extension_atr, weekly regime)

    (3) Strategy library (templates)
        └─ consumes CandleFrame + indicator columns

    (4) Backtest runner (backtesting.py)
        └─ produces stats + trades + equity curve

    (5) Optimizer + walk-forward
        └─ selects stable parameters and writes artifacts

    (6) Artifact store
        └─ JSON params + markdown report + optional heatmaps

## 3) Key Components & Responsibilities

### 3.1 CandleFrame Adapter
- Input: DataFrame from yfinance
- Output: DataFrame ready for backtesting.py:
  - columns: `Open`, `High`, `Low`, `Close` (+ optional `Volume`)
  - datetime index, sorted ascending, unique
- Drops or preserves extra columns depending on config, but never breaks backtesting.py requirements.

### 3.2 Indicator Engine
- Provider interface:
  - provider = "ta" (default) or "talib" (optional)
- Computes:
  - ATR(window)
  - BollingerBands(window, window_dev)
  - RSI(window)
  - Rolling SMA(window_sma)
  - Derived:
    - zscore(Close vs SMA): (Close - SMA) / rolling_std
    - extension_atr: (Close - SMA) / ATR
  - Weekly regime columns:
    - Resample OHLC to weekly, compute weekly MA(s), forward-fill to daily.

### 3.3 Strategy Templates
Each strategy is a thin wrapper around:
- entry rule
- exit rule
- risk rule(s)
- parameter schema

Templates are intended to generate parameter sets for playbooks, not to represent a final “best strategy”.

### 3.4 Backtest Runner
- `Backtest(data, strategy, cash=..., commission=..., spread=..., trade_on_close=...)`
- Runs:
  - fixed parameters (Backtest.run)
  - optimization (Backtest.optimize)

### 3.5 Optimization + Walk-forward
- Parameter search space: small, bounded
- Selection uses:
  - objective function (configurable)
  - stability score across folds
  - minimum-trades threshold

## 4) Suggested Package Layout

(Adapt to your repo’s existing structure)

src/<your_pkg>/technicals/
  adapter.py
  indicators/
    provider_base.py
    provider_ta.py
    provider_talib.py   (optional)
    derived.py
  strategies/
    base.py
    trend_pullback_atr.py
    mean_reversion_bbands.py
  backtest/
    runner.py
    optimizer.py
    walk_forward.py
    metrics.py
  artifacts/
    store.py

docs/technical_backtesting/
  PRD.md
  ARCHITECTURE.md
  DATA_CONTRACTS.md
  INDICATORS.md
  BACKTESTING_OPTIMIZATION.md
  LLM_IMPLEMENTATION_GUIDE.md
  RUNBOOK.md

## 5) Extensibility Rules
- Add a new indicator:
  - implement in provider + document in INDICATORS.md
  - add unit test for output shape/NaN warmup
- Add a new strategy template:
  - implement `strategies/<name>.py`
  - define param schema and constraints
  - add a smoke test that it runs on 1 ticker dataset