# LLM Implementation Guide — Technical Indicators + Backtesting

This guide is written so an LLM can implement the feature with minimal back-and-forth.

## 0) Non-negotiable Requirements
- Do NOT implement a custom backtesting engine.
- Use:
  - `ta` for indicators (default provider)
  - `backtesting.py` for simulation + optimization
- Preserve compatibility with upstream yfinance fetch code:
  - The module must accept a pandas DataFrame (CandleFrame) directly.

## 1) Implementation Plan (PR-sized chunks)

### PR1 — Data Adapter + Validation
Deliver:
- `adapter.py`
  - `standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame`
- Validations:
  - required columns exist (or raise with helpful message)
  - datetime index and sorted
- Unit tests:
  - handles extra columns
  - normalizes column names
  - detects duplicates / unsorted index

### PR2 — Indicator Provider (`ta`)
Deliver:
- `indicators/provider_ta.py`
  - `compute_indicators(df, config) -> FeatureFrame`
- Computes:
  - ATR(W), Bollinger(N,K), RSI(R), SMA(N), zscore(N), extension_atr
  - weekly regime columns (computed from resampled weekly)
- Unit tests:
  - expected columns exist
  - warm-up NaNs are present and then disappear
  - no lookahead: spot-check that indicator[t] depends only on data <= t

### PR3 — Strategy Templates (two strategies)
Deliver:
- `strategies/trend_pullback_atr.py`
- `strategies/mean_reversion_bbands.py`
Both must:
- subclass `backtesting.Strategy`
- declare indicators via `self.I(...)` where appropriate
- use `self.buy()` / `self.position.close()` / stops as configured

Add smoke tests:
- backtest runs end-to-end on a tiny sample dataset

### PR4 — Backtest Runner + Optimizer
Deliver:
- `backtest/runner.py`:
  - `run_backtest(df, StrategyClass, bt_config, strat_params) -> stats`
- `backtest/optimizer.py`:
  - `optimize_params(df, StrategyClass, bt_config, search_space, constraint, maximize, method, max_tries)`

### PR5 — Walk-forward + Artifact Writer
Deliver:
- `backtest/walk_forward.py`
- `artifacts/store.py`
- Writes JSON + markdown reports as specified

## 2) Coding Standards
- Type hints everywhere
- Deterministic defaults: set random_state if using randomized search
- Logging must include: ticker, date range, adjustment settings, strategy name, params

## 3) Config-Driven Behavior
Implement config loading (yaml/json) to control:
- indicator windows and defaults
- strategy parameter spaces
- backtesting.py parameters (commission/spread/trade_on_close)

## 4) Definition of Done
- `pytest` green
- Running optimization for a single ticker writes:
  - params.json
  - summary.md
- Walk-forward run produces a stability score and final recommended params.