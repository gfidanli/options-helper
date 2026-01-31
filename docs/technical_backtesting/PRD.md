# PRD — Technical Indicators + Backtesting/Optimization Foundation

**Status:** Draft  
**Last updated:** 2026-01-31

## 1) Summary

Build a technical-analysis and backtesting module that:
1) Standardizes yfinance OHLCV data into a consistent CandleFrame.
2) Computes a minimal, robust set of technical indicators (daily + weekly-aligned).
3) Runs backtests for a small library of “entry/exit strategy templates”.
4) Optimizes strategy parameters (per ticker and/or volatility bucket) to generate a stable **parameter set** that the later “playbook” system will use for live decisioning.

## 2) Background / Context

We manage options positions (90–120 DTE and ~6–12 month) but the technical signals are derived from the underlying on daily/weekly candles. Because yfinance does not provide full historical option chains, this module backtests on the underlying as a proxy to calibrate timing rules.

The output of this module is not “alpha in a box”; it is a calibrated and versioned configuration that downstream playbooks consume.

## 3) Goals

### G1 — Indicator Engine
- Compute ATR-based volatility scaling and mean-reversion/trend context on daily candles
- Provide weekly regime context (computed on weekly bars, forward-filled to daily)

### G2 — Backtesting Engine (3rd-party)
- Use `backtesting.py` to simulate single-asset strategies.
- Ensure no lookahead bias in signal construction and execution assumptions.

### G3 — Parameter Optimization
- Optimize a small, controlled parameter space for each strategy template.
- Support walk-forward evaluation and stability scoring (not just best in-sample stats).

### G4 — Artifacts for Downstream Playbooks
- Persist “best parameters” per ticker (or per bucket) with metadata: date, train/validate spans, objective, constraints, version, and performance summary.

## 4) Non-goals (v1)
- Simulating option P&L, greeks, IV dynamics, or historical chain-driven signals.
- Intraday signals; daily is primary (weekly is derived).
- Live execution/broker integration.

## 5) Users / Personas
- Primary: the repo owner running nightly or ad-hoc research to calibrate playbooks.

## 6) User Stories
- US1: As a trader, I want ATR and mean-reversion/trend indicators computed consistently across tickers.
- US2: I want to backtest and compare a small set of entry/exit templates per ticker.
- US3: I want a stable parameter file I can feed into playbooks without manual tweaking.
- US4: I want a walk-forward report that discourages overfit parameter sets.

## 7) Functional Requirements

### FR1 — CandleFrame Adapter
- Input: pandas DataFrame from yfinance (index is datetime; columns may include extra fields).
- Output: standardized columns: `Open`, `High`, `Low`, `Close`, optional `Volume`.
- Enforce sorting, uniqueness, and dtype normalization.

### FR2 — Indicator Computation
Must compute (at minimum):
- ATR (Average True Range) + ATR%
- Bollinger Bands (upper/lower/middle, %B, bandwidth)
- RSI (optional but recommended)
- Derived features:
  - `zscore_close_vs_sma` (rolling)
  - `extension_atr` = (Close − SMA) / ATR

### FR3 — Strategy Templates
Implement at least two strategy templates for calibration:
- **TrendPullbackATR (long-only)**: enter on pullback in an up-regime; exit on extension or ATR stop.
- **MeanReversionBollinger (long-only)**: enter on lower-band excursion; exit on mean reversion (mid-band or %B threshold).

### FR4 — Backtest Runner
- Run `backtesting.py` for each (ticker, strategy, parameter set).
- Configurable assumptions:
  - cash, commission, spread, trade_on_close, exclusive_orders.

### FR5 — Optimizer Runner
- Use `Backtest.optimize()` to search parameter combinations, with:
  - constraints to prevent invalid combos
  - objective function that balances return and drawdown (configurable)
  - optional heatmap output

### FR6 — Walk-forward Evaluation
- Partition data into rolling train/validate windows.
- Select parameter sets that are stable across folds.
- Produce a stability score and “fallback” behavior:
  - If unstable, fall back to bucket defaults or global defaults.

### FR7 — Artifacts and Reports
Write:
- `artifacts/technicals/params/{ticker}/{strategy}.json`
- `artifacts/technicals/reports/{ticker}/{strategy}/summary.md`
- Optional: `artifacts/technicals/reports/{ticker}/{strategy}/heatmap.csv`

## 8) Non-functional Requirements
- Deterministic runs (seeded optimization if randomized).
- Clear logging and failure isolation (one ticker failing doesn’t stop the run).
- Unit tests for: data adapter, indicator shapes/NaNs, backtest runner.

## 9) Risks & Mitigations
- Overfitting → walk-forward, small parameter space, stability filters.
- Corporate actions / adjusted prices → explicit yfinance adjustment settings captured in metadata.
- Data gaps → adapter validates continuity; warns on missing bars.

## 10) Acceptance Criteria (v1)
- Given an OHLCV DataFrame, the module:
  - computes indicators without errors
  - runs both strategy templates
  - outputs a parameter JSON + summary markdown for one ticker
- Walk-forward produces a stability score and recommended parameters.