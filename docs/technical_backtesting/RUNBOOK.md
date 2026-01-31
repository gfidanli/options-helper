# Runbook — Technical Indicators + Backtesting/Optimization

## 1) Dependencies
Install (pin versions in your repo as you prefer):
- yfinance
- pandas, numpy
- ta
- backtesting
- (optional, required for `optimization.method: sambo`) `sambo` + `scikit-learn`
- (optional) TA-Lib

## 2) Upstream Data Fetch Requirements
Your existing yfinance fetch must:
- return a DataFrame with `Open`, `High`, `Low`, `Close` (+ optional Volume)
- set `auto_adjust` / `back_adjust` explicitly and record the choice

References:
- yfinance.download params:
  https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- yfinance history params:
  https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html

## 3) Typical Workflow

### 3.1 One ticker, one strategy (no optimization)
- Standardize DF
- Compute indicators
- Run backtest with fixed params
- Write summary artifact

### 3.2 One ticker, optimize strategy params
- Provide `search_space` dict
- Provide `constraint` callable
- Run `Backtest.optimize(method="grid" or "sambo")`
- Persist best params + report

### 3.3 Walk-forward
- Define train/validate windows
- Optimize on train, score on validate
- Roll forward, compute stability, choose final params
- Persist final params.json

## 4) CLI Examples

### Compute indicators from a local OHLC file
- `./.venv/bin/options-helper technicals compute-indicators --ohlc-path data/ohlc/AAPL.csv --output data/ohlc/AAPL_features.parquet`

### Optimize a single strategy (writes artifacts)
- `./.venv/bin/options-helper technicals optimize --strategy TrendPullbackATR --symbol AAPL --cache-dir data/candles`

### Walk-forward calibration (writes artifacts)
- `./.venv/bin/options-helper technicals walk-forward --strategy MeanReversionBollinger --symbol AAPL --cache-dir data/candles`

### Run all strategies for multiple tickers
- `./.venv/bin/options-helper technicals run-all --tickers AAPL,MSFT,SPY --cache-dir data/candles`

### Re-run without overwriting old artifacts
By default, `config/technical_backtesting.yaml` sets `artifacts.overwrite: false`.

For a “fresh run” (new output directory), copy the config to a local override and change only the artifacts section:
- `cp config/technical_backtesting.yaml data/technicals/technical_backtesting_local.yaml`
- Edit:
  - `artifacts.base_dir` (e.g., `artifacts/technicals_cvx_rerun_2026-01-31`)
  - `artifacts.overwrite: true`
- Run with `--config data/technicals/technical_backtesting_local.yaml`

## 5) Troubleshooting
- If backtests start late (missing early period):
  - rolling indicators produce NaNs; simulation starts once all indicators are valid
- If results differ after changing yfinance adjustment settings:
  - expected; record `auto_adjust/back_adjust` in artifact metadata
- If optimization is too slow:
  - reduce grid space, use `max_tries` randomized grid, or switch to `method="sambo"`
