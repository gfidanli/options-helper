# Backtesting + Parameter Optimization Spec

## 1) Backtesting Framework: backtesting.py

We use backtesting.py because it:
- runs on a single asset OHLC DataFrame
- is well suited for optimizing entry/exit logic and technical-indicator-driven strategies
- supports parameter optimization with constraints and heatmaps

Data requirements:
- DataFrame with columns `Open`, `High`, `Low`, `Close` (+ optional `Volume`)
- Datetime index recommended

Execution assumption:
- If `trade_on_close=False` (default), market orders fill next bar open.
- If `trade_on_close=True`, market orders fill current bar close.

## 2) Strategy Templates (v1)

### 2.1 TrendPullbackATR (long-only)
**Intent:** Proxy for long calls / LEAPS management: enter on pullback inside weekly uptrend; exit/trim on extension or volatility stop.

Inputs (daily):
- `Close`, ATR(W), SMA(N), zscore(N), extension_atr
Inputs (weekly, forward-filled to daily):
- `weekly_trend_up` boolean

Rules (high level):
- Entry:
  - If `weekly_trend_up` AND `zscore_N <= add_z` then buy
- Exit:
  - If `extension_atr >= trim_ext` then close
  - OR if Close falls below trailing stop (e.g., entry_price - stop_mult * ATR) then close

Parameters to optimize:
- ATR window W ∈ [10, 21]
- SMA/Z window N ∈ [15, 40]
- add_z ∈ [-0.5, -2.5]
- trim_ext ∈ [1.0, 2.5]
- stop_mult ∈ [1.5, 4.0]
- weekly MA fast/slow (optional in v1; otherwise keep fixed)

Constraints:
- N >= W (optional; not mandatory but can reduce unstable combos)
- stop_mult > 0
- trim_ext > 0

### 2.2 MeanReversionBollinger (long-only)
**Intent:** Range regime / oversold dips; enter on lower band excursion; exit on mean reversion.

Inputs:
- Bollinger Bands (N, K): lband, mavg, pband
- Optional filter: RSI

Rules (high level):
- Entry:
  - If Close crosses below lband OR pband <= p_entry then buy
- Exit:
  - If Close >= mavg OR pband >= p_exit then close
- Optional stop:
  - max adverse move in ATR units

Parameters to optimize:
- BB window N ∈ [10, 40]
- BB dev K ∈ [1.5, 3.0]
- p_entry ∈ [0.0, 0.2]
- p_exit ∈ [0.4, 0.7]
- ATR window W ∈ [10, 21] (if using ATR-based stops)
- stop_mult ∈ [1.0, 4.0] (if using ATR-based stops)

Constraints:
- p_entry < p_exit
- stop_mult >= 1.0 (if enabled)

## 3) Optimization Approach (v1)

Use `Backtest.optimize()` with:
- `method="grid"` for deterministic baseline runs
- optional `method="sambo"` for faster search on larger spaces
- `constraint=` function to filter invalid combos
- `return_heatmap=True` for analysis

Default maximize metric:
- A custom objective is recommended (see below), but start with:
  - maximize `'SQN'` or `'Return (Ann.) [%]'`

## 4) Objectives & Stability (avoid overfitting)

### 4.1 Custom objective (recommended)
Define a scalar score:
- `score = Return (Ann.) - λ * abs(Max. Drawdown) - μ * (#Trades / years)`
Tune λ and μ to discourage fragile/overtrading configs.

Also enforce:
- minimum number of trades (e.g., >= 10 in train window)
- maximum drawdown cap (optional)

### 4.2 Walk-forward evaluation
Split the dataset into folds:
- Train: 3–5 years
- Validate: 6–12 months
- Roll forward by validate window

Selection:
- For each fold: find best params on train, score on validate
- Choose final params by median rank / average validate score
- Compute stability:
  - stddev(validate_scores), dispersion of chosen params

Fallback:
- If unstable, use bucket defaults (volatility buckets) or global defaults.

## 5) Outputs (Artifacts)

Write per ticker+strategy:
- `params.json`
  - chosen params
  - provider settings (auto_adjust/back_adjust)
  - train/validate windows
  - objective + constraints + method
  - timestamp and git commit hash (if available)

- `summary.md`
  - best train stats
  - validate stats
  - stability score
  - notes (data gaps, warnings)

- optional `heatmap.csv`
  - multi-index param grid -> objective value

## 6) References
- backtesting.py quick start and data/strategy notes:
  https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html
- backtesting.py optimize API:
  https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html
- parameter heatmap tutorial:
  https://kernc.github.io/backtesting.py/doc/examples/Parameter%20Heatmap%20%26%20Optimization.html