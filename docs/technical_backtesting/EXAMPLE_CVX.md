# Worked Example — CVX Walk-Forward Calibration

This document is a **worked example** of running the Technical Indicators + Backtesting/Optimization feature on **CVX**.

This project is **not financial advice**. These outputs are research artifacts intended to support later playbooks/decisioning.

## 1) Prereqs

From repo root:
- Create venv + install deps:
  - `python3 -m venv .venv`
  - `./.venv/bin/pip install -e ".[dev]"`

If you plan to use `optimization.method: sambo`:
- Install the extra dependency required by backtesting.py:
  - `./.venv/bin/pip install sambo scikit-learn`

## 2) Ensure Enough Candle History

Default walk-forward settings require at least `walk_forward.min_history_years` (default: 6 years).

This repo’s default candle cache lives in `data/candles/` (gitignored). For CVX we ensured a ~10y window ending **2026-01-30**:

- `./.venv/bin/python - <<'PY'`
  `from datetime import date`
  `from pathlib import Path`
  `from options_helper.data.candles import CandleStore`
  `store = CandleStore(Path("data/candles"))`
  `df = store.get_daily_history("CVX", period="10y", today=date(2026, 1, 31))`
  `print(len(df), df.index.min(), df.index.max())`
  `PY`

## 3) Config Used For This Run

The canonical config is `config/technical_backtesting.yaml`.

For this CVX run we used a local override (not committed) at:
- `data/technicals/technical_backtesting_cvx.yaml`

Overrides applied:
- `data.candles.price_adjustment.auto_adjust: false` (matches the cached CSV which includes `Adj Close`)
- `optimization.method: sambo` (faster than full grid for walk-forward folds)

## 4) Run Commands

### 4.1 TrendPullbackATR walk-forward
- `./.venv/bin/options-helper technicals walk-forward --strategy TrendPullbackATR --symbol CVX --cache-dir data/candles --config data/technicals/technical_backtesting_cvx.yaml`

### 4.2 MeanReversionBollinger walk-forward
- `./.venv/bin/options-helper technicals walk-forward --strategy MeanReversionBollinger --symbol CVX --cache-dir data/candles --config data/technicals/technical_backtesting_cvx.yaml`

## 5) Outputs

Artifacts are written under `artifacts/technicals/` (gitignored):

### TrendPullbackATR
- Params: `artifacts/technicals/params/CVX/TrendPullbackATR.json`
- Report: `artifacts/technicals/reports/CVX/TrendPullbackATR/summary.md`
- Heatmap: `artifacts/technicals/reports/CVX/TrendPullbackATR/heatmap.csv`

### MeanReversionBollinger
- Params: `artifacts/technicals/params/CVX/MeanReversionBollinger.json`
- Report: `artifacts/technicals/reports/CVX/MeanReversionBollinger/summary.md`
- Heatmap: `artifacts/technicals/reports/CVX/MeanReversionBollinger/heatmap.csv`

## 6) Results Snapshot (Run Date: 2026-01-31)

Data window used (both strategies):
- Start: **2016-02-03**
- End: **2026-01-30**

Walk-forward config (from `config/technical_backtesting.yaml`):
- Train: 4 years
- Validate: 6 months
- Step: 6 months
- Stability gate: `max_validate_score_cv = 0.60`
- Objective: `custom_score`

### 6.1 Outcome: Fallback to defaults

Both strategies returned `used_defaults=true` with `reason="unstable_or_low_trades"`.

Primary cause:
- In 6-month validation windows, the strategies produced **very few trades** (often 0–2).
- The config’s `optimization.custom_score.min_trades = 10` is too high to satisfy per-fold validation when validation windows are only 6 months.

Practical implication:
- Unless you extend the validation window (e.g., 12 months) or relax the trade-count gating for validation, walk-forward will often fail stability checks and fall back to defaults for “slow” strategies.

## 7) (Optional) In-sample Optimization Snapshot

If you want a quick “best in-sample” fit (not walk-forward), run `technicals optimize`.

For this CVX run we wrote in-sample results to a separate base dir to avoid overwriting walk-forward artifacts:
- `artifacts/technicals_cvx_optimize/params/CVX/TrendPullbackATR.json`
- `artifacts/technicals_cvx_optimize/params/CVX/MeanReversionBollinger.json`

Observed (in-sample) summary:
- TrendPullbackATR best params: `atr_window=21, sma_window=30, z_window=30, add_z=-1.0, trim_ext_atr=1.0, stop_mult_atr=2.5` (B&H return over same window still higher).
- MeanReversionBollinger best params: `bb_window=20, bb_dev=2.5, p_entry=0.05, p_exit=0.4, atr_window=21, stop_mult_atr=2.5` (higher drawdown; walk-forward stability still failed).
