# Indicator Catalog + Parameterization

## 0) Philosophy
Use a small set of indicators that are:
- orthogonal (trend/regime vs location vs volatility scaling)
- stable across tickers
- easy to optimize without exploding the search space

Default implementation uses `ta` (bukosabino/ta). Optional provider for `TA-Lib`.

## 1) Core Indicators (Required)

### 1.1 ATR (Average True Range)
Purpose: volatility scaling for thresholds and stops.

Implementation (`ta`):
- `ta.volatility.AverageTrueRange(high, low, close, window=W).average_true_range()`

Optional (`TA-Lib`):
- `talib.ATR(high, low, close, timeperiod=W)` (note TA-Lib mentions unstable period)

Parameters:
- `W ∈ [10, 21]` default 14

Derived:
- `ATR% = ATR / Close`

### 1.2 Bollinger Bands
Purpose: mean reversion signals + “compression/expansion” via bandwidth.

Implementation (`ta`):
- `BollingerBands(close, window=N, window_dev=K)`
  - `bollinger_mavg()`, `bollinger_hband()`, `bollinger_lband()`
  - `bollinger_pband()`, `bollinger_wband()`

Parameters:
- `N ∈ [10, 40]` default 20
- `K ∈ [1.5, 3.0]` default 2.0

### 1.3 RSI
Purpose: overbought/oversold context and filters.

Implementation (`ta`):
- `RSIIndicator(close, window=R).rsi()`

Parameters:
- `R ∈ [7, 21]` default 14

## 2) Derived Features (Required)

These can be computed via pandas using core indicators.

### 2.1 SMA (rolling mean)
- `sma_N = Close.rolling(N).mean()`
Parameters:
- `N ∈ [10, 60]` default 20

### 2.2 Z-Score of Close vs SMA
- `zscore_N = (Close - sma_N) / Close.rolling(N).std()`

Note:
- If std is 0 or NaN, zscore is NaN.

### 2.3 Extension in ATR units
- `extension_atr = (Close - sma_N) / atr_W`

## 3) Weekly Regime Columns (Required)
Purpose: align daily timing with weekly trend context.

Process:
1) Resample daily OHLC to weekly OHLC (define consistent rule: e.g., week ending Friday).
2) Compute weekly MAs (e.g., 10W and 20W).
3) Define:
   - `weekly_trend_up = (weekly_close > weekly_sma_fast) AND (weekly_sma_fast > weekly_sma_slow)`
4) Forward-fill weekly values back to daily rows.

Parameters:
- weekly fast MA: [8, 13] default 10
- weekly slow MA: [18, 30] default 20

## 4) Warm-up / NaNs
All rolling indicators will have NaNs for the first `max(lookback windows)` bars.
Backtesting engine must begin only once all required indicators are non-NaN.

## 5) References
- `ta` docs (ATR, BollingerBands, RSI):
  https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
- TA-Lib ATR docs:
  https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html