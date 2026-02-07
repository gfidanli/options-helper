# options_helper/analysis/ — Analytics & Heuristics

## Core rule
Analysis code should be **pure**:
- No network calls
- No filesystem writes
- Deterministic outputs for a given input DataFrame/config

## Technical analysis guidelines
- Operate on cached daily candles (timezone-naive `DatetimeIndex`).
- Resample from daily to higher timeframes (weekly, multi-day) inside analysis modules.
- Always handle missing/empty data gracefully (return `None` values + human-readable reasons).
- Prevent lookahead bias:
  - when signals are close-confirmed, downstream entry/return anchors must use next-bar open
  - when swing detection uses right-side bars, consume swings only after confirmation lag

## Options research guidelines
- Recommendations should be explainable: always produce a “why” list.
- Treat greeks/IV from Yahoo as best-effort; handle zeros/NaNs.
- Prefer liquidity-aware selection (OI/volume thresholds) with safe fallbacks.

## Flow usage guidelines
- Flow is a heuristic: communicate it as “positioning proxy”, not certainty.
- Use it to adjust conviction (aligned vs conflicting technicals), not to override all other signals.

## Tests
- Add unit tests for indicator math and edge cases (NaNs, small samples).
- Avoid pandas object-dtype pitfalls (prefer `NaN` over `pd.NA` in numeric series).
