# Performance Architecture Guide

This repo is **not financial advice**. Performance guidance here exists to keep research tooling responsive, testable, and reliable.

## Scope

Use these rules when designing or changing:
- strategy modeling and trade simulation loops,
- signal/event studies on large candle sets,
- data ingest/backfill workflows that run across many symbols/dates.

## Core rules

1. Design for hot paths early
- Identify expected hot loops before implementation (event x target, symbol x day, contract x session).
- Keep runtime complexity explicit in PR notes/doc updates.

2. Precompute reusable data once
- Normalize and sort intraday/daily inputs once per symbol.
- Build reusable arrays/masks for timestamps, sessions, and OHLC values.
- Reuse prepared structures across all downstream calculations.

3. Resolve event-level decisions once
- Compute entry row/entry price/stop/risk/reject reason once per event.
- Reuse that decision for each target on the ladder instead of recomputing.

4. Keep pandas out inner loops
- Do not call `iterrows()`, repeated `to_datetime()`, repeated timezone conversions, or repeated DataFrame slicing per trade.
- Use vectorized/index-based lookups (`np.searchsorted`, boolean masks, ndarray access) in inner loops.

5. Preserve observability during long runs
- Long-running CLI commands should emit stage-level progress.
- Capture per-stage timings so bottlenecks are visible without a profiler.

6. Validate with tests and timing
- Add deterministic regression tests for behavior changes in hot paths.
- Run targeted before/after timings for representative command scopes.

## Anti-patterns to avoid

- Recomputing entry row selection for every `(event, target)` pair.
- Converting timestamps/timezones repeatedly inside the same loop.
- Building temporary DataFrames per trade when scalar/array access is enough.
- Shipping performance-critical changes without timing evidence.

## Minimal validation workflow

```bash
./.venv/bin/python -m pytest tests/test_strategy_simulator.py tests/test_strategy_modeling_policy.py
```

```bash
/usr/bin/time -p ./.venv/bin/options-helper technicals strategy-model \
  --strategy sfp \
  --symbols SPY,AAPL,NVDA,AMZN \
  --start-date 2025-01-01 \
  --end-date 2026-01-31 \
  --intraday-timeframe 1Min \
  --intraday-source alpaca \
  --show-progress \
  --out /tmp/strategy_modeling_perf_check
```

When stage timings show one dominant stage, optimize that stage first before broader refactors.
