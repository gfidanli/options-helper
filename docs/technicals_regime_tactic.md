# Regime Tactic (`technicals regime-tactic`)

This project is **not financial advice**. Regime/tactic outputs and any backtests are illustrative only; there is no guarantee of future results.

## Scope and Status

This page documents the intended command contract for `options-helper technicals regime-tactic` in the current milestone.

- The command is designed as a deterministic, decision-support classifier.
- It classifies both a symbol and a market proxy into coarse price regimes.
- It maps those regime tags into a tactic suggestion for long-side research (`breakout`, `undercut_reclaim`, or `avoid`).

If implementation details evolve during rollout, prefer CLI `--help` and tests as the source of truth.

## High-Level Regime Computation

Regime classification is expected to use only completed OHLC bars (no forward data), with NaN-safe handling and a minimum history guard.

Planned regime tags:
- `trend_up`
- `trend_down`
- `choppy`
- `mixed`
- `insufficient_data`

Planned signal ingredients (high-level):
- EMA slope/spacing context (for directional trend quality).
- CHOP-style trend-vs-range context.
- Recent crossing/churn context (for choppy detection).
- Minimum bar count threshold before emitting non-`insufficient_data` tags.

## Regime -> Tactic Mapping (Long-Side)

The mapping is deterministic and intentionally simple:

| Market Regime | Symbol Regime | Tactic | Support Model |
|---|---|---|---|
| trend_up / trend_down (directional) | trend_up / trend_down (aligned directional) | `breakout` | `ema` |
| choppy / mixed | choppy / mixed | `undercut_reclaim` | `static` |
| Any | insufficient_data | `avoid` | `static` |
| Conflicting / low-conviction combinations | Any | `avoid` | `static` |

Notes:
- The exact threshold constants are implementation details and may be tuned.
- The command should also emit a short rationale list to keep the recommendation auditable.

## Offline CLI Usage

Use explicit OHLC files for fully offline/deterministic runs:

```bash
./.venv/bin/options-helper technicals regime-tactic \
  --symbol AAPL \
  --ohlc-path tests/fixtures/regime_tactic/aapl_daily.csv \
  --market-symbol SPY \
  --market-ohlc-path tests/fixtures/regime_tactic/spy_daily.csv
```

Offline intent:
- `--ohlc-path` supplies the symbol series.
- `--market-ohlc-path` supplies the market proxy series.
- With both paths provided, the command should not need network access.

## OHLC Input Expectations (Conservative Contract)

For both symbol and market OHLC inputs, expect:
- Required columns equivalent to timestamp + OHLC prices (`open`, `high`, `low`, `close`).
- Parseable timestamps.
- Monotonic ascending timestamps.
- No duplicate timestamps.

If input validation fails, the CLI should return an actionable error rather than silently continuing.

## Planned Artifact Fields

Expected JSON fields (subject to additive changes):
- `schema_version`
- `asof_date` (latest candle date in the provided data period)
- `symbol`
- `market_symbol`
- `symbol_regime` + diagnostics
- `market_regime` + diagnostics
- `tactic`
- `support_model`
- `rationale`
- `disclaimer`

## Limitations

- Regime/tactic outputs are heuristic summaries, not execution instructions.
- Labels can change with threshold tuning or different input history windows.
- If either input has insufficient history or low-quality bars, the expected outcome is conservative (`avoid`/`insufficient_data`).

Not financial advice. Backtests and modeled outputs are illustrative only, and no result is guaranteed.
