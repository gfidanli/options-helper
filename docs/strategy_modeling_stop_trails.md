# Strategy Modeling Stop Trails

This project is **not financial advice**. Stop-trail modeling and backtests are illustrative only; there is no guarantee of future performance.

## Purpose

Stop trails are a staged stop-management policy for `technicals strategy-model`.

They are intended to model behavior like:
- Early trend capture with a wider EMA anchor (for example EMA21).
- Later profit protection with a tighter EMA anchor (for example EMA9) after a profit threshold in R.

For baseline stop moves (`trigger_r:stop_r`), see [Strategy Modeling Stop Management](STRATEGY_MODELING_STOP_MANAGEMENT.md).

## Rule Syntax (Planned CLI Contract)

Use repeatable `--stop-trail` values:

- `start_r:ema_span`
- `start_r:ema_span:buffer_atr_multiple`

Definitions:
- `start_r`: activation threshold in realized/favorable R (`>= 0`).
- `ema_span`: EMA anchor span (planned allowed set: `9`, `21`, `50`, `200`).
- `buffer_atr_multiple`: optional non-negative ATR buffer factor.

Disable all stop trails (even if a profile includes them):

- `--disable-stop-trails`

Example:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --symbols SPY \
  --stop-trail 0.0:21 \
  --stop-trail 1.0:9:0.25
```

Interpretation:
- Start with EMA21 trailing behavior from entry (`0.0R`).
- After at least `+1.0R` is reached, switch to EMA9 with an ATR buffer.

## Lookahead-Bias Guardrails

Stop trails are expected to preserve strict anti-lookahead semantics:

- Stage activation should use only completed-bar information.
- EMA/ATR context should be taken from the prior completed daily session.
- A newly computed stop should become active no earlier than the next session open.
- No same-bar, retroactive stop tightening.

## Tighten-Only and Rule Precedence

Stop updates are expected to be tighten-only:

- Long: `new_stop = max(current_stop, proposed_candidates...)`
- Short: `new_stop = min(current_stop, proposed_candidates...)`

When both stop-move rules and stop-trail rules are configured, both can propose stops, and the tighten-only reducer determines the active stop.

## Offline/Deterministic Usage Notes

For deterministic runs:
- Use local fixtures for candles/intraday inputs (no network dependency).
- Keep timeframe and date ranges stable across runs.
- Preserve policy metadata in artifacts for reproducibility.

## Limitations

- Requires usable daily context for EMA/ATR anchoring; missing context should result in no trail update for that step.
- Trailing updates are model simplifications and may not reflect live fill mechanics.
- Staged rules are sensitive to bar resolution and data quality.
- This is a decision-support model, not an execution engine.

Not financial advice. Backtests are illustrative, and no outcomes are guaranteed.
