# Strategy Modeling Stop Management

This project is **not financial advice**. Stop-management modeling is informational research output for decision support only.

## Purpose

Strategy-modeling can optionally **tighten stops after entry** when price moves in your favor.

This lets you model common management rules like:
- Move stop to breakeven after +1.0R.
- Step the stop up as profit grows (for example +0R at +1R, then +1R at +2R).

## Stop Move Rules (Close-Confirmed)

Stop moves are expressed in **R multiples from entry**:
- `trigger_r`: when the trade closes at or above this R (favorable direction)
- `stop_r`: move the stop to this R level (must be `<= trigger_r`)

Semantics (anti-lookahead):
- Trigger evaluation is **on bar close**.
- The updated stop becomes active **starting on the next bar** (no same-bar stop updates).
- Rules are applied in increasing `trigger_r` order and must **tighten** (non-decreasing `stop_r`).

## CLI Usage

Use one or more `--stop-move` options:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --symbols SPY \
  --stop-move 1.0:0.0 \
  --stop-move 2.0:1.0
```

Disable stop moves (even if a loaded profile includes them):

```bash
./.venv/bin/options-helper technicals strategy-model \
  --profile my_profile \
  --disable-stop-moves
```

## Outputs

- `trades.csv` includes both `stop_price` (initial) and `stop_price_final` (last active stop).
- `summary.json -> policy_metadata.stop_move_rules` echoes the configured rules for the run.

Not financial advice.
