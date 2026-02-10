# Fib Retracement Strategy (`fib_retracement`)

This project is **not financial advice**. The fib retracement strategy output is an informational decision-support artifact.

## Model Semantics

`fib_retracement` is a close-confirmed structure workflow:

1. MSB trigger.
   - Long setup starts on a bullish MSB close.
   - Short setup starts on a bearish MSB close.
2. Next swing pivot after the MSB.
   - Long range: swing low before MSB -> next confirmed swing high after MSB.
   - Short range: swing high before MSB -> next confirmed swing low after MSB.
   - Fib-touch scanning starts only after pivot confirmation is knowable plus one full bar.
3. Fib touch.
   - `fib_retracement_pct` accepts percent (`61.8`) or ratio (`0.618`) input and is normalized to percent.
   - Long fib level: `range_high - (range_high - range_low) * fib_ratio`
   - Short fib level: `range_low + (range_high - range_low) * fib_ratio`
   - Touch condition: first bar where `low <= fib_level <= high`.

Stop anchors:
- Long: `stop_price = range_low`
- Short: `stop_price = range_high`

## Anti-Lookahead Contract

- Signal is recorded on the fib-touch bar close (`signal_ts == signal_confirmed_ts`).
- Entry is always the next bar open (`entry_ts = next bar timestamp`).
- If the touch occurs on the final available bar, the event is dropped because no next-bar entry exists.

This keeps signal confirmation and trade entry strictly ordered to avoid lookahead bias.

## CLI Usage

Default fib level (`61.8` percent):

```bash
./.venv/bin/options-helper technicals strategy-model \
  --strategy fib_retracement \
  --symbols SPY,QQQ \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --fib-retracement-pct 61.8 \
  --out data/reports/technicals/strategy_modeling
```

Equivalent ratio input:

```bash
./.venv/bin/options-helper technicals strategy-model \
  --strategy fib_retracement \
  --symbols SPY \
  --fib-retracement-pct 0.618
```

## Limitations

- Entry is **not** a resting limit order fill at the fib level; execution is modeled at the next bar open.
- Because entry is next-bar-open, gaps/slippage can make fill price materially different from the fib touch level.
- Setup confirmation depends on right-side swing confirmation bars, so pivots and touches are recognized with deterministic lag.
