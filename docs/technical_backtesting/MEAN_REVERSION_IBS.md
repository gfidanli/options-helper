# MeanReversionIBS Canon (T0)

This specification is the contract for MeanReversionIBS implementation and parity checks.

This tooling is for informational research and decision-support only. It is not financial advice.

## Canonical Signal Rules
- Timeframe: daily bars.
- Entry signal at bar close `t`:
  - `range_t = high_t - low_t`
  - `ibs_t = (close_t - low_t) / range_t` when `range_t > 0`, else `ibs_t = 0.5` (deterministic neutral fallback).
  - `hh10_t = max(high_{t-9} ... high_t)`
  - `avg_range25_t = mean((high-low)_{t-24} ... (high-low)_t)`
  - Enter long when both are true:
    - `close_t < (hh10_t - 2.5 * avg_range25_t)`
    - `ibs_t < 0.3`
- Exit signal at bar close `t`:
  - Exit long when `close_t > high_{t-1}`.

## Execution Anchor and No-Lookahead Rule
- Signals are evaluated using bar-close information only.
- When `trade_on_close=false` (canonical parity mode), orders generated from bar `t` are filled at the next bar open `t+1`.
- When `trade_on_close=true`, fills are allowed on the same close bar.
- If no next bar exists, no next-open fill can occur.

## Aggregate Rule (Batch Composite)
- Per symbol, compute daily strategy return series on the run calendar.
- Daily aggregate return is equal-weight across symbols with a valid return on that date:
  - `r_agg,d = mean(r_i,d for i in active_symbols_d)`
- Aggregate equity is compounded from `r_agg`.
- Symbols missing a date are excluded from that date's denominator (not implicitly zero-return weighted).

## Benchmark Rule
- Batch benchmark is SPY buy-and-hold.
- Compute SPY close-to-close return series on the same analysis date window used for aggregate reporting.
- Benchmark equity is compounded from those SPY returns and aligned for side-by-side comparison with aggregate strategy equity.

## Reference Parity Targets (Locked)
Source of truth:  
`/Users/sergio/Library/Mobile Documents/iCloud~md~obsidian/Documents/Personal/Clippings/Found a simple mean reversion setup with 70% win rate but only invested 20% of the time.md`

### SPY targets (reference window noted as 2006-03 to 2026-03)
| Metric | Target |
|---|---:|
| Total Return | 334.84% |
| CAGR | 7.75% |
| Profit Factor | 2.02 |
| Win Rate | 75.00% |
| Max Drawdown | 15.26% |
| Time Invested | 21.02% |
| Total Trades | 240 |
| Final Capital | $434,835.64 |

### QQQ targets (reference window noted as 2011 to 2026)
| Metric | Target |
|---|---:|
| Total Return | 265.74% |
| CAGR | 9.18% |
| Profit Factor | 2.15 |
| Win Rate | 70.74% |
| Max Drawdown | 11.92% |
| Time Invested | 16.41% |
| Total Trades | 188 |
| Final Capital | $365,740.47 |

## Tolerance Policy for Parity Gate
Unless a tighter test fixture is explicitly declared, parity is considered passing when all checked headline metrics stay within:
- Percent metrics (`Total Return`, `CAGR`, `Win Rate`, `Max Drawdown`, `Time Invested`): absolute delta <= `1.00` percentage point.
- `Profit Factor`: absolute delta <= `0.10`.
- `Total Trades`: absolute delta <= `3` trades.
- `Final Capital`: relative delta <= `1.50%`.

These tolerances are intentionally narrow enough to catch semantic drift (formula/anchor/cost timing changes) while allowing small implementation and data-normalization differences.
