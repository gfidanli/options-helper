# Research — Options Ideas From Technicals (MVP)

## Goal
Given a ticker (or a watchlist), produce **best-effort** option ideas:

- **Short-dated** (30–90 DTE) for momentum / tactical swings
- **Long-dated** (LEAPS) for higher-timeframe thesis

This is a rule-based “idea generator”, not financial advice.

## Inputs
- Daily candles (from the local candle cache; fetched from `yfinance` as needed)
- `yfinance` option chains (expiries + strikes + bid/ask/IV/OI/volume)
- Your `portfolio.json` risk profile for:
  - breakout lookback window (`breakout_lookback_weeks`)
  - liquidity thresholds (`min_open_interest`, `min_volume`)

## Technicals used (MVP)
Underlying trend / momentum (computed from daily candles):

- **Weekly trend**: bullish when price > weekly EMA50 and weekly EMA20 ≥ EMA50
- **Weekly breakout**: close exceeds prior N-week high (`breakout_lookback_weeks`)
- **Weekly RSI(14)**
- **Daily RSI(14)**
- **Daily StochRSI(14/14/3)** (scaled 0–100)

The tool prints the “why” as bullet reasons so you can sanity-check the logic.

## Contract selection (MVP)
The tool chooses:

### Short-dated (30–90 DTE)
- expiry closest to ~60 DTE within the range
- **calls** for bullish setups, **puts** for bearish setups
- strike nearest a target delta:
  - calls: ~`+0.40`
  - puts:  ~`-0.40`

### LEAPS
- expiry closest to ~540 DTE in the 365+ range (or farthest available ≥ 365 DTE)
- strike nearest a higher-delta target:
  - calls: ~`+0.70` (more stock-like, lower theta pressure)
  - puts:  ~`-0.70`

## Trade levels (MVP)
For directional setups, the tool prints **suggested underlying entry and stop levels** (best-effort):

- **Entry**: current spot (latest close in the candle cache)
- **Stop**: a simple “technical invalidation” level, based on:
  - weekly breakout level (when a weekly breakout is detected), or
  - weekly/daily EMA support + recent swing levels (depending on risk tolerance)

These levels are meant to help you frame a plan. They are not financial advice.

## Output reports (MVP)
Every `research` run can be saved as a plain text report:

- default location: `data/research/`
- default filename: `research-<symbol-or-watchlist>-YYYY-MM-DD.txt`

### Strike window (efficiency)
To avoid scanning huge chains, selection is restricted to a strike window around spot:
- default: `--window-pct 0.30` (±30% around spot)

### Liquidity (best-effort)
Preference is given to strikes with:
- open interest ≥ `min_open_interest` OR volume ≥ `min_volume`

If nothing passes the filter, it falls back to the best strike by delta/ATM.

## CLI usage
Run research on your default watchlist named `watchlist`:

```bash
options-helper research portfolio.json --watchlist watchlist
```

Run research for a single ticker:

```bash
options-helper research portfolio.json --symbol IREN
```

Disable saving the report:

```bash
options-helper research portfolio.json --symbol IREN --no-save
```

## Notes / limitations
- Uses Black–Scholes to estimate delta from `impliedVolatility` (best-effort).
- Yahoo options data can be stale/incomplete, especially on illiquid chains.
- This does not consider your total capital/risk limits in the recommendation (by request).

## Future improvements (not in MVP yet)
- Multi-leg strategies (debit spreads, calendars, diagonals, etc.).
- Entry/stop logic that uses explicit support/resistance detection, ATR-based stops, and multi-timeframe confirmation.
