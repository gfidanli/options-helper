# Feature Ideas (Backlog)

For a ranked backlog with full PRDs, milestones, and dependencies, see `docs/BACKLOG.md`.

## Research + Flow Integration (Conviction Score)
Enhance `options-helper research` by incorporating the latest locally captured options-flow snapshot deltas:

- Auto-load the latest `flow` summary for **portfolio symbols** (calls ΔOI$ vs puts ΔOI$).
- Display a simple **flow alignment** flag per symbol:
  - bullish technicals + net call-building = higher conviction
  - bullish technicals + net put-building = caution / wait for pullback
  - bearish technicals + net put-building = higher conviction (bearish)
- Add a lightweight **conviction score** (e.g., `0–100`) combining:
  - technical trend strength (weekly trend + RSI + breakout)
  - momentum state (daily RSI + daily StochRSI)
  - flow confirmation (net ΔOI$ and/or top strike ΔOI)
- Print (and save) a short “today’s plan” block:
  - primary candidates (aligned technicals + flow)
  - caution list (technicals vs flow conflict)
  - neutral / no-trade list

## Flow Snapshots For Watchlists
Flow currently snapshots only **portfolio symbols and expiries in positions**. Extend to watchlists:

- Snapshot windowed chains for a watchlist symbol set (configurable cadence).
- Allow expiry selection rules (e.g., nearest monthly, 30–90DTE, farthest LEAPS).
- Surface the same flow summary (`calls ΔOI$ / puts ΔOI$`) inside `research`.
