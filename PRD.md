# Options Helper (CLI) — MVP PRD

## Summary
A local, CLI-first “options portfolio helper” that:

- Reads a JSON portfolio (cash + option positions + risk profile).
- Fetches the latest available underlying + options chain data using `yfinance`.
- Matches each position to its option contract in the chain.
- Computes diagnostics (PnL, DTE, IV, basic Greeks, simple technicals).
- Produces explainable, rule-based suggestions (hold/close/roll/add/reduce) that respect
  the portfolio owner’s capital and risk limits.

Notes:

- Data is best-effort; Yahoo Finance is not guaranteed real-time.
- Output is informational only (not financial advice).

## Problem
Options holders often need repeatable, consistent checks:

- “How are my contracts doing vs cost basis?”
- “What’s my risk vs my capital?”
- “Is the underlying trend still supportive of my thesis?”
- “Is it time to roll or close?”

## Target user
Retail options traders managing a small number of single-leg long calls/puts who want
a simple, explainable workflow from a terminal.

## Goals (MVP)
- Import/maintain a small options portfolio in a JSON file.
- Fetch the latest available underlying price + relevant option chains.
- Compute per-position metrics and portfolio-level risk usage.
- Provide explainable rule-based suggestions aligned to a risk profile.
- Support polling (“watch mode”) to refresh at regular intervals.

## Non-goals (MVP)
- Broker integration / auto-trading.
- Guaranteed real-time quotes.
- Complex strategies (multi-leg spreads) beyond a roadmap placeholder.
- Tax/accounting.

## MVP user stories
- As a user, I can create a portfolio JSON template.
- As a user, I can add/remove/list positions via CLI.
- As a user, I can run an analysis and see metrics + suggested actions.
- As a user, I can run watch mode to refresh and reprint advice.
- As a user, I can set risk limits (portfolio and per-position).

## Success criteria
- Correct contract matching for typical single-leg options.
- Graceful handling of missing/partial chain data.
- Each recommendation includes: action, confidence, and rationale.
- “Add” suggestions are blocked when risk budget is exceeded.

---

## Core features (MVP)

### 1) Portfolio JSON (local storage)
Stores:

- Cash balance
- Positions (symbol, expiry, strike, call/put, contracts, cost basis)
- Risk profile thresholds

Validated on load (clear error messages).

### 2) Data fetching (yfinance)
Underlying:

- Latest available price (best-effort)
- OHLC history (e.g. 6 months daily) for indicators

Options:

- Fetch expirations needed by positions
- Optionally fetch 1–2 candidate expirations for roll recommendations

### 3) Analytics
Per position:

- Mark price (mid if bid/ask present else last)
- DTE, moneyness, breakeven
- Unrealized PnL ($, %)
- Liquidity checks (volume / open interest thresholds)
- IV (from chain when available)
- Approx Greeks (Black–Scholes using IV; best-effort)

Underlying technicals (simple, explainable):

- SMA20 vs SMA50 trend state
- RSI(14) state

### 4) Advice engine (rule-based)
Actions: `HOLD`, `CLOSE`, `ROLL`, `ADD`, `REDUCE`

Always produce:

- Recommended action
- Confidence (low/medium/high)
- Rationale bullets (which rules triggered)

### 5) CLI UX
Commands (MVP):

- `init` (create portfolio template)
- `add-position` / `remove-position` / `list`
- `refresh` (fetch + cache snapshot)
- `analyze` (compute + print advice)
- `watch` (poll refresh+analyze)

Use `rich` for readable tables.

---

## Data model (MVP JSON)
Example `portfolio.json`:

```json
{
  "base_currency": "USD",
  "cash": 5000.0,
  "risk_profile": {
    "tolerance": "medium",
    "max_portfolio_risk_pct": 0.25,
    "max_single_position_risk_pct": 0.07,
    "take_profit_pct": 0.6,
    "stop_loss_pct": 0.35,
    "roll_dte_threshold": 21,
    "preferred_roll_dte": 60,
    "min_open_interest": 100,
    "min_volume": 10
  },
  "positions": [
    {
      "id": "uroy-2026-04-17-5c",
      "symbol": "UROY",
      "option_type": "call",
      "expiry": "2026-04-17",
      "strike": 5.0,
      "contracts": 1,
      "cost_basis": 0.45,
      "opened_at": "2026-01-10"
    }
  ]
}
```

Conventions:

- `cost_basis` is premium per share (1 contract cost = `cost_basis * 100`).
- Store dates in ISO format; accept shorthand in CLI and normalize.

---

## Advice engine rules (MVP)
Keep rules simple and explainable.

### Risk gating (always first)
- If “max loss at expiry” (premium paid for long options) exceeds
  `max_single_position_risk_pct * capital`, suggest `REDUCE` and block `ADD`.
- If total premium-at-risk exceeds `max_portfolio_risk_pct * capital`, block `ADD`.

### Close / Reduce
- Take profit: if PnL% ≥ `take_profit_pct`, suggest `CLOSE` (or `REDUCE` if >1 contract).
- Stop loss: if PnL% ≤ -`stop_loss_pct`, suggest `CLOSE`/`REDUCE` (optionally softened if
  trend strongly supports the thesis).
- Liquidity warning: if OI/volume below minimum, warn about exit/roll difficulty.

### Roll
If DTE ≤ `roll_dte_threshold`:

- If thesis still favorable (trend bullish for calls / bearish for puts): suggest `ROLL` to ~`preferred_roll_dte`.
- Else suggest `CLOSE`.

### Hold / Add
- If DTE healthy and trend supports position: suggest `HOLD`.
- Suggest `ADD` only if risk budget remains and liquidity is acceptable.
- If technicals disagree: suggest `HOLD` with caution or `REDUCE` depending on risk.

### Roll candidate selection (heuristic)
- Choose expiry nearest target DTE (e.g. 45–75 days).
- Choose strike near slightly OTM, or based on a target delta if reliably computed.

---

## Architecture (suggested layout)

- `options_helper/cli.py` (Typer commands)
- `options_helper/models.py` (Pydantic models: Portfolio, Position, RiskProfile)
- `options_helper/storage.py` (load/save JSON + validation)
- `options_helper/data/yf_client.py` (yfinance wrapper, normalization, retries)
- `options_helper/analysis/indicators.py` (SMA, RSI)
- `options_helper/analysis/greeks.py` (Black–Scholes greeks)
- `options_helper/analysis/advice.py` (rule engine + rationale)
- `options_helper/reporting.py` (Rich output formatting)
- `tests/` (unit tests for math + advice rules; no network)

## Polling / freshness strategy
- Always “best-effort” fetch; do not claim real-time.
- Default `watch` interval: 15 minutes (rate-limit friendly).
- Optional: snapshot cache to `./data/snapshots/<timestamp>.json` for debugging.

## Implementation plan (MVP)
1) Define JSON schema + validators (Portfolio/Position/RiskProfile)
2) Build CLI CRUD (`init`, `add-position`, `list`, `remove-position`)
3) Implement yfinance fetch + contract matching
4) Implement metrics (mark, PnL, DTE, IV, basic Greeks)
5) Implement technical indicators (SMA/RSI)
6) Implement advice rules + rationale output
7) Add `watch` + snapshot caching
8) Add focused unit tests (pricing/greeks/indicators + advice rules)

