# Live Portfolio Streaming Page (Read-only)

Status: Implemented

This page provides live monitoring for an existing options portfolio using Alpaca streams. It is **read-only**: the page does **not** submit orders, does not write portfolio state, and is for informational/educational use only. **Not financial advice.**

Market data and trade updates are **best effort**. Expect occasional gaps, stale quotes, delayed updates, or missing fields depending on feed entitlements, market state, and upstream provider behavior.

## What it shows

- Stream health: status, queue depth, dropped events, reconnect attempts, last event timestamps, and last error.
- Recent fills/order updates from Alpaca `TradingStream` (`subscribe_trade_updates`).
- Live position tables for:
  - single-leg options (mark, PnL, spread %, quote age, warnings)
  - multi-leg structures (net mark, net PnL, quote-age max, warnings)
  - multi-leg legs (leg-level marks and warnings)

## Prerequisites

Install UI + Alpaca extras:

```bash
pip install -e ".[dev,ui,alpaca]"
```

Required credentials:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

Optional:

- `APCA_API_BASE_URL` (paper/live endpoint selection)
- `OH_ALPACA_STOCK_FEED` (default stock feed when page selector is `Auto`)
- `OH_ALPACA_OPTIONS_FEED` (default options feed when page selector is `Auto`)

Recommended local secret file:

1. Copy `config/alpaca.env.example` to `config/alpaca.env`.
2. Set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`.

Launch the portal:

```bash
./.venv/bin/options-helper ui
```

Then open page `13 Live Portfolio` from the Streamlit sidebar.

## Feed selection and entitlements

The Live Portfolio page lets you choose feeds in the sidebar or use `Auto (env/default)`:

- Stock feed options: `iex`, `sip`, `delayed_sip`
- Options feed options: `opra`, `indicative`

Feed behavior:

- UI selector value takes precedence.
- If selector is `Auto`, the app uses `OH_ALPACA_STOCK_FEED` / `OH_ALPACA_OPTIONS_FEED` when set.
- If no explicit feed is set, Alpaca SDK defaults apply.

Entitlement notes:

- `IEX` is typically available on free stock data plans.
- `SIP` usually requires paid stock-data entitlement.
- `OPRA` options data usually requires options market-data entitlement.
- `indicative` can be useful when full OPRA access is unavailable, but coverage/latency may differ.

## How to use the page

1. Set `Portfolio JSON path` (default `portfolio.json`) and click `Reload Portfolio` if needed.
2. Choose which streams to run:
   - `Stream stocks`
   - `Stream options`
   - `Stream fills`
3. Select stock/options feeds (or keep `Auto`).
4. Set operational controls:
   - `Refresh cadence (seconds)`
   - `Stale threshold (seconds)`
   - `Max option contracts`
5. Press `Start` to begin streaming.
6. Watch:
   - `Stream Health` for running/reconnecting/error state
   - `Recent Fills / Order Updates`
   - `Live Position Metrics`
7. Press `Stop` to halt background stream workers.

`Start` is disabled until a valid portfolio is loaded. Streaming does not start on page import/load; it only starts after pressing `Start`.

## Safety and limitations

- **Read-only:** no order submission, no position mutation, no `portfolio.json` writes, no DuckDB writes.
- **Best effort data quality:** quotes/trades can be missing, stale, delayed, or incomplete.
- **Subscription cap:** `Max option contracts` enforces a pre-subscribe cap; when truncated, warnings are shown.
- **Backpressure policy:** callback ingestion is non-blocking; when event queue is full, events are dropped and counted.
- **Not financial advice:** outputs are decision-support telemetry only.

## Troubleshooting

### Missing credentials

If Start fails with a message like `Missing Alpaca credentials`, set:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

Then restart Streamlit.

### No data due to feed entitlements (OPRA/IEX/SIP)

Symptoms:

- stream status is `reconnecting` or `error`
- `last_error` shows auth/permission/feed-related failures
- little or no quote/trade updates for enabled streams

Common causes and actions:

- Stock feed set to `sip` without SIP entitlement:
  - switch to `iex` (or `delayed_sip`) and press `Restart`
  - or enable SIP entitlement in Alpaca account settings
- Options feed set to `opra` without OPRA entitlement:
  - switch to `indicative` and press `Restart`
  - or enable OPRA entitlement in Alpaca account settings
- Permission-denied responses (often HTTP `403` / `402`) indicate plan/entitlement mismatch for the chosen feed.

### Running but little/no updates

- Confirm market session/activity for subscribed symbols.
- Confirm portfolio symbols/contracts are valid and not truncated by `Max option contracts`.
- Check per-stream `last_event_ts` and `dropped_events` in the health table.

### Stale-data banner is red

- Increase `Stale threshold (seconds)` only if your expected latency is higher.
- Verify feed entitlements and network stability.
- If `dropped_events` grows quickly, reduce symbol/contract load or restart.

### Portfolio load/start issues

- If Start is disabled, fix portfolio path/JSON parse errors first.
- Use `Reload Portfolio` after edits.

## Related docs

- Streamlit portal overview: `docs/PORTAL_STREAMLIT.md`
- Provider setup and feed env vars: `docs/PROVIDERS.md`
- Alpaca rate-limit behavior: `docs/ALPACA_RATE_LIMITS.md`
