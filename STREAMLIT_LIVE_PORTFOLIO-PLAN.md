# Plan: Streamlit “Live Portfolio” (Alpaca streaming prices + fills)

**Generated:** 2026-02-10  
**Product intent:** Read-only, informational/educational decision support only — **not financial advice**.  
**Decisions locked (from you):**
- **Portfolio source:** `portfolio.json`
- **Streaming implementation:** **Direct websockets in Streamlit**
- **Decision layer:** **Metrics + alerts only** (no explicit HOLD/CLOSE/ROLL labels)

---

## Overview

Add a new Streamlit portal page that lets a user **start/stop** Alpaca streaming for:
- **Stock prices** (quotes/trades) for underlyings in `portfolio.json`
- **Option prices** (quotes/trades) for contracts implied by portfolio positions (single + multi-leg)
- **Fills / order updates** via Alpaca **TradingStream** (`subscribe_trade_updates`)

The page auto-refreshes using **Streamlit fragments** (`@st.fragment(run_every=...)`), and surfaces:
- Per-position **live mark**, **PnL $ / %**, **DTE**, **spread %**, **quote age (staleness)**, and **warnings**
- Per-structure **multi-leg net mark** and **net PnL** (best-effort)
- **Recent fills/order updates** table (read-only)
- Stream health: connected/alive, last event timestamps, reconnect attempts, dropped-event counters

No order placement, no portfolio file writes, no DuckDB writes. Network I/O happens **only after the user presses Start**.

---

## Prerequisites

- Install extras: `pip install -e ".[dev,ui,alpaca]"`
- Alpaca credentials available via env or `config/alpaca.env`:
  - `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`
  - Optional: `APCA_API_BASE_URL` (paper vs live inference)
  - Optional: `OH_ALPACA_STOCK_FEED`, `OH_ALPACA_OPTIONS_FEED`

---

## Public APIs / Interfaces to add

### 1) Trading updates streaming (fills)
- **New module:** `options_helper/data/streaming/alpaca_trading_stream.py`
  - `class AlpacaTradingStreamer`: wrapper around `alpaca.trading.stream.TradingStream` with lazy imports
    - `__init__(..., paper: bool | None = None, stream: Any | None = None, stream_cls: type[Any] | None = None, on_trade_updates: Callable[[Any], Any] | None = None)`
    - `subscribe_trade_updates()` (no symbols)
    - `run()` (blocking)
    - `stop()` (best effort)

- **New module:** `options_helper/data/streaming/trading_normalizers.py`
  - `normalize_trade_update(update: Any) -> dict[str, Any] | None`  
    Output fields (best-effort, stable keys):
    - `timestamp` (UTC ISO or `datetime`)
    - `event` (e.g., `fill`, `partial_fill`, `canceled`, `rejected`, …)
    - `order_id`, `symbol`, `side`, `qty`, `filled_qty`, `filled_avg_price`, `status`
    - plus optional extras if present (type, tif, limit_price, stop_price)

### 2) Compact option contract symbols for subscriptions
- **Update module:** `options_helper/analysis/osi.py`
  - Add `format_osi_compact(parsed: ParsedContract) -> str`  
    Returns *no padded spaces*, e.g. `AAPL240119C00150000`.  
  - This stays provider-agnostic; Alpaca “`.` vs `-`” normalization is applied later via `to_alpaca_symbol(...)`.

### 3) Subscription planning (shared, enforces max contracts before any subscribe)
- **New module:** `options_helper/data/streaming/subscriptions.py`
  - `@dataclass SubscriptionPlan(stocks: list[str], option_contracts: list[str], warnings: list[str], truncated: bool, truncated_count: int)`
  - `build_subscription_plan(portfolio: Portfolio, *, stream_stocks: bool, stream_options: bool, max_option_contracts: int) -> SubscriptionPlan`
    - Stocks: unique normalized underlyings
    - Options: derive contract symbols from positions/legs using `format_osi_compact` then map with `to_alpaca_symbol`
    - Enforce `max_option_contracts` **here** so LiveStreamManager never over-subscribes

### 4) Live streaming manager (Streamlit-agnostic)
- **New module:** `options_helper/data/streaming/live_manager.py`
  - `@dataclass LiveStreamConfig(...)` (explicit lists + toggles + feeds + reconnect + queue sizing)
  - `class LiveStreamManager`
    - `start(config: LiveStreamConfig) -> None` (idempotent; restart if config changes)
    - `stop() -> None`
    - `is_running() -> bool`
    - `snapshot() -> LiveSnapshot` (thread-safe copy)
  - Internals:
    - Background workers for:
      - `AlpacaStockStreamer` (quotes/trades)
      - `AlpacaOptionStreamer` (quotes/trades)
      - `AlpacaTradingStreamer` (trade updates)
    - A bounded `queue.Queue(maxsize=N)` from async callbacks → consumer
    - Backpressure policy: **never block callbacks**; on `queue.Full`, **drop** and increment `dropped_events`
    - Reconnect loop with exponential backoff (reuse `compute_backoff_seconds`)
    - Health metrics surfaced in snapshot:
      - `alive`, `last_error`, `reconnect_attempts`, `last_event_ts_by_stream`, `queue_depth`, `dropped_events`

### 5) Pure live portfolio metrics aggregation
- **New module:** `options_helper/analysis/live_portfolio_metrics.py`
  - `compute_live_position_rows(portfolio: Portfolio, live: LiveSnapshot, *, stale_after_seconds: float) -> pd.DataFrame`
  - `compute_live_multileg_rows(...) -> (structure_df, legs_df)`
  - Warnings include: missing quotes, stale quotes, wide spreads, missing legs, etc.
  - Math conventions match existing CLI:
    - Single: `pnl_abs = (mark - cost_basis) * 100 * contracts`
    - Multi-leg: `net_mark = Σ(mark_leg * signed_contracts * 100)`; `net_pnl_abs = net_mark - net_debit`

---

## Dependency graph (high-level)

```
T1 (Trading stream wrapper + normalizer) ─┐
                                         ├─ T4 (LiveStreamManager) ─┐
T2 (OSI compact formatter) ─┐            │                           ├─ T7 (Streamlit component)
                            ├─ T3 (Subscription planning) ──────────┘
                            └─ T5 (Live metrics aggregation) ────────┘
T8 (Page wiring + nav) depends on T7
T9 (Tests) depends on T1–T8
T10 (Docs) depends on T7–T8
```

---

## Tasks (dependency-aware, swarm-ready)

### T1: Alpaca TradingStream wrapper + trade-update normalizer
- **depends_on**: []
- **location**:
  - `options_helper/data/streaming/alpaca_trading_stream.py`
  - `options_helper/data/streaming/trading_normalizers.py`
  - `tests/test_alpaca_trade_update_normalizers.py`
- **description**:
  - Implement lazy-import `TradingStream` wrapper with `subscribe_trade_updates`.
  - Normalize update payloads to stable dict rows.
  - Support dependency injection (`stream_cls` / `stream`) so tests do not require `alpaca-py`.
- **validation**:
  - `pytest -k trade_update_normalizers` passes offline (no network).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T2: Add `format_osi_compact` for option contract symbols
- **depends_on**: []
- **location**:
  - `options_helper/analysis/osi.py`
  - `tests/test_osi_compact_format.py`
- **description**:
  - Add compact OSI formatter with no padded spaces.
  - Ensure `parse_contract_symbol(format_osi_compact(...))` works for roundtrip expectations.
- **validation**:
  - `pytest -k osi_compact_format` passes.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3: Subscription planner (portfolio → stocks + contracts + warnings) with max-contracts enforcement
- **depends_on**: [T2]
- **location**:
  - `options_helper/data/streaming/subscriptions.py`
  - `tests/test_streaming_subscription_plan.py`
- **description**:
  - Build stock/option subscription lists from `Portfolio` (single + multi-leg legs).
  - Map symbols through `options_helper/data/alpaca_symbols.py::to_alpaca_symbol`.
  - Enforce `max_option_contracts` before any manager starts (prevents accidental huge websocket subscriptions).
- **validation**:
  - Unit tests cover truncation + warning messaging + multileg leg enumeration.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4: LiveStreamManager (in-memory caches + worker lifecycle + backpressure)
- **depends_on**: [T1, T3]
- **location**:
  - `options_helper/data/streaming/live_manager.py`
  - `tests/test_live_stream_manager.py`
- **description**:
  - Implement manager that starts/stops 3 streams (stocks/options/fills) in background threads.
  - Async callbacks only enqueue; consumer thread updates caches.
  - Expose health snapshot (alive/reconnect/last_event_ts/queue depth/dropped events/errors).
  - Idempotent start/stop; restart when config changes.
- **validation**:
  - Tests inject stub stream classes that emit deterministic events and block until `stop()`.
  - No `streamlit` dependency; no network.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5: Pure live portfolio metrics aggregation (tables + warnings)
- **depends_on**: [T2]
- **location**:
  - `options_helper/analysis/live_portfolio_metrics.py`
  - `tests/test_live_portfolio_metrics.py`
- **description**:
  - Compute display-ready DataFrames for single + multi-leg positions using live cache snapshots.
  - Include staleness + spread % + missing data warnings.
  - Keep logic deterministic and independent of Streamlit.
- **validation**:
  - Unit tests with synthetic snapshots.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6: Streamlit component (UI controls + fragment refresh + no auto-start)
- **depends_on**: [T4, T5]
- **location**:
  - `apps/streamlit/components/live_portfolio_page.py`
  - `tests/portal/test_live_portfolio_page_smoke.py`
- **description**:
  - Sidebar controls:
    - portfolio path + reload behavior (mtime change detection or explicit Reload button)
    - enable toggles: stocks/options/fills
    - feeds: stock/options
    - refresh cadence seconds
    - stale threshold seconds
    - max option contracts
  - Store a single `LiveStreamManager` in `st.session_state`; guard against duplicate starts on rerun.
  - Start/Stop/Restart buttons:
    - Start validates portfolio load success; attempts manager start; surfaces errors in UI.
    - Stop always stops safely.
  - `@st.fragment(run_every=...)` region:
    - reads `manager.snapshot()`
    - shows health + fills + portfolio tables
  - Never call `st.*` from background threads (enforced by design).
- **validation**:
  - `pytest -k live_portfolio_page_smoke` (skips if `streamlit` not installed).
  - Manual run: `./.venv/bin/options-helper ui`, open Live page, press Start/Stop.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7: New Streamlit page + portal navigation wiring
- **depends_on**: [T6]
- **location**:
  - `apps/streamlit/pages/12_Live_Portfolio.py`
  - `apps/streamlit/streamlit_app.py`
  - `tests/portal/test_streamlit_scaffold.py`
- **description**:
  - Add a new page that calls the component renderer and includes explicit disclaimer text.
  - Add page link in `streamlit_app.py`.
  - Add the new page to scaffold smoke list so `runpy.run_path(...)` verifies “no start on import”.
- **validation**:
  - `pytest -k streamlit_scaffold` passes (or skips if Streamlit missing).
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: Error handling hardening + UX polish (stale indicators, reconnect banners)
- **depends_on**: [T6]
- **location**:
  - `options_helper/data/streaming/live_manager.py`
  - `apps/streamlit/components/live_portfolio_page.py`
- **description**:
  - UI clearly distinguishes:
    - “not started”
    - “running”
    - “reconnecting”
    - “error” (show last error string, last event time, dropped counters)
  - Add prominent stale-data banner when quote age exceeds threshold.
- **validation**:
  - Unit test for “queue full → dropped counter increments”.
  - Manual: simulate error by unsetting creds and pressing Start; verify clean message.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9: Docs (one feature per doc) + MkDocs nav update
- **depends_on**: [T7]
- **location**:
  - `docs/PORTAL_LIVE_STREAMING.md` (new)
  - `docs/PORTAL_STREAMLIT.md` (update)
  - `mkdocs.yml` (nav)
- **description**:
  - Document setup, credentials, feeds/entitlements, how to use live page, safety/limitations, and troubleshooting.
  - Explicitly state:
    - read-only (no orders submitted),
    - best-effort data quality,
    - not financial advice,
    - troubleshooting for missing OPRA/IEX/SIP permissions.
- **validation**:
  - `mkdocs build` (optional) and nav includes the new doc.
- **status**: Not Completed
- **log**:
- **files edited/created**:

---

## Parallel execution groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T2 | Immediately |
| 2 | T3 | T2 complete |
| 2 | T5 | T2 complete |
| 3 | T4 | T1 + T3 complete |
| 4 | T6 | T4 + T5 complete |
| 5 | T7, T8 | T6 complete |
| 6 | T9 | T7 complete |

---

## Testing strategy (deterministic, offline)

- Unit tests (no Streamlit, no Alpaca network):
  - contract symbol formatting
  - trade update normalization
  - subscription plan truncation/warnings
  - live manager queue/backpressure + start/stop + stub streams
  - metrics aggregation for single + multi-leg
- Portal tests (guarded by `pytest.importorskip("streamlit")`):
  - import smoke + page runpy smoke (ensures default streaming off, no credential requirement on import)

---

## Risks & mitigations

- **Streamlit reruns spawning duplicate threads** → store a single manager in `st.session_state`, and make `start()` idempotent.
- **Queue overload / memory growth** → bounded queue + drop counter + stale banner.
- **Entitlements/feeds causing silent no-data** → surface “running but no events” health state; show last-event timestamps per stream.
- **Missing/invalid portfolio file** → disable Start until portfolio loads; show explicit error.
- **Optional dependencies** → all Alpaca imports remain lazy; tests inject stubs.

---

## Acceptance criteria

- From Streamlit, user can:
  - load `portfolio.json`
  - press **Start** and see live updating stock/option marks + PnL + staleness indicators
  - see **recent fills/order updates** populate (when the account has activity)
  - press **Stop** and streams stop cleanly (health shows stopped; no duplicate threads on rerun)
- Default page load (and `runpy` smoke) does **not** start streaming or require Alpaca credentials.
- Docs clearly state read-only + not-financial-advice + best-effort data.

---

## Assumptions / defaults

- **Paper vs live for TradingStream:** default “auto” inferred from `APCA_API_BASE_URL` containing `paper`; UI exposes an override toggle.
- Default refresh cadence: `2s`; default stale threshold: `30s`; default max option contracts: `250`.
- Options streaming uses **quotes/trades only** (no option bars streaming per current Alpaca SDK limitations).
