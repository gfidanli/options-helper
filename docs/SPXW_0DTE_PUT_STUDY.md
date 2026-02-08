# SPXW 0DTE Put Study (SPY Proxy)

This workflow is informational decision support only and **not financial advice**.

## What This Study Does
- Estimates conditional breach probability for downside close thresholds on same-day (`0DTE`) puts.
- Maps risk tiers (for example `<=1%`, `<=5%`) to strike candidates.
- Tracks walk-forward and forward-test outcomes with deterministic artifact outputs.

## Scope And Proxy Caveat
- Default proxy underlying is `SPY`.
- Target exposure label is `SPXW`.
- Results are a **SPY proxy**, not true SPX index/SPXW microstructure.
- SPY proxy caveat must be treated as first-class in CLI output, docs, and dashboard interpretation.

## Not Financial Advice
- Outputs are for research, diagnostics, and process support.
- They are not investment recommendations, trade instructions, or suitability analysis.
- Users are responsible for independent risk controls, execution assumptions, and broker/market constraints.

## Locked Defaults
| Setting | Locked Default |
|---|---|
| Proxy underlying | `SPY` |
| Target underlying label | `SPXW` |
| Benchmark decision mode | `fixed_time` |
| Benchmark fixed decision time (ET) | `10:30` |
| Supported decision modes | `fixed_time`, `rolling` |
| Rolling interval (minutes) | `15` |
| Fill model default | `bid` |
| Risk tiers (max breach probability) | `0.5%`, `1%`, `2%`, `5%` |
| Exit modes (always tracked) | `hold_to_close`, `adaptive_exit` |
| Position concurrency caps | `max_open_positions_per_symbol=1`, `max_open_positions_total=1` |
| Decision bar policy | `bar_close_snapshot` |
| Entry anchor policy | `next_tradable_bar_or_quote_after_decision` |
| Settlement rule | `same_day_close_intrinsic_proxy` |

## Methodology Summary
1. Build intraday state rows at decision timestamps (`fixed_time` or `rolling`).
2. Compute features using only bars known at decision time.
3. Build labels using close-known events with explicit entry-anchor semantics.
4. Resolve strike/premium snapshots with quote-quality gating.
5. Fit conditional close-tail model with shrinkage and score breach probabilities.
6. Apply policy ranking by risk tier, then simulate outcome tracks (`hold_to_close`, `adaptive_exit`).
7. Run walk-forward splits with `trained_through_session < session_date`.
8. Persist study artifacts and frozen active model for forward-only scoring.

## Anti-Lookahead Assumptions (Hard Rules)
- Decision features may use only data at or before `decision_ts`.
- If a signal is known only at bar close, entry anchor must be the **next tradable bar/quote**.
- No same-bar entry anchoring for close-confirmed decisions.
- If no next tradable anchor exists, the row fails closed with `skip_reason=no_entry_anchor`.
- Walk-forward and forward snapshot scoring require frozen model state from prior sessions only.

## CLI Usage
Run study/backtest artifact generation:

```bash
./.venv/bin/options-helper market-analysis zero-dte-put-study \
  --symbol SPY \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --decision-mode fixed_time \
  --decision-times 10:30 \
  --risk-tiers 0.005,0.01,0.02,0.05 \
  --strike-grid -0.03,-0.02,-0.015,-0.01,-0.005 \
  --fill-model bid \
  --out data/reports
```

Run forward snapshot scoring from frozen active model:

```bash
./.venv/bin/options-helper market-analysis zero-dte-put-forward-snapshot \
  --symbol SPY \
  --out data/reports
```

## Artifact Outputs
- Study artifact JSON: `data/reports/zero_dte_put_study/{SYMBOL}/{AS_OF}.json`
- Active model: `data/reports/zero_dte_put_study/{SYMBOL}/active_model.json`
- Forward snapshots JSONL: `data/reports/zero_dte_put_study/{SYMBOL}/forward_snapshots.jsonl`

## Dashboard Interpretation
- Probability surface: conditional breach probabilities by strike return and decision context.
- Strike table: ranked candidates by risk tier with quality-gated premium assumptions.
- Walk-forward summary: aggregate `hold_to_close` and `adaptive_exit` outcome summaries.
- Calibration: forward finalized rows only; pending rows are excluded by design.
- Forward snapshots: latest scored rows with reconciliation state (`pending_close` vs `finalized`).

## Assumptions And Caveats
- Quote quality is best effort and can force skips/fallbacks (`stale`, `wide`, `crossed`, `zero_bid`).
- Fill assumptions (`bid`/slippage/fees) materially affect simulation outputs.
- SPY proxy behavior can diverge from true SPX/SPXW settlement and liquidity dynamics.
- Early-close/partial sessions can reduce coverage and increase skip rates.

## Acceptance Criteria
1. Schema defaults validate to the locked values above.
2. Study and forward outputs include SPY-proxy and not-financial-advice disclaimers.
3. Anchor fields (`decision_ts`, `entry_anchor_ts`, `close_label_ts`) are present and causal.
4. Walk-forward model snapshots enforce `trained_through_session < session_date`.
5. Forward snapshot upserts are idempotent on deterministic key fields.
