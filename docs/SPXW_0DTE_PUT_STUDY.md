# SPXW 0DTE Put Study (SPY Proxy)

This workflow is informational decision support only and **not financial advice**.

## Scope
- Product focus: same-day (`0DTE`) put-sell research for SPXW-style exposure.
- Proxy default: `SPY` intraday state is used as a proxy for SPX/SPXW conditions.
- Caveat: SPY and SPX/SPXW can diverge in microstructure, index composition effects, and settlement behavior.

## Locked Defaults (T0)
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

## Anti-Lookahead Contract
- Decision-time features may use only data known at/through `decision_ts`.
- If decision is computed on a completed bar, tradable entry is anchored at the next tradable bar/quote timestamp.
- If no tradable entry anchor exists, rows must set `skip_reason=no_entry_anchor`.

## Acceptance Criteria (Measurable)
1. Schema defaults validate to the exact locked values listed above.
2. Artifact metadata contains explicit informational-use and SPY-proxy caveat fields.
3. Probability, strike-ladder, and simulation rows all require anchor metadata (`session_date`, `decision_ts`, `decision_bar_completed_ts`, `close_label_ts`, `entry_anchor` policy).
4. `quote_quality_status` accepts only the locked enum values; invalid statuses fail schema validation.
5. `skip_reason` supports `no_entry_anchor`; rows with missing `entry_anchor_ts` must use that skip reason.
6. Study artifact JSON round-trips (`model_dump_json` -> `model_validate_json`) without semantic drift.
7. Consumers can rely on dual exit-mode support (`hold_to_close` and `adaptive_exit`) from defaults.

