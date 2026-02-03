# Repo Improvements (Ranked)

This is the **ranked improvement backlog** for options-helper.

**Ranking heuristic:** prioritize changes that improve (1) data quality, (2) measurement/feedback loops, and (3) risk discipline.
Sorted roughly by **alpha-per-effort** first, then by strategic unlocks.

Each item has an implementation plan under `docs/plans/` (stable IDs so agents can reference them).

Plan index: `docs/plans/README.md`.

> Reminder: This project is informational/educational only — not financial advice.

---

## Ranked backlog

| Rank | ID | Improvement | Effort | Alpha potential | Why it matters | Plan |
|---:|---|---|:---:|:---:|---|---|
| 1 | IMP-001 | Spread-aware contract selection everywhere (research / roll-plan / advice / briefing) | S | High | Prevents “paper edge” that disappears at the fill; makes outputs execution-realistic. | `docs/plans/IMP-001.md` |
| 2 | IMP-012 | Earnings-aware event risk warnings + DTE gating | S | High | Avoids accidental event exposure; improves roll/entry hygiene with minimal complexity. | `docs/plans/IMP-012.md` |
| 3 | IMP-013 | Quote staleness + data quality scoring (per contract + per run) | S | High | Yahoo can be stale/zero; a quality score makes every downstream output more trustworthy. | `docs/plans/IMP-013.md` |
| 4 | IMP-002 | IV vs Realized Vol + IV percentile + term-structure slope in daily loop | M | High | Adds regime context (premium cheap/expensive) to research and position management. | `docs/plans/IMP-002.md` |
| 5 | IMP-005 | `analyze` offline-first mode (`--as-of latest --offline`) | M | Medium | Deterministic daily outputs enable evaluation and reduce Yahoo dependency. | `docs/plans/IMP-005.md` |
| 6 | IMP-004 | Portfolio-level Greeks + scenario stress (“what can hurt me fast?”) | M | Med–High | “Anti-blowup alpha”: better sizing, roll urgency, and risk budget enforcement. | `docs/plans/IMP-004.md` |
| 7 | IMP-003 | Explainable confluence/conviction score (flow + technicals + IV regime) | M | Med–High | Unifies signals into a ranked, auditable decision aid (no black box). | `docs/plans/IMP-003.md` |
| 8 | IMP-008 | Upgrade scanner to multi-factor ranking (not just extension tails) | M–L | Med–High | Reduces false positives; surfaces fewer, better candidates. | `docs/plans/IMP-008.md` |
| 9 | IMP-006 | Built-in signal journal + outcome tracking | M | High | Without feedback loops, you can’t tell which rules work *for you*. | `docs/plans/IMP-006.md` |
| 10 | IMP-014 | Artifact data contracts: schema-versioned JSON + validation + fixtures | M | Medium | Makes the system safer to evolve; enables golden tests and reliable agent workflows. | `docs/plans/IMP-014.md` |
| 11 | IMP-007 | Options backtester using snapshot history (daily resolution, spread-aware) | L | Very High | Turns the repo into an empirical research engine using your own stored chains. | `docs/plans/IMP-007.md` |
| 12 | IMP-009 | Data-provider abstraction (swap Yahoo for better data) | L | Very High | Yahoo is the limiting reagent; a provider layer is the biggest unlock. | `docs/plans/IMP-009.md` |
| 13 | IMP-020 | Canonical option contract IDs (OSI) + normalization layer | M | High | Stabilizes joins across providers/snapshots; critical for backtests and multi-leg. | `docs/plans/IMP-020.md` |
| 14 | IMP-010 | Multi-leg support (verticals / calendars / diagonals) + roll-plans | XL | Med–High | Enables structure selection by IV/skew regime and risk limits (big build). | `docs/plans/IMP-010.md` |
| 15 | IMP-017 | Engineering leverage: CI, pre-commit, type checks, perf/observability | S–M | Medium | Doesn’t create “edge” directly, but prevents regressions and speeds iteration. | `docs/plans/IMP-017.md` |
| 16 | IMP-011 | Reporting UX (TUI dashboards / web UI) | M–L | Low | Improves review speed and adoption; not an edge source by itself. | `docs/plans/IMP-011.md` |

---

## How to use this backlog
- Pick the top 1–3 items.
- Implement the plan doc.
- Add tests + docs.
- Re-run your daily workflow and confirm the outputs got more *decision-useful*.
