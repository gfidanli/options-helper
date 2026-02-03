# Plans Index

This directory contains **implementation plans** for the ranked improvement backlog.

- Ranked backlog: `docs/REPO_IMPROVEMENTS.md`
- Plan template: `_TEMPLATE.md`

These plans are written to be **LLM-agent friendly**: clear goals, bounded scope, specific file touchpoints, and test/acceptance criteria.

## Status legend

- **draft** — still being designed; may have open questions or missing test/acceptance detail.
- **ready** — implementable as-written; remaining decisions are minor/obvious.
- **in-progress** — currently being implemented (keep scope tight; finish tests/docs).
- **done** — implemented and shipped; plan remains as historical record.
- **deferred** — intentionally paused (usually due to effort/complexity or dependency).

> Tip: you can add qualifiers like `draft (large)` for big/strategic items where the plan is intentionally higher level.

## Current plans (sorted by backlog rank)

| Rank | ID | Plan | Status | Effort | Alpha potential |
|---:|---|---|---|:---:|:---:|
| 1 | IMP-001 | Spread-aware contract selection everywhere ([IMP-001](./IMP-001.md)) | done | S | High |
| 2 | IMP-012 | Earnings-aware event risk warnings + DTE gating ([IMP-012](./IMP-012.md)) | done | S | High |
| 3 | IMP-013 | Quote staleness + data quality scoring ([IMP-013](./IMP-013.md)) | done | S | High |
| 4 | IMP-002 | IV vs Realized Vol + IV percentile + term structure in the daily loop ([IMP-002](./IMP-002.md)) | done | M | High |
| 5 | IMP-005 | `analyze` offline-first mode ([IMP-005](./IMP-005.md)) | done | M | Medium |
| 6 | IMP-004 | Portfolio-level Greeks + scenario stress ([IMP-004](./IMP-004.md)) | done | M | Med–High |
| 7 | IMP-003 | Explainable confluence/conviction score ([IMP-003](./IMP-003.md)) | done | M | Med–High |
| 8 | IMP-008 | Multi-factor scanner ranking ([IMP-008](./IMP-008.md)) | ready | M–L | Med–High |
| 9 | IMP-006 | Signal journal + outcome tracking ([IMP-006](./IMP-006.md)) | ready | M | High |
| 10 | IMP-014 | Artifact data contracts (schema-versioned JSON + validation) ([IMP-014](./IMP-014.md)) | ready | M | Medium |
| 11 | IMP-007 | Options backtester using snapshot history (daily) ([IMP-007](./IMP-007.md)) | draft (large) | L | Very High |
| 12 | IMP-009 | Data-provider abstraction (swap Yahoo for better data) ([IMP-009](./IMP-009.md)) | draft | L | Very High |
| 13 | IMP-020 | Canonical option contract IDs (OSI) + normalization layer ([IMP-020](./IMP-020.md)) | ready | M | High |
| 14 | IMP-010 | Multi-leg support (verticals/calendars/diagonals) + roll-plans ([IMP-010](./IMP-010.md)) | draft (large) | XL | Med–High |
| 15 | IMP-017 | Engineering leverage: CI, pre-commit, type checks, perf/observability ([IMP-017](./IMP-017.md)) | ready | S–M | Medium |
| 16 | IMP-011 | Reporting UX (TUI dashboards / web UI) ([IMP-011](./IMP-011.md)) | draft | M–L | Low |

## Workflow for agents

1. Pick the top 1–3 items in `docs/REPO_IMPROVEMENTS.md`.
2. Read the plan doc(s) end-to-end.
3. Implement in small, testable increments.
4. Update or add tests (offline + deterministic).
5. Update docs and artifacts schemas if outputs changed.

## Updating plan status

When you start/finish a plan, update the header line in the plan doc, e.g.:

```md
- **Status:** in-progress
```

If an item’s *rank* changes, update `docs/REPO_IMPROVEMENTS.md` to keep the backlog and this index consistent.
