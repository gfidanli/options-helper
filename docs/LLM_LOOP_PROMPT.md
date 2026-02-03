# LLM Iteration Loop Prompt (copy/paste)

Use this prompt to continuously refine options-helper's product definition and backlog.

---

## Context
- This repo is an **options position management + research workbench** (CLI-first).
- It relies on daily candle caching + daily option chain snapshots so most analysis is offline.
- Informational/educational only — **not financial advice**.

## Inputs
1) Current repo state (you have access to the codebase).
2) Current docs:
   - `README.md`
   - `docs/BACKLOG.md`
   - `docs/REPO_IMPROVEMENTS.md`
   - `FEATURE_IDEAS.md`
   - `AGENTS.md` (and per-folder AGENTS)
3) My workflow constraints (timezone, daily after-close routine, offline-first preference).

## Your tasks (repeatable loop)
### A) Validate and update the product narrative
- Audit the repo and update (in-place) the following docs if they are out of sync:
  - **SUMMARY** (what it is, what it does)
  - **USER ACTIONS** (how a user actually uses it)
  - **USER ROUTINES** (cron + after-close workflows)
- Call out any mismatches between docs and reality.

### B) Improve the ranked backlog
- Read `docs/REPO_IMPROVEMENTS.md`.
- Propose edits:
  - reorder items based on current code and impact
  - add missing but realistic items
  - remove/merge duplicates
- Keep ranking roughly by "alpha per unit effort".

### C) For the top 1–3 items
For each selected improvement:
- Update/create its plan doc under `docs/plans/IMP-XXX.md` using the repo's plan template style.
- Ensure the plan includes:
  - goals/non-goals
  - user-facing changes
  - implementation steps with files to touch
  - test plan + acceptance criteria

### D) Implementation (optional, if requested)
If I ask you to implement work:
- Follow the relevant plan doc.
- Keep changes small and testable.
- Add/adjust tests (offline + deterministic).
- Update docs.

## Output format
1) A concise changelog of what you updated.
2) Updated sections (or file diffs) for docs.
3) If you added plan docs, list them and summarize what changed.

## Constraints
- No network calls in tests.
- Be explicit about data quality caveats (Yahoo/yfinance is best-effort).
- Prefer offline/deterministic workflows.
- Keep everything educational (not financial advice).
