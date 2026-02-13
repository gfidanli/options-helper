# docs/ — Documentation Conventions

## One feature per doc
Keep docs modular (separate file per logical feature), e.g.:
- `docs/CANDLE_CACHE.md`
- `docs/OPTIONS_FLOW.md`
- `docs/RESEARCH.md`
- `docs/WATCHLISTS.md`

## Update triggers
When changing behavior, also update docs:
- CLI flags/defaults
- data layouts and filenames
- cron behavior
- known limitations and “best effort” caveats

## Governance docs
- `docs/TECH_DEBT_GUARDRAILS.md` is the canonical anti-debt policy.
- `docs/plans/` is the canonical plan directory.
- Keep legacy plan migrations indexed in `docs/plans/LEGACY_INDEX.md`.

## Content guidelines
- Prefer concrete examples (commands + output paths).
- Call out data assumptions (e.g., “uses latest daily close”, “Yahoo can be stale”).
- Avoid long walls of text; use short sections and bullet lists.
