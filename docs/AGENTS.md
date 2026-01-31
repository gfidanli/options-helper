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

## Content guidelines
- Prefer concrete examples (commands + output paths).
- Call out data assumptions (e.g., “uses latest daily close”, “Yahoo can be stale”).
- Avoid long walls of text; use short sections and bullet lists.

