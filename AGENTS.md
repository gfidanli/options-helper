# options-helper — Agent Instructions (Repo Root)

## Mission
Build a CLI-first system that:
- Collects and caches **underlying candles**, **option chain snapshots**, and derived metrics.
- Surfaces **actionable insights** for options position management (entries, adds, rolls, exits).
- Stays fast, testable, and robust despite `yfinance`/Yahoo quirks.

This repo is an information/decision-support tool. Keep outputs and docs clear that it is **not financial advice**.

## Quick commands
- Install: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
- Run CLI (preferred): `./.venv/bin/options-helper --help`
- Run tests: `./.venv/bin/python -m pytest`

## Git workflow
- Make **regular, logical commits** when a unit of work is complete and tests pass.
- Prefer small commits with clear messages over large “mega-commits”.
- Don’t commit local state (e.g. `data/`, `portfolio.json`).

## Local state (do not commit)
- `portfolio.json` and everything under `data/` are gitignored.
- Prefer adding fixtures under `tests/` instead of committing real market data.

## Architecture expectations
- CLI entrypoint lives in `options_helper/cli.py` and should be thin (wiring only).
- Command implementations should live in `options_helper/commands/` and be registered from `options_helper/cli.py`.
- Provider/store construction should go through `options_helper/cli_deps.py` so tests can monkeypatch a stable seam.
- External I/O (yfinance, filesystem caches) lives under `options_helper/data/`.
- Pure computations/heuristics live under `options_helper/analysis/` (no network calls).
- Each new feature should ship with:
  - a CLI entrypoint (if user-facing),
  - docs (one feature per doc in `docs/`),
  - tests (offline, deterministic).

## Data reliability rules
- `yfinance` data is “best effort”; assume missing fields, zeros, stale quotes, and inconsistent timezones.
- Prefer cached daily candles as the “source of truth” for technical indicators.
- When saving snapshots, date folders by the **data period** (latest candle date), not wall-clock run time.

## Product direction
- Optimize for **position management** and **signal clarity** over complex modeling.
- Start with simple rules and measurable outputs; iterate with tests and docs.
- Track next ideas in `FEATURE_IDEAS.md`.

## Planning artifacts
- Ranked improvements: `docs/REPO_IMPROVEMENTS.md`
- Feature PRDs/milestones: `docs/BACKLOG.md`
- Implementation plans: `docs/plans/`
- Iteration loop prompt: `docs/LLM_LOOP_PROMPT.md`

## Plan hygiene
- Before starting an IMP, check recent commits for it (`git log --grep "IMP-XXX"`), since plan docs/status may lag.

## Completion sound (Terminal.app)
- When you finish a task and are **waiting for user input**, play the macOS “Glass” sound via: `afplay /System/Library/Sounds/Glass.aiff`
- Don’t rely on the terminal bell (`\a`) for notifications (it’s disabled/muted in this setup).
