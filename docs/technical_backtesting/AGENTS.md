# docs/technical_backtesting/ — Documentation Conventions

This folder is the long-form design + runbook for the `technicals_backtesting` subsystem.

## What belongs here
- PRD and architecture decisions
- data contracts and schema notes
- indicator definitions and assumptions
- runbooks for common workflows (backtest, walk-forward, optimization)

## Change rules
- If you change indicator semantics or defaults, update:
  - `INDICATORS.md`
  - `CONFIG_SCHEMA.md`
  - the JSON schema (`config/technical_backtesting.schema.json`) if relevant
  - at least one test that would fail under the old behavior

## Writing style
- Bias toward **concrete**:
  - commands to run
  - expected outputs / file paths
  - examples of config snippets
- Keep it modular: prefer adding a new doc over making one doc huge.

## LLM friendliness
- Prefer explicit definitions, numbered steps, and stable headings.
- When describing a pipeline, include:
  - inputs
  - outputs
  - invariants
  - failure modes
