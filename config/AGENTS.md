# config/ — Configuration & Schemas

## Purpose
This directory holds user-tunable configuration and the machine-readable contracts that validate it.

## Files
- `technical_backtesting.yaml`: default configuration used by technical indicator + backtesting subsystem.
- `technical_backtesting.schema.json`: JSON Schema for the above.

## Rules
- Keep configs backward compatible when possible.
- When you must break compatibility:
  - bump a schema/version field in the consuming code,
  - document the migration in `docs/technical_backtesting/CONFIG_SCHEMA.md`,
  - add a validation test that catches the old format.

## Schema hygiene
- All new fields should:
  - have a description,
  - have a default (or be explicitly required),
  - include bounds/enum constraints when sensible.

## Validation
- Prefer validating config at CLI startup (fail fast with a helpful error) and in tests.
