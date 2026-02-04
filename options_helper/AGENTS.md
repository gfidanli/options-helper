# options_helper/ — Package Conventions

## Style
- Prefer clear, typed Python (`from __future__ import annotations`).
- Keep functions small; avoid cleverness.
- Favor dataclasses for internal structs and Pydantic models for persisted/validated schemas.

## CLI conventions
- `options_helper/cli.py` should be “wiring” only (app setup + shared options + command registration).
- New commands/command-groups should live in `options_helper/commands/*` and be registered from `options_helper/cli.py`.
- Keep CLI logic thin: parse args, call analysis/data modules, render output.
- Use `rich` tables for human output; keep widths reasonable (`COLUMNS=200` friendly).
- Persist user-facing outputs under `data/` (gitignored).
- Prefer dependency builders from `options_helper/cli_deps.py` and import as a module
  (`import options_helper.cli_deps as cli_deps`) so tests can monkeypatch a stable seam.

## Error handling
- Network/data failures must not crash the entire run when avoidable.
- Prefer per-symbol warnings and continue.
- Raise `typer.BadParameter` for user input issues.

## Dependencies
- Avoid adding heavy deps unless a clear feature requires it.
- Any new external API should be wrapped behind a client in `options_helper/data/`.
