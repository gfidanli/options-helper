# scripts/ — Automation & Cron

## Goals
- Make daily jobs **idempotent**, **safe**, and **observable**.
- Prefer logging to `data/logs/` (gitignored) with timestamps.

## Shell script rules
- Use `#!/usr/bin/env bash` + `set -euo pipefail`.
- Resolve paths relative to the repo root (don’t assume current working dir).
- Always call the project venv binary (`.venv/bin/options-helper`).
- Never require interactive input from cron.

## macOS considerations
- `cron` won’t run while the machine is asleep.
- Jobs referencing `/Volumes/...` will fail if the volume isn’t mounted.
- For reliability, consider a future migration to `launchd` LaunchAgents.

