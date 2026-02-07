# Artifact Schemas (JSON)

This repo produces **schema-versioned JSON artifacts** intended for agents and downstream tooling.
All artifacts include:

- `schema_version` — integer schema version (start at `1`).
- `generated_at` — UTC ISO-8601 timestamp.
- `as_of` — data date (YYYY-MM-DD) for the artifact.

> Not financial advice. For informational/educational use only.

## Artifacts

### Daily briefing
- Path: `data/reports/daily/{YYYY-MM-DD}.json`
- Source: `options-helper briefing ... --write-json`
- Schema: `options_helper.schemas.briefing.BriefingArtifact`
- Notes: `sections[]` contains per-symbol summaries; nested objects are best-effort.

### Chain report
- Path: `data/reports/chains/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper chain-report ... --out ...`
- Schema: `options_helper.schemas.chain_report.ChainReportArtifact`

### Compare report
- Path: `data/reports/compare/{SYMBOL}/{FROM}_to_{TO}.json`
- Source: `options-helper compare ... --out ...`
- Schema: `options_helper.schemas.compare.CompareArtifact`

### Flow report
- Path: `data/reports/flow/{SYMBOL}/{FROM}_to_{TO}_w{N}_{group_by}.json`
- Source: `options-helper flow ... --out ...`
- Schema: `options_helper.schemas.flow.FlowArtifact`

### IV surface report
- Path: `data/reports/iv_surface/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper market-analysis iv-surface ... --out ...`
- Schema: `options_helper.schemas.iv_surface.IvSurfaceArtifact`

### Exposure report
- Path: `data/reports/exposure/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper market-analysis exposure ... --out ...`
- Schema: `options_helper.schemas.exposure.ExposureArtifact`

### Intraday flow report
- Path: `data/reports/intraday_flow/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper intraday flow ... --out ...`
- Schema: `options_helper.schemas.intraday_flow.IntradayFlowArtifact`

### Levels report
- Path: `data/reports/levels/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper market-analysis levels ... --out ...`
- Schema: `options_helper.schemas.levels.LevelsArtifact`

### Scenarios report
- Path: `data/reports/scenarios/{PORTFOLIO_DATE}/{POSITION_KEY}_{CONTRACT}.json`
- Source: `options-helper scenarios ... --out ...`
- Schema: `options_helper.schemas.scenarios.ScenariosArtifact`

### Scanner shortlist
- Path: `data/scanner/runs/{RUN_ID}/shortlist.json`
- Source: `options-helper scanner run ... --write-shortlist`
- Schema: `options_helper.schemas.scanner_shortlist.ScannerShortlistArtifact`

### Tail risk
- Path: `data/reports/tail_risk/{SYMBOL}/tail_risk_{ASOF}_h{H}_n{N}_seed{SEED}.json` (when `--out` is used)
- Source: `options-helper market-analysis tail-risk ... --out ...`
- Schema: `options_helper.schemas.tail_risk.TailRiskArtifact`

## Validation

- Offline fixtures live under `tests/fixtures/artifacts/`.
- Tests validate each fixture against its schema in `tests/test_artifact_fixtures.py`.

### Runtime validation (`--strict`)

Most commands that write JSON accept `--strict` to validate the artifact against its schema
before writing. This is useful when you want failures to surface immediately during a run.
