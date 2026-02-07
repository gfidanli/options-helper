# Diagrams (beautiful-mermaid)

This repo renders Mermaid diagrams at build time into committed SVG assets.
The docs site serves static files, so there is no browser-side Mermaid runtime.

## Layout
- Mermaid sources: `docs/diagrams/src/**/*.mmd`
- Generated SVGs: `docs/assets/diagrams/generated/**/*.svg`
- Renderer: `scripts/render_diagrams.mjs`

## Local commands
Install diagram tooling:

```bash
npm install --no-package-lock
```

Render all diagrams:

```bash
npm run render-diagrams
```

Verify generated assets are up to date:

```bash
npm run check-diagrams
```

## Add a new diagram
1. Create a Mermaid source file under `docs/diagrams/src/`.
2. Run `npm run render-diagrams`.
3. Commit both the `.mmd` source and generated `.svg`.
4. Embed in docs with Markdown, for example:

```markdown
![Ingestion pipeline](assets/diagrams/generated/ingest_pipeline.svg)
```

## Notes
- Default render theme: `github-light`.
- Generated SVGs are treated as build artifacts and are validated in docs CI.
