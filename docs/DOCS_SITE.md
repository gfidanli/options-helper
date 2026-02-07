# Documentation site (MkDocs + GitHub Pages)

This repo’s docs live in `docs/` and can be rendered as a static website using [MkDocs](https://www.mkdocs.org/) + the Material theme.

## Local preview

1. Install dependencies:

```bash
pip install -e ".[dev,docs]"
npm install --no-package-lock
```

2. Validate diagram artifacts:

```bash
npm run check-diagrams
```

3. Run the live-reload docs server:

```bash
mkdocs serve
```

Then open the printed local URL (typically `http://127.0.0.1:8000/`).

## Build (static output)

```bash
npm run check-diagrams
mkdocs build
```

The output goes to `site/` (gitignored).

## GitHub Pages

This repo includes a GitHub Actions workflow that builds the site and deploys it to GitHub Pages on pushes to `main`.

One-time setup in GitHub:

1. Repo Settings → Pages
2. Source: **GitHub Actions**

After that, the published URL will show up in the Pages settings and workflow logs.
