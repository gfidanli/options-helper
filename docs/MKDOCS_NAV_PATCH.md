# MkDocs nav patch (visibility + portal docs)

Status: Applied in `mkdocs.yml`.

The docs site now includes a dedicated navigation section for visibility/portal/orchestration content.

## Pages included

- `docs/NEXT_STAGE_VISIBILITY_PORTAL.md`
- `docs/OBSERVABILITY.md`
- `docs/PORTAL_STREAMLIT.md`
- `docs/DAGSTER_OPTIONAL.md`

## Current `mkdocs.yml` section

```yaml
- Visibility & Portal:
    - Next Stage Overview: NEXT_STAGE_VISIBILITY_PORTAL.md
    - Observability + Health: OBSERVABILITY.md
    - Streamlit Portal: PORTAL_STREAMLIT.md
    - Dagster (Optional): DAGSTER_OPTIONAL.md
```

## Notes

- This section is documentation-only; runtime behavior is unchanged.
- Keep filenames stable to avoid breaking bookmarks and external links.
