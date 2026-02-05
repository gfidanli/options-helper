# MkDocs nav patch (add the new docs)

Add the new docs pages to `mkdocs.yml` so they show up in the site navigation.

## New files added by this pack

- `docs/NEXT_STAGE_VISIBILITY_PORTAL.md`
- `docs/OBSERVABILITY.md`
- `docs/PORTAL_STREAMLIT.md`
- `docs/PORTAL_STOCKPEERS_STYLE.md`
- `docs/DAGSTER_OPTIONAL.md`
- `docs/plans/VISIBILITY_PORTAL_IMPLEMENTATION_PLAN.md`

## Suggested `mkdocs.yml` snippet

Place this under your existing `nav:` section (adjust the grouping to match your current style):

```yaml
nav:
  # ... existing entries ...

  - Platform:
      - Next stage (visibility + portal): NEXT_STAGE_VISIBILITY_PORTAL.md
      - Observability (run ledger + checks): OBSERVABILITY.md
      - Portal (Streamlit): PORTAL_STREAMLIT.md
      - Portal styling (Stockpeers): PORTAL_STOCKPEERS_STYLE.md
      - Dagster (optional): DAGSTER_OPTIONAL.md

  - Plans:
      - Visibility + portal implementation plan: plans/VISIBILITY_PORTAL_IMPLEMENTATION_PLAN.md
```

## Notes
- Keep filenames stable once published; MkDocs links and bookmarks will depend on them.
- If you prefer, you can place these under your existing “Roadmap” or “Architecture” groups instead of adding a new “Platform” section.
