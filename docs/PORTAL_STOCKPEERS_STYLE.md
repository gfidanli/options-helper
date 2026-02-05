# Portal styling: Stockpeers look & feel (Theme + Layout patterns)

Status: Draft  
Goal: emulate the clean “product-y” vibe of the Streamlit Stockpeers demo.

This repo includes the Stockpeers theme config at:

- `.streamlit/config.toml`

Upstream demo uses Streamlit `>= 1.44.2` and Altair for charts.  
If you want the theme keys like `baseFontWeight` / `headingFontWeights`, you should use a modern Streamlit version.

---

## 1) Theme (copy/paste)

Use the included file:

- `.streamlit/config.toml`

### Notes
- Theme is **dark** base with a strong primary accent and high-contrast panels.
- It sets a custom font:
  - `Space Grotesk` loaded from Google Fonts
- It enables widget borders so controls feel “app-like”.

If you run Streamlit from the repo root, it will pick up `.streamlit/config.toml` automatically.

---

## 2) Layout patterns that make it feel like Stockpeers

### Pattern A — Avoid `st.sidebar` for the main filter panel
Stockpeers uses a 2-column layout:

- Left: a narrow “filters” panel
- Right: the main dashboard canvas

Example:

```python
import streamlit as st

cols = st.columns([1, 3])

filter_panel = cols[0].container(border=True, height="stretch", vertical_alignment="center")
main_panel   = cols[1].container()

with filter_panel:
    st.write("Filters")
    symbols = st.multiselect(
        "Symbols",
        options=["AAPL", "MSFT", "NVDA"],
        default=["AAPL", "MSFT"],
        placeholder="Choose symbols…",
        accept_new_options=True,
    )

with main_panel:
    st.subheader("Overview")
    st.write("…charts…")
```

This pattern:
- keeps filters visible without collapsing
- looks more like a “real app” than a sidebar

### Pattern B — Border containers for “cards”
Use:
- `st.container(border=True)` and/or `st.columns(...)` to build card-like blocks

### Pattern C — Title + short subtitle + whitespace
Stockpeers does:

- `st.set_page_config(layout="wide")`
- a big title line
- a short, plain-English subtitle
- then a blank line for spacing

A similar pattern is good for the Portal pages.

---

## 3) Query params (bookmarkable dashboards)

Stockpeers uses `st.query_params` so the selection is shareable as a URL.

Recommended for the Portal:
- `?symbols=AAPL,MSFT,NVDA`
- `?as_of=latest`

Pattern:

```python
def list_to_csv(xs: list[str]) -> str:
    return ",".join(xs)

default = ["AAPL", "MSFT"]

if "symbols" not in st.session_state:
    st.session_state.symbols = st.query_params.get("symbols", list_to_csv(default)).split(",")

def sync_query_params():
    if st.session_state.symbols:
        st.query_params["symbols"] = list_to_csv(st.session_state.symbols)
    else:
        st.query_params.pop("symbols", None)

st.multiselect(
    "Symbols",
    options=sorted(set(default + st.session_state.symbols)),
    default=st.session_state.symbols,
    key="symbols",
    on_change=sync_query_params,
    accept_new_options=True,
)
```

For `options-helper`, this is especially useful for:
- watchlists
- portfolio underlyings
- “as-of date” navigation

---

## 4) Charting choices

To keep a Stockpeers-like vibe:
- Prefer **Altair** for standard time-series and bar charts.
- Always use `use_container_width=True`.
- Keep chart count low: 2–4 strong charts per page.

For option-chain style plots:
- a strike-vs-OI chart
- an IV smile chart
- a flow heatmap (strike × expiry) if the data supports it

---

## 5) Table styling

The theme includes:
- `dataframeHeaderBackgroundColor` matching the background

Use:
- `st.dataframe(..., use_container_width=True)`
- show only the columns users care about; hide raw IDs behind “Details” expanders.

---

## 6) Recommended Streamlit version pin

Stockpeers uses:

- `streamlit>=1.44.2`

If you pin lower than that, some theme keys may be ignored.
