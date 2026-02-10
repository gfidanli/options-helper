from __future__ import annotations

import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from apps.streamlit.components.live_portfolio_page import render_live_portfolio_page

DISCLAIMER_TEXT = "Informational and educational use only. Not financial advice."

render_live_portfolio_page()

with st.sidebar:
    st.caption(DISCLAIMER_TEXT)
