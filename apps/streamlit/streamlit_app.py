from __future__ import annotations

import streamlit as st

DISCLAIMER_TEXT = (
    "Informational and educational use only. This portal is decision support and not financial advice."
)

st.set_page_config(
    page_title="options-helper Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("options-helper Portal")
st.caption(DISCLAIMER_TEXT)
st.info(
    "Read-only Streamlit scaffold for visibility and research pages. "
    "Select a page from the sidebar or use the quick links below."
)

st.subheader("Pages")
st.page_link("pages/01_Health.py", label="01 Health")
st.page_link("pages/02_Portfolio.py", label="02 Portfolio")
st.page_link("pages/03_Symbol_Explorer.py", label="03 Symbol Explorer")
st.page_link("pages/04_Flow.py", label="04 Flow")
st.page_link("pages/05_Derived_History.py", label="05 Derived History")
st.page_link("pages/06_Data_Explorer.py", label="06 Data Explorer")
st.page_link("pages/07_Market_Analysis.py", label="07 Market Analysis")
st.page_link("pages/08_Coverage.py", label="08 Coverage")
st.page_link("pages/09_SFP.py", label="09 SFP Research")
st.page_link("pages/10_MSB.py", label="10 Market Structure Breaks")
st.page_link("pages/11_0DTE_Put_Study.py", label="11 0DTE Put Study")

with st.sidebar:
    st.markdown("### Portal")
    st.caption(DISCLAIMER_TEXT)
    st.write("This scaffold is read-only. Dashboard content is added in later tasks.")
