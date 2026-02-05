from __future__ import annotations

import pandas as pd
import streamlit as st

from apps.streamlit.components.data_explorer_page import (
    build_select_sql,
    list_database_schemas,
    list_database_tables,
    load_table_columns,
    preview_table_rows,
)


def _fmt_row_count(value: object) -> str:
    try:
        if value is None:
            return "-"
        return f"{int(value):,d}"
    except (TypeError, ValueError):
        return "-"


st.title("Data Explorer")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only debug view for DuckDB schemas, tables, and sample rows.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input("DuckDB path", value="")
    preview_limit = st.slider("Preview row limit", min_value=10, max_value=200, value=50, step=10)

database_arg = database_path or None
schemas, schemas_note = list_database_schemas(database_path=database_arg)
if schemas_note:
    st.warning(f"Schemas unavailable: {schemas_note}")
if not schemas:
    st.info("No schemas found. Initialize and populate DuckDB first.")
    st.stop()

default_schema = "main" if "main" in schemas else schemas[0]
selected_schema = st.selectbox(
    "Schema",
    options=schemas,
    index=schemas.index(default_schema),
)

tables_df, tables_note = list_database_tables(schema=selected_schema, database_path=database_arg)
if tables_note:
    st.warning(f"Tables unavailable: {tables_note}")
if tables_df.empty:
    st.info(f"No tables found in schema `{selected_schema}`.")
    st.stop()

st.subheader("Tables")
display_tables = tables_df.copy()
display_tables["row_count"] = display_tables["row_count"].map(_fmt_row_count)
st.dataframe(display_tables, hide_index=True, use_container_width=True)

table_names = tables_df["table_name"].astype("string").tolist()
selected_table = st.selectbox(
    "Table",
    options=table_names,
    index=0,
    help="Select a table/view to inspect schema and rows.",
)

columns_df, columns_note = load_table_columns(
    selected_schema,
    selected_table,
    database_path=database_arg,
)
if columns_note:
    st.warning(f"Column metadata unavailable: {columns_note}")
elif columns_df.empty:
    st.info(f"No columns found for `{selected_schema}.{selected_table}`.")
else:
    st.subheader("Columns")
    st.dataframe(columns_df, hide_index=True, use_container_width=True)

rows_df, rows_note = preview_table_rows(
    selected_schema,
    selected_table,
    limit=preview_limit,
    database_path=database_arg,
)
if rows_note:
    st.warning(f"Preview query unavailable: {rows_note}")
else:
    st.subheader("Preview Rows")
    if rows_df.empty:
        st.info("Selected table has no rows.")
    else:
        preview = rows_df.copy()
        for column in preview.columns:
            if pd.api.types.is_datetime64_any_dtype(preview[column]):
                preview[column] = preview[column].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(preview, hide_index=True, use_container_width=True)

st.subheader("Query Snippet")
st.code(
    build_select_sql(
        selected_schema,
        selected_table,
        limit=preview_limit,
    ),
    language="sql",
)
