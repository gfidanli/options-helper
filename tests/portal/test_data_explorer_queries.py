from __future__ import annotations

from pathlib import Path

from apps.streamlit.components.data_explorer_page import (
    build_select_sql,
    list_database_schemas,
    list_database_tables,
    load_table_columns,
    preview_table_rows,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_db(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO derived_daily(
              symbol, date, spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike, atm_iv_near,
              em_near_pct, skew_near_pp, rv_20d, rv_60d, iv_rv_20d, atm_iv_near_percentile, iv_term_slope
            )
            VALUES ('AAPL', '2026-02-05', 100.0, 1.0, 1.0, 110, 90, 100, 0.20, 0.03, -0.01, 0.18, 0.20, 1.11, 0.40, 0.01)
            """
        )


def test_data_explorer_query_helpers(tmp_path: Path) -> None:
    db_path = tmp_path / "explorer.duckdb"
    _seed_db(db_path)

    schemas, schemas_note = list_database_schemas(database_path=db_path)
    assert schemas_note is None
    assert "main" in schemas
    assert "meta" in schemas

    main_tables_df, tables_note = list_database_tables(schema="main", database_path=db_path)
    assert tables_note is None
    assert not main_tables_df.empty
    derived_row = main_tables_df[main_tables_df["table_name"] == "derived_daily"].iloc[0].to_dict()
    assert derived_row["table_schema"] == "main"
    assert int(derived_row["row_count"]) == 1

    columns_df, columns_note = load_table_columns("main", "derived_daily", database_path=db_path)
    assert columns_note is None
    assert not columns_df.empty
    assert "symbol" in set(columns_df["column_name"].astype(str))
    assert "date" in set(columns_df["column_name"].astype(str))

    preview_df, preview_note = preview_table_rows("main", "derived_daily", limit=10, database_path=db_path)
    assert preview_note is None
    assert len(preview_df) == 1
    assert str(preview_df.iloc[0]["symbol"]).upper() == "AAPL"

    assert build_select_sql("main", "derived_daily", limit=10) == 'SELECT * FROM "main"."derived_daily" LIMIT 10;'


def test_data_explorer_handles_missing_db_and_missing_table(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.duckdb"
    schemas, schemas_note = list_database_schemas(database_path=missing_db)
    assert schemas == []
    assert schemas_note is not None
    assert "not found" in schemas_note.lower()

    db_path = tmp_path / "explorer.duckdb"
    _seed_db(db_path)

    missing_cols_df, missing_cols_note = load_table_columns("main", "missing_table", database_path=db_path)
    assert missing_cols_df.empty
    assert missing_cols_note is None

    missing_rows_df, missing_rows_note = preview_table_rows("main", "missing_table", database_path=db_path)
    assert missing_rows_df.empty
    assert missing_rows_note is not None
