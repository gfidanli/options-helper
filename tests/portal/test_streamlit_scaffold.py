from __future__ import annotations

import importlib
import runpy
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMLIT_DIR = REPO_ROOT / "apps" / "streamlit"
PAGE_FILES = [
    STREAMLIT_DIR / "pages" / "01_Health.py",
    STREAMLIT_DIR / "pages" / "02_Portfolio.py",
    STREAMLIT_DIR / "pages" / "03_Symbol_Explorer.py",
    STREAMLIT_DIR / "pages" / "04_Flow.py",
    STREAMLIT_DIR / "pages" / "05_Derived_History.py",
    STREAMLIT_DIR / "pages" / "06_Data_Explorer.py",
    STREAMLIT_DIR / "pages" / "07_Market_Analysis.py",
    STREAMLIT_DIR / "pages" / "08_Coverage.py",
    STREAMLIT_DIR / "pages" / "09_SFP.py",
    STREAMLIT_DIR / "pages" / "10_MSB.py",
]


def test_streamlit_scaffold_files_exist() -> None:
    required = [
        STREAMLIT_DIR / "streamlit_app.py",
        STREAMLIT_DIR / "components" / "db.py",
        STREAMLIT_DIR / "components" / "queries.py",
        STREAMLIT_DIR / "components" / "gap_planner.py",
    ]
    for path in required + PAGE_FILES:
        assert path.exists(), f"Missing scaffold file: {path}"


def test_streamlit_module_import_smoke() -> None:
    pytest.importorskip("streamlit")

    importlib.import_module("apps.streamlit.streamlit_app")
    importlib.import_module("apps.streamlit.components.db")
    importlib.import_module("apps.streamlit.components.queries")
    importlib.import_module("apps.streamlit.components.gap_planner")
    importlib.import_module("apps.streamlit.components.coverage_page")
    importlib.import_module("apps.streamlit.components.sfp_page")
    importlib.import_module("apps.streamlit.components.msb_page")

    for page_file in PAGE_FILES:
        runpy.run_path(str(page_file), run_name=f"__streamlit_page_{page_file.stem}__")


def test_cached_db_and_query_helpers_are_read_only(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    db_module = importlib.import_module("apps.streamlit.components.db")
    queries_module = importlib.import_module("apps.streamlit.components.queries")

    db_path = tmp_path / "portal.duckdb"
    seed_conn = duckdb.connect(str(db_path))
    seed_conn.execute("create table sample(value integer)")
    seed_conn.execute("insert into sample values (1), (2), (3)")
    seed_conn.close()

    db_module.get_read_only_connection.clear()
    queries_module.run_cached_query.clear()

    conn = db_module.get_read_only_connection(database_path=str(db_path))
    assert conn.execute("select count(*) from sample").fetchone()[0] == 3
    with pytest.raises(duckdb.Error):
        conn.execute("create table should_fail(value integer)")

    df = queries_module.run_query(
        sql="select sum(value) as total from sample",
        database_path=str(db_path),
    )
    assert int(df.loc[0, "total"]) == 6


def test_gap_planner_query_param_helpers() -> None:
    pytest.importorskip("streamlit")
    gap_module = importlib.import_module("apps.streamlit.components.gap_planner")

    params: dict[str, str] = {}
    assert (
        gap_module.read_query_param(
            name="symbol",
            default="SPY",
            query_params=params,
        )
        == "SPY"
    )

    result = gap_module.sync_query_param(
        name="symbol",
        value="AAPL",
        query_params=params,
    )
    assert result == "AAPL"
    assert params["symbol"] == "AAPL"

    symbols = gap_module.sync_csv_query_param(
        name="symbols",
        values=["AAPL", " ", "MSFT"],
        query_params=params,
    )
    assert symbols == ["AAPL", "MSFT"]
    assert params["symbols"] == "AAPL,MSFT"
