from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from apps.streamlit.components.flow_page import (
    build_flow_option_type_summary,
    build_flow_timeseries,
    list_flow_groups,
    list_flow_symbols,
    load_flow_asof_bounds,
    load_flow_partition_summaries,
    load_flow_rows_for_partition,
    resolve_flow_group_selection,
    sync_flow_group_query_param,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_flow(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO options_flow(
              symbol, as_of, from_date, to_date, window_size, group_by, row_key,
              contract_symbol, expiry, option_type, strike,
              delta_oi, delta_oi_notional, volume_notional, delta_notional, n_pairs
            )
            VALUES
              (
                'AAPL', '2026-02-04', '2026-02-03', '2026-02-04', 2, 'expiry-strike', 'aapl-1',
                'AAPL260320C00100000', '2026-03-20', 'call', 100.0,
                10.0, 10000.0, 50000.0, 8000.0, 2
              ),
              (
                'AAPL', '2026-02-04', '2026-02-03', '2026-02-04', 2, 'expiry-strike', 'aapl-2',
                'AAPL260320P00095000', '2026-03-20', 'put', 95.0,
                -7.0, -7000.0, 30000.0, -5000.0, 2
              ),
              (
                'AAPL', '2026-02-05', '2026-02-04', '2026-02-05', 2, 'expiry-strike', 'aapl-3',
                'AAPL260320C00100000', '2026-03-20', 'call', 100.0,
                22.0, 22000.0, 65000.0, 15000.0, 2
              ),
              (
                'AAPL', '2026-02-05', '2026-02-04', '2026-02-05', 2, 'expiry-strike', 'aapl-4',
                'AAPL260320P00095000', '2026-03-20', 'put', 95.0,
                -4.0, -4000.0, 25000.0, -2000.0, 2
              ),
              (
                'AAPL', '2026-02-05', '2026-02-04', '2026-02-05', 2, 'strike', 'aapl-5',
                NULL, '2026-03-20', 'call', 100.0,
                9.0, 12000.0, 40000.0, 9000.0, 2
              ),
              (
                'MSFT', '2026-02-05', '2026-02-04', '2026-02-05', 2, 'expiry-strike', 'msft-1',
                'MSFT260320C00400000', '2026-03-20', 'call', 400.0,
                4.0, 18000.0, 35000.0, 12000.0, 2
              )
            """
        )


def test_flow_query_helpers_filter_and_summarize(tmp_path: Path) -> None:
    db_path = tmp_path / "flow.duckdb"
    _seed_flow(db_path)

    symbols, symbols_note = list_flow_symbols(database_path=db_path)
    assert symbols_note is None
    assert symbols == ["AAPL", "MSFT"]

    groups, groups_note = list_flow_groups("AAPL", database_path=db_path)
    assert groups_note is None
    assert groups == ["strike", "expiry-strike"]

    min_as_of, max_as_of, bounds_note = load_flow_asof_bounds(
        "AAPL",
        group_by="expiry-strike",
        database_path=db_path,
    )
    assert bounds_note is None
    assert min_as_of == date(2026, 2, 4)
    assert max_as_of == date(2026, 2, 5)

    partitions_df, partitions_note = load_flow_partition_summaries(
        "AAPL",
        group_by="expiry-strike",
        database_path=db_path,
    )
    assert partitions_note is None
    assert len(partitions_df) == 2
    assert float(partitions_df.iloc[0]["net_delta_oi_notional"]) == 18000.0
    assert float(partitions_df.iloc[0]["total_volume_notional"]) == 90000.0
    assert int(partitions_df.iloc[0]["rows"]) == 2

    filtered_df, filtered_note = load_flow_partition_summaries(
        "AAPL",
        group_by="expiry-strike",
        as_of_start="2026-02-05",
        as_of_end="2026-02-05",
        database_path=db_path,
    )
    assert filtered_note is None
    assert len(filtered_df) == 1

    timeseries_df = build_flow_timeseries(partitions_df)
    assert len(timeseries_df) == 2
    assert list(timeseries_df["net_delta_oi_notional"]) == [3000.0, 18000.0]

    selected_partition = partitions_df.iloc[0]
    rows_df, rows_note = load_flow_rows_for_partition(
        "AAPL",
        as_of=selected_partition["as_of"],
        from_date=selected_partition["from_date"],
        to_date=selected_partition["to_date"],
        window=int(selected_partition["window"]),
        group_by="expiry-strike",
        top_n=10,
        database_path=db_path,
    )
    assert rows_note is None
    assert len(rows_df) == 2
    assert float(rows_df.iloc[0]["delta_oi_notional"]) == 22000.0
    assert str(rows_df.iloc[0]["option_type"]).lower() == "call"

    option_summary = build_flow_option_type_summary(rows_df).set_index("option_type")
    assert float(option_summary.loc["call", "net_delta_oi_notional"]) == 22000.0
    assert float(option_summary.loc["put", "net_delta_oi_notional"]) == -4000.0


def test_flow_group_query_param_helpers() -> None:
    assert (
        resolve_flow_group_selection(
            ["strike", "expiry-strike"],
            query_group="expiry-strike",
        )
        == "expiry-strike"
    )
    assert (
        resolve_flow_group_selection(
            ["strike", "expiry-strike"],
            query_group="contract",
        )
        == "strike"
    )

    params: dict[str, str] = {}
    assert sync_flow_group_query_param(group_by="expiry-strike", query_params=params) == "expiry-strike"
    assert "group_by" not in params

    assert sync_flow_group_query_param(group_by="strike", query_params=params) == "strike"
    assert params["group_by"] == "strike"


def test_flow_query_helpers_handle_missing_db_or_table(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.duckdb"
    symbols, symbols_note = list_flow_symbols(database_path=missing_db)
    assert symbols == []
    assert symbols_note is not None
    assert "not found" in symbols_note.lower()

    raw_db = tmp_path / "raw.duckdb"
    conn = duckdb.connect(str(raw_db))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
        conn.execute("INSERT INTO sample VALUES (1)")
    finally:
        conn.close()

    table_symbols, table_note = list_flow_symbols(database_path=raw_db)
    assert table_symbols == []
    assert table_note is not None
    assert "options_flow table not found" in table_note

    partitions_df, partitions_note = load_flow_partition_summaries(
        "AAPL",
        group_by="expiry-strike",
        database_path=raw_db,
    )
    assert partitions_df.empty
    assert partitions_note is not None
    assert "options_flow table not found" in partitions_note
