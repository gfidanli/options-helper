from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from options_helper.analysis.strategy_modeling_trade_review import rank_trades_for_review

_SCOPE_LABEL_BY_CODE: dict[str, str] = {
    "accepted_closed_trades": "Accepted closed trades",
    "closed_nonrejected_trades": "Closed non-rejected trades",
}


def build_trade_review_tables(
    trade_df: pd.DataFrame,
    accepted_trade_ids: Sequence[str] | None,
    *,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    source_df = trade_df if isinstance(trade_df, pd.DataFrame) else pd.DataFrame()
    ranked = rank_trades_for_review(
        source_df.to_dict(orient="records"),
        accepted_trade_ids=accepted_trade_ids,
        top_n=top_n,
        metric="realized_r",
    )
    columns = _display_columns(source_df, ranked.top_best_rows, ranked.top_worst_rows)
    best_df = _rows_to_display_df(ranked.top_best_rows, columns=columns)
    worst_df = _rows_to_display_df(ranked.top_worst_rows, columns=columns)
    scope_label = _scope_label(ranked.scope)
    return best_df, worst_df, scope_label


def _display_columns(
    source_df: pd.DataFrame,
    *ranked_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    ordered = ["rank"]
    seen = {"rank"}

    for column in source_df.columns:
        name = str(column)
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)

    for rows in ranked_rows:
        for row in rows:
            for key in row:
                name = str(key)
                if not name or name in seen:
                    continue
                seen.add(name)
                ordered.append(name)

    return ordered


def _rows_to_display_df(rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    frame = frame.reindex(columns=list(columns))
    # Use a clean RangeIndex so Streamlit selection row indices map directly to .iloc.
    return frame.reset_index(drop=True)


def _scope_label(scope: object) -> str:
    code = str(scope or "").strip()
    if not code:
        return "Unknown scope"
    label = _SCOPE_LABEL_BY_CODE.get(code)
    if label is not None:
        return label
    return code.replace("_", " ")


__all__ = ["build_trade_review_tables"]
