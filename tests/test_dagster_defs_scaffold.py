from __future__ import annotations

import importlib

import pytest


def test_dagster_defs_scaffold_import_smoke() -> None:
    pytest.importorskip("dagster")

    module = importlib.import_module("apps.dagster.defs")

    from dagster import Definitions

    assert isinstance(module.defs, Definitions)
