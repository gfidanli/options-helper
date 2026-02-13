from __future__ import annotations

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).with_name("11_Strategy_Modeling_legacy.py")),
    run_name=__name__,
)
