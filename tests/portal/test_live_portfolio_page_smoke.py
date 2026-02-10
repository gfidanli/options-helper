from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_live_portfolio_page_import_and_render_without_auto_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    st = pytest.importorskip("streamlit")
    module = importlib.import_module("apps.streamlit.components.live_portfolio_page")

    class FakeManager:
        init_calls = 0
        start_calls = 0

        def __init__(self) -> None:
            type(self).init_calls += 1
            self._running = False

        def start(self, config: object) -> None:
            _ = config
            type(self).start_calls += 1
            self._running = True

        def stop(self) -> None:
            self._running = False

        def is_running(self) -> bool:
            return self._running

        def snapshot(self) -> object:
            return module.LiveSnapshot(running=self._running)

    monkeypatch.setattr(module, "LiveStreamManager", FakeManager)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_BASE_URL", raising=False)

    missing_portfolio = tmp_path / "missing.json"
    st.session_state.clear()
    st.session_state[module._WIDGET_PORTFOLIO_PATH_KEY] = str(missing_portfolio)

    module.render_live_portfolio_page()
    module.render_live_portfolio_page()

    manager = st.session_state.get(module._STATE_MANAGER_KEY)
    assert isinstance(manager, FakeManager)
    assert FakeManager.init_calls == 1
    assert FakeManager.start_calls == 0
