"""Smoke test: GuiApp constructs without opening a window.

We don't drive the DPG event loop here (it needs a real display + main
thread). The constructor exercises load_model, IK init, embedded-renderer
plumbing, and slider-bound calculations — which is what's most likely to
regress in a config or model change.
"""
from __future__ import annotations

import pytest


def test_gui_app_constructs():
    pytest.importorskip("dearpygui.dearpygui")
    from quadruped.gui.app import GuiApp

    app = GuiApp(backend_kind="sim", port=None, view_mode="none")
    assert app.ik_model.nu == 3
    assert app.ik is not None
    assert app.view_mode == "none"


def test_gui_app_rejects_unknown_view_mode():
    pytest.importorskip("dearpygui.dearpygui")
    from quadruped.gui.app import GuiApp

    with pytest.raises(ValueError):
        GuiApp(backend_kind="sim", port=None, view_mode="bogus")
