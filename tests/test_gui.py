"""GuiApp without a window.

The DPG event loop needs a real display and the main thread, so `run()` stays
untested here. Everything underneath it does not: construction, the backend
lifecycle, and the slider -> backend send path all run headlessly, which is
where a config or model change actually regresses.
"""
from __future__ import annotations

import pytest

from quadruped.backends import BACKEND_CHOICES, MujocoBackend
from quadruped.config import CONFIG


@pytest.fixture
def app():
    pytest.importorskip("dearpygui.dearpygui")
    from quadruped.gui.app import GuiApp
    return GuiApp(backend_kind="sim", port=None, view_mode="none")


def test_gui_app_constructs(app):
    assert app.ik_model.nu == 12
    assert len(app._joint_qpos_idx) == 12
    assert app._target_default.shape == (3,)
    assert app.view_mode == "none"


def test_gui_app_rejects_unknown_view_mode():
    pytest.importorskip("dearpygui.dearpygui")
    from quadruped.gui.app import GuiApp

    with pytest.raises(ValueError):
        GuiApp(backend_kind="sim", port=None, view_mode="bogus")


def test_offers_every_backend_choice():
    """The dropdown is built from BACKEND_CHOICES.

    It used to be a hardcoded ["sim", "hw", "mirror"], which is how the GUI
    silently lacked `dxl` while every command had it.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "src" / "quadruped" / "gui" / "app.py"
    ).read_text()
    assert "items=list(BACKEND_CHOICES)" in src
    assert '"sim", "hw", "mirror"' not in src


def test_open_backend_connects_and_reports(app, capsys):
    app._open_backend("sim")
    assert isinstance(app.backend, MujocoBackend)
    assert app.backend_kind == "sim"
    assert "backend = sim" in capsys.readouterr().err
    app._close_backend()
    assert app.backend is None


def test_open_backend_survives_a_bad_kind(app, capsys):
    """A failed open must leave the app usable, not half-connected."""
    app._open_backend("nonsense")
    assert app.backend is None
    assert "backend open failed" in capsys.readouterr().err


def test_send_reaches_the_backend(app):
    """The whole point of the GUI: slider value -> joint dict -> backend."""
    app._open_backend("sim")
    name = CONFIG.joint_names[0]
    idx = app.backend._qpos_idx[name]

    app._send({name: 0.25})
    assert app.backend.data.qpos[idx] == pytest.approx(0.25)

    app._send({name: -0.4})
    assert app.backend.data.qpos[idx] == pytest.approx(-0.4)
    app._close_backend()


def test_send_without_a_backend_is_a_noop(app):
    assert app.backend is None
    app._send({CONFIG.joint_names[0]: 0.1})  # must not raise


def test_send_reports_backend_errors_instead_of_crashing(app, capsys):
    class Boom:
        def set_joint_targets(self, q):
            raise RuntimeError("bus fell over")

    app.backend = Boom()
    app._send({CONFIG.joint_names[0]: 0.1})
    assert "send failed: bus fell over" in capsys.readouterr().err
