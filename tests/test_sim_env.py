"""Model loading and the viewer launch helper."""
from __future__ import annotations

import sys
import types

import pytest

from quadruped.sim.env import launch_viewer, load_model, model_path


def test_model_path_defaults_to_config():
    assert model_path().name.endswith(".xml")
    assert model_path("single_leg").name == "single_leg.xml"
    assert model_path("quadruped").name == "quadruped.xml"


def test_both_models_load_and_pass_yaml_assertions():
    """quadruped.xml has 12 actuated joints plus a free base (13 total)."""
    for name, njnt in (("single_leg", 3), ("quadruped", 13)):
        model, data = load_model(model_path(name))
        assert model.njnt == njnt


def test_quadruped_has_a_free_base():
    """Without this the robot marches in place — it can't translate or turn."""
    model, _ = load_model(model_path("quadruped"))
    root = model.joint("root")
    assert model.nq == 19, "12 hinge joints + 7 free-joint qpos"
    assert int(root.qposadr[0]) == 0


def _fake_viewer(monkeypatch, exc):
    mod = types.SimpleNamespace(launch_passive=lambda *a, **k: (_ for _ in ()).throw(exc))
    monkeypatch.setitem(sys.modules, "mujoco.viewer", mod)


def test_macos_mjpython_error_becomes_an_actionable_message(monkeypatch):
    """MuJoCo says 'requires mjpython' without saying what to type.

    On macOS an interactive viewer must run under mjpython, and hitting this as
    a bare traceback repeatedly is pure friction — so we exit with the exact
    command instead.
    """
    _fake_viewer(monkeypatch, RuntimeError(
        "`launch_passive` requires that the Python script be run under `mjpython` on macOS"
    ))
    monkeypatch.setattr(sys, "argv", ["scripts/view.py", "--model", "quad"])

    with pytest.raises(SystemExit) as e:
        launch_viewer(object(), object())

    msg = str(e.value)
    assert "mjpython scripts/view.py --model quad" in msg
    assert ".venv/bin/mjpython" in msg


def test_unrelated_viewer_errors_are_not_swallowed(monkeypatch):
    _fake_viewer(monkeypatch, RuntimeError("display server exploded"))
    with pytest.raises(RuntimeError, match="display server exploded"):
        launch_viewer(object(), object())
