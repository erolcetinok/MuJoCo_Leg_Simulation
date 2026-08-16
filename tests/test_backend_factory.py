"""`make_backend` is the one place a --backend string becomes an object.

Every command and the GUI go through it, so this is what stops them drifting
apart again — the GUI previously had its own private copy that had fallen a
backend behind.

Construction only: no backend touches a port until connect().
"""
from __future__ import annotations

import argparse

import pytest

from quadruped.backends import (
    BACKEND_CHOICES,
    ArduinoBackend,
    DynamixelBackend,
    MirrorBackend,
    MujocoBackend,
    make_backend,
)
from quadruped.cli_args import add_backend_args, build_backend

EXPECTED = {
    "sim": MujocoBackend,
    "dxl": DynamixelBackend,
    "mirror": MirrorBackend,
}


def test_choices_are_exactly_the_documented_three():
    assert set(BACKEND_CHOICES) == set(EXPECTED)


def test_arduino_is_archived_not_selectable():
    """The UNO bridge stays importable for the old rig, but no --backend string
    reaches it: the U2D2 is the one supported hardware path."""
    assert "hw" not in BACKEND_CHOICES
    assert ArduinoBackend(port="/dev/null").port == "/dev/null"


@pytest.mark.parametrize("kind", sorted(EXPECTED))
def test_make_backend_builds_each_kind(kind):
    b = make_backend(kind, port="/dev/null")
    assert isinstance(b, EXPECTED[kind])


def test_mirror_fans_to_sim_and_hardware_with_hw_as_truth():
    b = make_backend("mirror", port="/dev/null")
    assert isinstance(b, MirrorBackend)
    kinds = [type(x) for x in b.backends]
    assert kinds == [MujocoBackend, DynamixelBackend]
    assert b._truth_idx == 1, "hardware must be the truth source, not the sim"


@pytest.mark.parametrize("kind", ["dxl", "mirror"])
def test_profile_velocity_reaches_the_servo_backend(kind):
    """A cautious first power-on (--profile-velocity 30) must survive the factory."""
    b = make_backend(kind, port="/dev/null", profile_velocity=30)
    dxl = b.backends[1] if kind == "mirror" else b
    assert dxl._profile_velocity == 30


def test_profile_velocity_defaults_to_the_streaming_value():
    assert make_backend("dxl", port="/dev/null")._profile_velocity == 0


def test_mirror_fans_out_the_torque_kill():
    """Inheriting the ABC no-op would make teleop's `z` silently do nothing here."""
    class _Spy(MujocoBackend):
        def __init__(self): self.torque = None
        def set_torque_all(self, on): self.torque = on

    spies = [_Spy(), _Spy()]
    MirrorBackend(spies).set_torque_all(False)
    assert [s.torque for s in spies] == [False, False]


def test_mirror_health_comes_from_whichever_backend_has_one():
    class _Silent(MujocoBackend):
        def __init__(self): pass

    class _Talker(MujocoBackend):
        def __init__(self): pass
        def health_check(self): return {"knee_FL": (0, 40)}

    assert MirrorBackend([_Silent(), _Talker()]).health_check() == {"knee_FL": (0, 40)}


def test_unknown_backend_names_the_valid_options():
    with pytest.raises(ValueError) as e:
        make_backend("nope")
    for kind in BACKEND_CHOICES:
        assert kind in str(e.value)


def test_argparse_offers_exactly_the_factory_choices():
    """The flag and the factory must not drift — that's how the GUI lost dxl."""
    p = argparse.ArgumentParser()
    add_backend_args(p)
    action = next(a for a in p._actions if a.dest == "backend")
    assert tuple(action.choices) == BACKEND_CHOICES


def test_build_backend_prefers_explicit_port_over_env(monkeypatch):
    monkeypatch.setenv("SERIAL_PORT", "/dev/from-env")
    p = argparse.ArgumentParser()
    add_backend_args(p)
    args = p.parse_args(["--backend", "dxl", "--port", "/dev/explicit"])
    assert build_backend(args).port == "/dev/explicit"


def test_build_backend_falls_back_to_env_port(monkeypatch):
    monkeypatch.setenv("SERIAL_PORT", "/dev/from-env")
    p = argparse.ArgumentParser()
    add_backend_args(p)
    args = p.parse_args(["--backend", "dxl"])
    assert build_backend(args).port == "/dev/from-env"
