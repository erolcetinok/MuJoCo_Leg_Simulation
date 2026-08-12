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
    "hw": ArduinoBackend,
    "mirror": MirrorBackend,
}


def test_choices_are_exactly_the_documented_four():
    assert set(BACKEND_CHOICES) == set(EXPECTED)


@pytest.mark.parametrize("kind", sorted(EXPECTED))
def test_make_backend_builds_each_kind(kind):
    b = make_backend(kind, port="/dev/null")
    assert isinstance(b, EXPECTED[kind])


def test_mirror_fans_to_sim_and_hardware_with_hw_as_truth():
    b = make_backend("mirror", port="/dev/null")
    assert isinstance(b, MirrorBackend)
    kinds = [type(x) for x in b.backends]
    assert kinds == [MujocoBackend, ArduinoBackend]
    assert b._truth_idx == 1, "hardware must be the truth source, not the sim"


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
    args = p.parse_args(["--backend", "hw", "--port", "/dev/explicit"])
    assert build_backend(args).port == "/dev/explicit"


def test_build_backend_falls_back_to_env_port(monkeypatch):
    monkeypatch.setenv("SERIAL_PORT", "/dev/from-env")
    p = argparse.ArgumentParser()
    add_backend_args(p)
    args = p.parse_args(["--backend", "hw"])
    assert build_backend(args).port == "/dev/from-env"
