"""Drive the interactive calibration loop with a fake backend + scripted input."""
import builtins

import pytest

from quadruped.calibration import goal_ticks
from quadruped.cli.calibrate import _calibrate_joint
from quadruped.config import CONFIG


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def write_goal_ticks(self, name, ticks):
        self.calls.append((name, ticks))


def test_calibrate_joint_offset_nudge_and_direction_flip(monkeypatch):
    # offset: nudge +5 then 'ok'; direction: 'n' (flip) then 'y'
    answers = iter(["+5", "ok", "n", "y"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(answers))

    j = CONFIG.joint("shoulder_FR")  # starts direction=1, offset=180
    be = _FakeBackend()
    direction, offset = _calibrate_joint(be, j, test_angle=0.3)

    assert direction == -1               # flipped by the 'n'
    assert offset == pytest.approx(185.0)  # 180 + 5
    # ends parked at zero pose using the FINAL calibration
    assert be.calls[-1] == ("shoulder_FR", goal_ticks(-1, 185.0, 0.0))


def test_calibrate_joint_no_changes(monkeypatch):
    # accept offset immediately, direction correct first try
    answers = iter(["ok", "y"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(answers))

    j = CONFIG.joint("knee_FL")
    direction, offset = _calibrate_joint(_FakeBackend(), j, test_angle=0.3)
    assert direction == j.direction and offset == pytest.approx(j.offset_deg)
