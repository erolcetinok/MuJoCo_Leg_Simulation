"""The stand-up ramp: the thing that stops tick 0 snapping twelve servos.

Every test uses rate=1000 so the deadline pacing costs milliseconds, not seconds.
"""
from __future__ import annotations

import mujoco
import pytest

from quadruped.config import CONFIG
from quadruped.control.locomotion import LocomotionController
from quadruped.control.sequencer import ramp_to, stance_pose
from quadruped.sim.env import leg_poses, model_path


class _RecordingBackend:
    """Captures every commanded pose; reports `present` as its joint state."""

    def __init__(self, present: dict = None):
        self.present = present
        self.sent: list = []

    def set_joint_targets(self, q):
        self.sent.append(dict(q))

    def read_joint_state(self):
        if self.present is None:
            return {}, {}
        return dict(self.present), {}


@pytest.fixture
def controller():
    model = mujoco.MjModel.from_xml_path(str(model_path()))
    return LocomotionController(leg_poses(model))


def test_stance_pose_covers_all_twelve_and_holds_the_phase(controller):
    before = controller.scheduler.phase
    pose = stance_pose(controller)
    assert set(pose) == set(CONFIG.joint_names)
    assert controller.scheduler.phase == before, "stance_pose must not advance the gait"


def test_ramp_interpolates_monotonically_from_the_measured_pose(controller):
    targets = {"shoulder_FL": 1.0, "knee_FL": -1.0}
    backend = _RecordingBackend(present={"shoulder_FL": 0.0, "knee_FL": 0.0})
    ramp_to(backend, targets, duration=0.05, rate=1000.0)

    assert len(backend.sent) == 50
    ups = [f["shoulder_FL"] for f in backend.sent]
    downs = [f["knee_FL"] for f in backend.sent]
    assert ups == sorted(ups) and downs == sorted(downs, reverse=True)
    assert ups[0] == pytest.approx(0.02, abs=1e-9)      # first step, not a jump
    assert backend.sent[-1] == pytest.approx(targets)   # lands exactly on target


def test_ramp_without_feedback_still_reaches_the_target(controller):
    """A backend with no read channel (sim) degenerates to writing the target."""
    targets = {"shoulder_FL": 0.4}
    backend = _RecordingBackend(present=None)
    ramp_to(backend, targets, duration=0.01, rate=1000.0)
    assert all(f == pytest.approx(targets) for f in backend.sent)


def test_ramp_survives_a_backend_that_cannot_be_read(controller):
    class _Angry(_RecordingBackend):
        def read_joint_state(self):
            raise RuntimeError("bus busy")

    backend = _Angry()
    out = ramp_to(backend, {"wing_BR": 0.2}, duration=0.01, rate=1000.0)
    assert out == {"wing_BR": 0.2} and backend.sent


def test_partially_readable_bus_ramps_only_what_it_can_see(controller):
    """A joint missing from the read starts at its target rather than at 0."""
    backend = _RecordingBackend(present={"shoulder_FL": 0.0})
    ramp_to(backend, {"shoulder_FL": 1.0, "knee_FL": 0.5}, duration=0.01, rate=1000.0)
    assert all(f["knee_FL"] == 0.5 for f in backend.sent)
    assert backend.sent[0]["shoulder_FL"] < 1.0


def test_ramp_ends_at_the_gait_start_pose(controller):
    """The handoff that matters: what the ramp lands on is what tick 0 commands."""
    backend = _RecordingBackend(present={n: 0.0 for n in CONFIG.joint_names})
    ramp_to(backend, stance_pose(controller), duration=0.01, rate=1000.0)
    assert backend.sent[-1] == pytest.approx(controller.step(0.0))
