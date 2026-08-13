"""LocomotionController — the full tick, and the joint-limit safety net.

The sweep test is the one that matters: it asserts that no combination of
velocity, turn rate and body pose the teleop UI can produce ever commands a
joint outside the limits in configs/robot.yaml. That is what stops the operator
from folding a leg through a hard stop.
"""
from __future__ import annotations

import itertools
import math

import mujoco
import numpy as np
import pytest

from quadruped.config import CONFIG
from quadruped.control.body import (
    MAX_SHIFT_MM,
    MAX_TILT_RAD,
    MAX_HEIGHT_MM,
    MAX_PITCH_RAD,
    MAX_ROLL_RAD,
    MAX_YAW_RAD,
    MIN_HEIGHT_MM,
    BodyPose,
)
from quadruped.control.locomotion import LocomotionController, home_positions
from quadruped.sim.env import leg_poses, model_path


@pytest.fixture(scope="module")
def poses():
    return leg_poses(mujoco.MjModel.from_xml_path(str(model_path())))


def test_home_positions_sit_at_the_design_stance(poses):
    """(±169, ±169, −80): 169 mm out along each body diagonal, 80 mm down."""
    home = home_positions(poses)
    assert set(home) == set(CONFIG.legs)
    for leg, p in home.items():
        assert abs(p[0]) == pytest.approx(169.0, abs=0.5)
        assert abs(p[1]) == pytest.approx(169.0, abs=0.5)
        assert p[2] == pytest.approx(-80.0, abs=0.5)


def test_step_returns_every_joint(poses):
    targets = LocomotionController(poses).step(0.02)
    assert set(targets) == set(CONFIG.joint_names)
    assert all(isinstance(v, float) for v in targets.values())


def test_unknown_gait_is_rejected(poses):
    with pytest.raises(ValueError, match="unknown gait"):
        LocomotionController(poses, gait="gallop")


def test_set_gait_preserves_phase(poses):
    c = LocomotionController(poses, gait="walk")
    for _ in range(7):
        c.step(0.02, body_velocity=(40.0, 0.0))
    phase = c.scheduler.phase
    c.set_gait("trot")
    assert c.gait == "trot"
    assert c.scheduler.phase == pytest.approx(phase), "gait switch must not jump the phase"


def test_identity_pose_matches_no_pose_exactly(poses):
    """Guards the pre-teleop gait_demo behaviour."""
    a = LocomotionController(poses, gait="trot")
    b = LocomotionController(poses, gait="trot")
    for _ in range(30):
        ta = a.step(0.02, body_velocity=(40.0, 10.0), yaw_rate=0.2)
        tb = b.step(0.02, body_velocity=(40.0, 10.0), yaw_rate=0.2, pose=BodyPose())
        assert ta == tb


def test_stance_summary_names_every_leg(poses):
    summary = LocomotionController(poses).stance_summary()
    for leg in CONFIG.legs:
        assert leg in summary
    assert summary.count("st") + summary.count("sw") == len(CONFIG.legs)


V_MAX = 100.0        # matches control.command.default_axes


def _envelope_poses():
    """Worst-case poses: full tilt swept around every direction, at both height
    extremes and both shift corners. Tilt direction matters because pitch and
    roll bind different legs."""
    out = [BodyPose()]
    for deg in range(0, 360, 45):
        a = math.radians(deg)
        pitch = MAX_TILT_RAD * math.cos(a)
        roll = MAX_TILT_RAD * math.sin(a)
        for z in (MIN_HEIGHT_MM, MAX_HEIGHT_MM):
            for s in (MAX_SHIFT_MM, -MAX_SHIFT_MM):
                out.append(
                    BodyPose(pitch=pitch, roll=roll, yaw=math.copysign(MAX_YAW_RAD, s),
                             x=s, y=s, z=z).clamped()
                )
    return out


@pytest.mark.parametrize("gait", ["walk", "trot"])
def test_joint_limits_hold_across_the_whole_command_envelope(poses, gait):
    """Sweep everything teleop can command, over a full gait cycle.

    This is the safety net that stops the operator folding a leg through a hard
    stop, and it is what set the envelope in control/body.py in the first place:
    the original ±15° box exceeded the wing joint's travel by 18%.
    """
    limits = {j.name: j.limit_rad for j in CONFIG.joints}
    velocities = [(0.0, 0.0), (V_MAX, 0.0), (-V_MAX, 0.0), (0.0, V_MAX),
                  (V_MAX * 0.7, V_MAX * 0.7)]
    yaw_rates = [0.0, 1.0, -1.0]
    poses_to_try = _envelope_poses()

    violations = []
    for v, w, pose in itertools.product(velocities, yaw_rates, poses_to_try):
        c = LocomotionController(poses, gait=gait)
        for _ in range(60):             # >1 full cycle at 50 Hz
            for name, q in c.step(0.02, body_velocity=v, yaw_rate=w, pose=pose).items():
                lo, hi = limits[name]
                if not (lo - 1e-6 <= q <= hi + 1e-6):
                    violations.append(f"{name}={q:.3f} outside ({lo:.3f},{hi:.3f}) "
                                      f"at v={v} w={w} pose={pose}")
    assert not violations, "commands outside joint limits:\n" + "\n".join(violations[:10])


@pytest.mark.parametrize("gait", ["walk", "trot"])
def test_foot_targets_stay_finite_and_reachable_looking(poses, gait):
    """No NaNs, and feet stay a plausible distance from their shoulders."""
    c = LocomotionController(poses, gait=gait)
    for _ in range(60):
        c.step(0.02, body_velocity=(80.0, 0.0), yaw_rate=0.5)
        for leg, p_local in c.foot_targets(BodyPose(pitch=0.1, z=10.0)).items():
            assert np.all(np.isfinite(p_local))
            assert 80.0 < float(np.linalg.norm(p_local)) < 260.0
