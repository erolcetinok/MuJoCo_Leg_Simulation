"""Kinematics tests.

  * test_fk_matches_mujoco — FK oracle: foot_position() vs MuJoCo, which
    computes the foot position from the same geometry configs/robot.yaml
    encodes, so it is an independent reference for the hand-derived chain.
  * test_ik_round_trip — IK is the inverse of FK: FK(IK(p)) must recover p.
"""
import mujoco
import numpy as np
import pytest

from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.ik import joint_angles
from quadruped.sim.env import load_model

# (shoulder, wing, knee) in radians — kept within the joint limits.
POSES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, -1.5),
    (-1.2, 0.6, -1.8),
    (0.7, -0.4, 0.9),
]


def _mujoco_foot(model, data, angles):
    """Foot-site world position MuJoCo computes for the given joint angles."""
    data.qpos[:] = angles
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site("foot_site").id].copy()


@pytest.mark.parametrize("angles", POSES)
def test_fk_matches_mujoco(angles):
    model, data = load_model()

    # Hip-frame origin = shoulder hinge anchor in world. Read from the model
    # so the test stays correct if the leg's mount position ever changes.
    mujoco.mj_forward(model, data)
    hip = data.xanchor[model.joint("shoulder_joint").id].copy()

    expected = _mujoco_foot(model, data, angles)
    actual = foot_position(*angles) + hip

    assert np.allclose(actual, expected, atol=1e-6), (
        f"angles={angles}: FK {actual} != MuJoCo {expected} "
        f"(err {np.linalg.norm(actual - expected):.3e} mm)"
    )


@pytest.mark.parametrize("angles", POSES)
def test_ik_round_trip(angles):
    """FK(IK(p)) must land the foot back on the target p.

    Tested as a *position* round-trip, not an angle round-trip: analytic IK
    has multiple branches, so IK(FK(angles)) may legitimately return a
    different joint triple — but it must still put the foot on the target.
    """
    target = foot_position(*angles)
    solved = joint_angles(*target)
    recovered = foot_position(*solved)

    assert np.allclose(recovered, target, atol=1e-6), (
        f"target={target}: FK(IK(target))={recovered} "
        f"(err {np.linalg.norm(recovered - target):.3e} mm)"
    )
