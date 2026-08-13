"""Body pose transform and body<->leg frame conversion."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quadruped.control.body import (
    MAX_HEIGHT_MM,
    MAX_PITCH_RAD,
    MAX_ROLL_RAD,
    MAX_SHIFT_MM,
    MAX_TILT_RAD,
    MIN_HEIGHT_MM,
    BodyPose,
    body_to_leg,
    leg_to_body,
    rotz_xy,
)

P = np.array([169.0, 169.0, -80.0])      # a front-left foot at rest


def test_identity_pose_is_a_no_op():
    """The regression guard: existing gait behaviour must be untouched."""
    p = BodyPose()
    assert p.is_identity
    assert p.apply(P) is P or np.array_equal(p.apply(P), P)


def test_pitch_moves_front_and_rear_feet_oppositely():
    """Right-handed about +y: positive pitch is nose DOWN."""
    front = np.array([169.0, 0.0, -80.0])
    rear = np.array([-169.0, 0.0, -80.0])
    p = BodyPose(pitch=math.radians(10.0))

    dz_front = p.apply(front)[2] - front[2]
    dz_rear = p.apply(rear)[2] - rear[2]

    assert dz_front > 0 and dz_rear < 0, "nose-down should raise front feet in body frame"


def test_roll_moves_left_and_right_feet_oppositely():
    left = np.array([0.0, 169.0, -80.0])
    right = np.array([0.0, -169.0, -80.0])
    p = BodyPose(roll=math.radians(10.0))

    assert (p.apply(left)[2] - left[2]) < 0 < (p.apply(right)[2] - right[2])


def test_height_offset_pushes_feet_down_by_exactly_that_much():
    """+z raises the body, so feet move down the same distance in body frame."""
    assert BodyPose(z=20.0).apply(P)[2] == pytest.approx(P[2] - 20.0)


def test_translation_shifts_feet_opposite_to_the_body():
    assert BodyPose(x=10.0).apply(P)[0] == pytest.approx(P[0] - 10.0)
    assert BodyPose(y=-5.0).apply(P)[1] == pytest.approx(P[1] + 5.0)


def test_body_yaw_rotates_in_plane_without_changing_height():
    out = BodyPose(yaw=math.radians(20.0)).apply(P)
    assert out[2] == pytest.approx(P[2])
    assert np.linalg.norm(out[:2]) == pytest.approx(np.linalg.norm(P[:2]))


def test_pure_rotation_preserves_distance_from_the_body_origin():
    for pose in (BodyPose(roll=0.2), BodyPose(pitch=-0.15), BodyPose(yaw=0.3)):
        assert np.linalg.norm(pose.apply(P)) == pytest.approx(np.linalg.norm(P))


def test_clamped_holds_the_command_envelope():
    wild = BodyPose(pitch=math.radians(80), roll=-math.radians(90), z=500.0, x=999.0).clamped()
    assert math.hypot(wild.pitch, wild.roll) == pytest.approx(MAX_TILT_RAD)
    assert wild.z == pytest.approx(MAX_HEIGHT_MM)
    assert abs(wild.x) <= MAX_SHIFT_MM + 1e-9


def test_single_axis_tilt_still_reaches_its_full_limit():
    assert BodyPose(pitch=math.radians(45)).clamped().pitch == pytest.approx(MAX_PITCH_RAD)
    assert BodyPose(roll=-math.radians(45)).clamped().roll == pytest.approx(-MAX_ROLL_RAD)


def test_pitch_and_roll_are_limited_together_not_independently():
    """They add on the diagonal legs, so a box limit isn't safe — see body.py."""
    both = BodyPose(pitch=MAX_PITCH_RAD, roll=MAX_ROLL_RAD).clamped()
    assert math.hypot(both.pitch, both.roll) == pytest.approx(MAX_TILT_RAD)
    assert both.pitch < MAX_PITCH_RAD, "combined tilt must scale each axis down"


def test_clamping_preserves_the_commanded_tilt_direction():
    p = BodyPose(pitch=math.radians(40), roll=math.radians(20)).clamped()
    assert p.roll / p.pitch == pytest.approx(0.5, rel=1e-6)


def test_height_cannot_crouch_below_nominal():
    """Nominal stance is already ~70% extended; crouching eats wing travel."""
    assert BodyPose(z=-50.0).clamped().z == pytest.approx(MIN_HEIGHT_MM)


@pytest.mark.parametrize("yaw_deg", [0.0, 45.0, 135.0, -135.0, -45.0])
def test_body_to_leg_round_trips(yaw_deg):
    shoulder = np.array([75.0, -75.0, 0.0])
    yaw = math.radians(yaw_deg)
    local = body_to_leg(P, shoulder, yaw)
    assert leg_to_body(local, shoulder, yaw) == pytest.approx(P)


def test_rotz_is_orthonormal():
    r = rotz_xy(0.7)
    assert (r @ r.T) == pytest.approx(np.eye(2))
