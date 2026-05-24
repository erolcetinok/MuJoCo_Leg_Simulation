"""Swing-foot trajectory tests.

Verifies that SwingFootTrajectory satisfies the boundary conditions used to
derive its control points:
  * endpoint positions match lift_pos / touch_pos
  * endpoint X/Y velocities match -body_velocity (C^1 with stance)
  * endpoint Z velocity is zero (no vertical body motion / soft landing)
  * apex Z position equals apex_height
  * apex Z velocity is zero (foot momentarily stationary at top of arc)
  * endpoint physical velocities are independent of T_swing (chain-rule check)
"""
import numpy as np
import pytest

from quadruped.control.trajectory import Bezier1D, SwingFootTrajectory


# (lift_pos, touch_pos, apex_height, T_swing, body_velocity)
CONFIGS = [
    # In-place stepping
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.04, 0.3, (0.0, 0.0)),
    # Forward walk
    ((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0), 0.04, 0.3, (0.2, 0.0)),
    # Combined forward + strafe
    ((-0.04, -0.02, 0.0), (0.04, 0.02, 0.0), 0.03, 0.25, (0.15, 0.1)),
    # Different lift/touch heights (e.g. stepping onto / off a curb)
    ((0.0, 0.0, 0.01), (0.05, 0.0, -0.01), 0.05, 0.3, (0.1, 0.0)),
]


@pytest.mark.parametrize("lift_pos, touch_pos, apex, T, v", CONFIGS)
def test_position_endpoints(lift_pos, touch_pos, apex, T, v):
    traj = SwingFootTrajectory(lift_pos, touch_pos, apex, T, v)
    assert np.allclose(traj.position_at(0.0), lift_pos)
    assert np.allclose(traj.position_at(1.0), touch_pos)


@pytest.mark.parametrize("lift_pos, touch_pos, apex, T, v", CONFIGS)
def test_position_apex(lift_pos, touch_pos, apex, T, v):
    traj = SwingFootTrajectory(lift_pos, touch_pos, apex, T, v)
    assert np.isclose(traj.position_at(0.5)[2], apex)


@pytest.mark.parametrize("lift_pos, touch_pos, apex, T, v", CONFIGS)
def test_velocity_endpoints(lift_pos, touch_pos, apex, T, v):
    """X/Y match -body_velocity (C^1 with stance); Z is zero (no vertical body motion)."""
    traj = SwingFootTrajectory(lift_pos, touch_pos, apex, T, v)
    expected = np.array([-v[0], -v[1], 0.0])
    assert np.allclose(traj.velocity_at(0.0), expected)
    assert np.allclose(traj.velocity_at(1.0), expected)


@pytest.mark.parametrize("lift_pos, touch_pos, apex, T, v", CONFIGS)
def test_velocity_apex_z_zero(lift_pos, touch_pos, apex, T, v):
    """Foot is momentarily stationary vertically at the top of the arc.

    Checked from both sides of s=0.5 so we cover both the up- and down-branch.
    """
    traj = SwingFootTrajectory(lift_pos, touch_pos, apex, T, v)
    eps = 1e-9
    assert np.isclose(traj.velocity_at(0.5 - eps)[2], 0.0, atol=1e-6)
    assert np.isclose(traj.velocity_at(0.5 + eps)[2], 0.0, atol=1e-6)


def test_endpoint_velocity_independent_of_T_swing():
    """Physical endpoint velocity is -body_velocity regardless of T_swing.

    The control points contain a T_swing factor; the chain rule introduces a
    1/T_swing factor in velocity_at. If either is wrong, this test breaks.
    """
    lift = (-0.05, 0.0, 0.0)
    touch = (0.05, 0.0, 0.0)
    v_body = (0.2, 0.0)

    slow = SwingFootTrajectory(lift, touch, 0.04, 0.4, v_body)
    fast = SwingFootTrajectory(lift, touch, 0.04, 0.1, v_body)

    for traj in (slow, fast):
        assert np.isclose(traj.velocity_at(0.0)[0], -v_body[0])
        assert np.isclose(traj.velocity_at(1.0)[0], -v_body[0])


# ---------------------------------------------------------------------------
# Bezier1D — lower-level sanity checks
# ---------------------------------------------------------------------------

def test_bezier1d_endpoints_equal_endpoint_control_points():
    """B(0) = P0, B(1) = P_n for any control points."""
    b = Bezier1D([1.0, 4.0, -2.0, 7.0])
    assert np.isclose(b(0.0), 1.0)
    assert np.isclose(b(1.0), 7.0)


def test_bezier1d_derivative_reduces_degree():
    b = Bezier1D([1.0, 2.0, 3.0, 4.0])  # degree 3
    db = b.derivative()
    assert db.degree == 2


def test_bezier1d_doubled_endpoints_have_zero_endpoint_derivative():
    """The 'double up the endpoints' trick: P0=P1 and P_{n-1}=P_n -> B'(0)=B'(1)=0."""
    b = Bezier1D([2.0, 2.0, 5.0, 5.0])
    db = b.derivative()
    assert np.isclose(db(0.0), 0.0)
    assert np.isclose(db(1.0), 0.0)
