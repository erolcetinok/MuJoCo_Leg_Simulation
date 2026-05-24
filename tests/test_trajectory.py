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

from quadruped.control.trajectory import Bezier1D, StanceFootTrajectory, SwingFootTrajectory


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
# StanceFootTrajectory
# ---------------------------------------------------------------------------

# (touch_pos, T_stance, body_velocity)
STANCE_CONFIGS = [
    # In-place hold
    ((0.0, 0.0, 0.0), 0.3, (0.0, 0.0)),
    # Forward walk
    ((0.05, 0.0, 0.0), 0.3, (0.2, 0.0)),
    # Combined forward + strafe
    ((0.04, 0.02, 0.0), 0.25, (0.15, 0.1)),
    # Foot on a raised surface
    ((0.05, 0.0, 0.01), 0.3, (0.1, 0.0)),
]


@pytest.mark.parametrize("touch_pos, T, v", STANCE_CONFIGS)
def test_stance_position_endpoints(touch_pos, T, v):
    """Start at touch_pos; end at touch_pos - v_body * T_stance (= next swing's lift_pos)."""
    traj = StanceFootTrajectory(touch_pos, T, v)
    expected_end = np.array([touch_pos[0] - v[0] * T,
                             touch_pos[1] - v[1] * T,
                             touch_pos[2]])
    assert np.allclose(traj.position_at(0.0), touch_pos)
    assert np.allclose(traj.position_at(1.0), expected_end)


@pytest.mark.parametrize("touch_pos, T, v", STANCE_CONFIGS)
def test_stance_velocity_constant(touch_pos, T, v):
    """Velocity is -body_velocity at every s; Z is always 0 (foot on ground)."""
    traj = StanceFootTrajectory(touch_pos, T, v)
    expected = np.array([-v[0], -v[1], 0.0])
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(traj.velocity_at(s), expected)


@pytest.mark.parametrize("touch_pos, T, v", STANCE_CONFIGS)
def test_stance_z_constant(touch_pos, T, v):
    """Z position never changes during stance — foot is planted on the ground."""
    traj = StanceFootTrajectory(touch_pos, T, v)
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.isclose(traj.position_at(s)[2], touch_pos[2])


def test_stance_to_swing_handoff_is_c1():
    """Stance's end velocity must equal swing's start velocity — same -v_body for both."""
    touch = (0.05, 0.0, 0.0)
    T_stance = 0.3
    T_swing = 0.2
    v = (0.2, 0.1)

    stance = StanceFootTrajectory(touch, T_stance, v)
    # Next swing lifts from where stance ended, lands at some new foothold.
    lift = stance.position_at(1.0)
    swing = SwingFootTrajectory(lift, (0.1, 0.05, 0.0), 0.04, T_swing, v)

    assert np.allclose(stance.velocity_at(1.0), swing.velocity_at(0.0))


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
