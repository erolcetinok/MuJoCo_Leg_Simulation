"""GaitScheduler tests.

Verifies GaitScheduler:
- advances phase by dt/period and wraps at 1.0
- stores body_velocity from advance()
- is_swing matches the [0, duty_factor) stance convention
- foot_position satisfies Raibert symmetry (foot at home at mid-stance)
- foot_velocity equals -body_velocity throughout stance
- swing apex (s=0.5) Z position = apex_height, Z velocity = 0
- C¹ continuity at the stance↔swing handoff in both directions
- Trot diagonal pairs (FL+BR, FR+BL) share contact state at every phase
"""
import numpy as np
import pytest

from quadruped.control.gait import (
    GaitScheduler,
    TROT_PRESET,
    WALK_PRESET,
)


HOME = {leg: np.array([0.0, 0.0, -100.0]) for leg in ("FL", "FR", "BL", "BR")}


def _walk():
    return GaitScheduler(**WALK_PRESET, home_positions=HOME)


def _trot():
    return GaitScheduler(**TROT_PRESET, home_positions=HOME)


# ---------------------------------------------------------------------------
# Construction and phase advancement
# ---------------------------------------------------------------------------

def test_initial_state():
    gs = _walk()
    assert gs.phase == 0.0
    assert gs.body_velocity == (0.0, 0.0)
    assert np.isclose(gs.T_stance + gs.T_swing, gs.period)


def test_advance_increments_phase():
    gs = _walk()  # period 1.0 → dt of 0.25 advances phase by 0.25
    gs.advance(0.25, (0.0, 0.0))
    assert np.isclose(gs.phase, 0.25)
    gs.advance(0.5, (0.0, 0.0))
    assert np.isclose(gs.phase, 0.75)


def test_advance_wraps_phase():
    gs = _walk()
    gs.advance(1.5, (0.0, 0.0))  # one full cycle plus 0.5
    assert np.isclose(gs.phase, 0.5)


def test_advance_stores_body_velocity():
    gs = _walk()
    gs.advance(0.1, (0.2, -0.3))
    assert gs.body_velocity == (0.2, -0.3)


# ---------------------------------------------------------------------------
# is_swing
# ---------------------------------------------------------------------------

def test_is_swing_stance_at_phase_zero():
    """FL has offset 0 and starts in stance (local_phase 0 < duty_factor 0.75)."""
    gs = _walk()
    assert gs.is_swing("FL") is False


def test_is_swing_swing_past_duty_factor():
    gs = _walk()
    gs.phase = 0.8  # FL local_phase = 0.8 ≥ 0.75
    assert gs.is_swing("FL") is True


def test_is_swing_boundary_at_duty_factor():
    """local_phase == duty_factor is the start of swing (docstring convention)."""
    gs = _walk()
    gs.phase = WALK_PRESET["duty_factor"]
    assert gs.is_swing("FL") is True


@pytest.mark.parametrize("phase", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
def test_trot_diagonal_pairs_synchronized(phase):
    """FL+BR share offset 0; FR+BL share offset 0.5. Pairs must always agree."""
    gs = _trot()
    gs.phase = phase
    assert gs.is_swing("FL") == gs.is_swing("BR")
    assert gs.is_swing("FR") == gs.is_swing("BL")


# ---------------------------------------------------------------------------
# foot_position
# ---------------------------------------------------------------------------

def test_at_rest_stance_foot_at_home():
    """v_body = 0 ⇒ stance foot is exactly at home."""
    gs = _walk()  # phase 0 → FL in stance
    assert np.allclose(gs.foot_position("FL"), HOME["FL"])


def test_at_rest_swing_apex_reaches_apex_height():
    """v_body = 0 ⇒ swing X/Y stay at home but Z arcs up to apex_height at s=0.5."""
    gs = _walk()
    # FL mid-swing: local_phase = duty_factor + 0.5 * (1 - duty_factor) = 0.875
    gs.phase = 0.875
    pos = gs.foot_position("FL")
    assert np.isclose(pos[0], HOME["FL"][0])
    assert np.isclose(pos[1], HOME["FL"][1])
    assert np.isclose(pos[2], WALK_PRESET["apex_height"])


@pytest.mark.parametrize("v_body", [(0.0, 0.0), (50.0, 0.0), (-30.0, 20.0)])
def test_raibert_mid_stance_foot_at_home(v_body):
    """Mid-stance (s=0.5) foot should sit at home for ANY v_body (Raibert symmetry)."""
    gs = _walk()
    gs.phase = WALK_PRESET["duty_factor"] / 2  # local_phase = duty/2 → s_stance = 0.5
    gs.body_velocity = v_body
    assert np.allclose(gs.foot_position("FL"), HOME["FL"])


def test_stance_starts_at_touch_pos():
    """Foot at local_phase=0 should be at touch_pos = home + v_body * T_stance/2."""
    gs = _walk()
    gs.advance(0.0, (50.0, 0.0))
    expected_x = HOME["FL"][0] + 50.0 * gs.T_stance / 2
    assert np.isclose(gs.foot_position("FL")[0], expected_x)


# ---------------------------------------------------------------------------
# foot_velocity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", [0.0, 0.2, 0.5, 0.7])
def test_stance_velocity_equals_neg_body_velocity(phase):
    """Stance velocity is constant -v_body throughout (X/Y axes; Z always 0)."""
    gs = _walk()
    gs.phase = phase
    gs.body_velocity = (50.0, -20.0)
    assert np.allclose(gs.foot_velocity("FL"), [-50.0, 20.0, 0.0])


def test_swing_apex_z_velocity_zero():
    """At swing apex, foot is momentarily stationary vertically."""
    gs = _walk()
    gs.phase = 0.875  # FL mid-swing
    gs.body_velocity = (50.0, 0.0)
    assert np.isclose(gs.foot_velocity("FL")[2], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# C¹ continuity at the stance↔swing handoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boundary_phase", [
    WALK_PRESET["duty_factor"],   # stance → swing transition for FL
    0.0,                          # swing → stance transition for FL (wrap)
])
def test_c1_continuity_across_handoff(boundary_phase):
    """Velocity equals -v_body from both sides of a stance↔swing handoff.

    The constant-velocity side (stance, or freshly-evaluated swing endpoint)
    matches -v_body exactly. The Bezier side at eps from its endpoint has Z
    velocity that has linearly ramped up to O(eps) — small but nonzero —
    hence the looser tolerance.
    """
    gs = _walk()
    gs.body_velocity = (50.0, 30.0)
    expected = np.array([-50.0, -30.0, 0.0])
    eps = 1e-9

    gs.phase = (boundary_phase - eps) % 1.0
    v_before = gs.foot_velocity("FL")

    gs.phase = (boundary_phase + eps) % 1.0
    v_after = gs.foot_velocity("FL")

    assert np.allclose(v_before, expected, atol=1e-4)
    assert np.allclose(v_after, expected, atol=1e-4)
