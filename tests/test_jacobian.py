"""Jacobian / DLS-IK / foot-force tests.

  * test_jacobian_matches_finite_difference — the analytic leg_jacobian must
    equal a central-difference of foot_position (independent oracle).
  * test_dls_ik_round_trip — FK(dls_ik(p)) recovers p, from both the default
    (analytic) seed and a supplied seed, and stays finite near full extension.
  * test_foot_force_roundtrip — the statics unit chain inverts cleanly:
    currents synthesised from a known force map back to that force.
"""
import numpy as np
import pytest

from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.jacobian import KT, dls_ik, foot_force, leg_jacobian

# (shoulder, wing, knee) in radians — within joint limits.
POSES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, -1.5),
    (-1.2, 0.6, -1.8),
    (0.7, -0.4, 0.9),
]


def _fd_jacobian(shoulder, wing, knee, eps=1e-5):
    """Central-difference ∂foot/∂θ. eps≈1e-5 sits near the roundoff optimum:
    truncation ~eps²·f''' and roundoff ~ε_mach·|foot|/eps both stay << 1e-6."""
    angles = [shoulder, wing, knee]
    cols = []
    for i in range(3):
        up, dn = list(angles), list(angles)
        up[i] += eps
        dn[i] -= eps
        cols.append((foot_position(*up) - foot_position(*dn)) / (2 * eps))
    return np.column_stack(cols)


@pytest.mark.parametrize("angles", POSES)
def test_jacobian_matches_finite_difference(angles):
    analytic = leg_jacobian(*angles)
    numeric = _fd_jacobian(*angles)
    assert np.allclose(analytic, numeric, atol=1e-6), (
        f"angles={angles}: analytic J\n{analytic}\n!= finite-diff\n{numeric}\n"
        f"(max err {np.max(np.abs(analytic - numeric)):.3e} mm/rad)"
    )


@pytest.mark.parametrize("angles", POSES)
def test_dls_ik_round_trip(angles):
    """FK(dls_ik(target)) lands the foot back on target, for both seeds."""
    target = foot_position(*angles)

    for label, q0 in (("default-seed", None), ("supplied-seed", angles)):
        solved = dls_ik(target, q0=q0)
        recovered = foot_position(*solved)
        assert np.allclose(recovered, target, atol=1e-3), (
            f"{label} target={target}: FK(dls_ik)={recovered} "
            f"(err {np.linalg.norm(recovered - target):.3e} mm)"
        )


def test_dls_ik_converges_from_poor_seed():
    """A far-off seed still converges (exercises the iteration loop)."""
    target = foot_position(-1.2, 0.6, -1.8)
    solved = dls_ik(target, q0=(0.0, 0.0, 0.0))
    assert np.linalg.norm(foot_position(*solved) - target) < 1e-3


def test_dls_ik_stays_finite_out_of_reach():
    """Damping must keep the solve bounded (finite) on an unreachable target."""
    solved = dls_ik((0.0, -400.0, -400.0))
    assert np.all(np.isfinite(solved))


@pytest.mark.parametrize("angles", [
    (0.3, -0.2, 0.4),
    (0.0, 0.0, 0.5),
    (-0.5, 0.3, -0.8),
])
def test_foot_force_roundtrip(angles):
    """Synthesise the currents a known force would produce, then recover it.

    tau = (J[m/rad])ᵀ · f ;  current = tau / KT ;  foot_force(current) ≈ f.
    Locks the mm→m conversion and the pinv statics path together.
    """
    f_true = np.array([2.0, -5.0, 10.0])            # N, leg-local
    J_m = leg_jacobian(*angles) / 1000.0            # m/rad
    tau = J_m.T @ f_true                            # N·m
    current = tau / KT                              # A
    f_hat = foot_force(angles, current)
    assert np.allclose(f_hat, f_true, atol=1e-6), (
        f"angles={angles}: foot_force={f_hat} != {f_true} "
        f"(err {np.linalg.norm(f_hat - f_true):.3e} N)"
    )


def test_kt_matches_config():
    from quadruped.config import CONFIG
    assert KT == CONFIG.actuator.torque_constant_nm_per_a
