"""Jacobian, Jacobian-based IK, and statics for the leg.

The Jacobian relates joint velocities to end-effector velocities and is the
core building block for both velocity IK (via J^+) and damped least-squares
(via J^T (J J^T + λI)^-1). This module hosts three things built on it:

- ``leg_jacobian`` — the analytic 3×3 geometric Jacobian (∂foot/∂θ), derived
  in closed form from the link geometry in ``CONFIG.links_mm``.
- ``dls_ik`` — damped-least-squares numerical IK, seedable from the current
  pose (a smooth, singularity-robust alternative to the analytic ``ik.py``).
- ``foot_force`` — LORIS-style statics: map measured joint torque (from the
  Dynamixel present-current) through the Jacobian to a Cartesian foot force.

Everything is in the leg-local frame, mm, radians — same conventions as
``fk.py`` and ``ik.py``.
"""
import numpy as np

from quadruped.config import CONFIG
from quadruped.kinematics.fk import L1, L2, foot_position, rot_x, rot_z
from quadruped.kinematics.ik import joint_angles

_X = np.array([1.0, 0.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


def leg_jacobian(shoulder: float, wing: float, knee: float) -> np.ndarray:
    """Analytic geometric Jacobian, columns = ∂foot/∂θ_i, in mm/rad.

    Revolute cross-product form: column i is ``axis_i × (foot - origin_i)``,
    where ``axis_i`` / ``origin_i`` are the i-th joint's rotation axis and a
    point on it, both in the leg-local frame.

      - shoulder: Z hinge through the origin.
      - wing, knee: X hinges, but *rotated by the shoulder* — their axes are
        both ``Rz(θ1)·x̂`` (the knee's is unchanged by the wing because
        ``Rx(θ2)·x̂ = x̂``), which is why axis2 and axis3 coincide.
    """
    foot = foot_position(shoulder, wing, knee)
    Rz = rot_z(shoulder)
    axis_wk = Rz @ _X                      # shared wing/knee axis
    o1 = np.zeros(3)
    o2 = Rz @ L1                           # wing joint position
    o3 = Rz @ (L1 + rot_x(wing) @ L2)      # knee joint position
    return np.column_stack([
        np.cross(_Z, foot - o1),
        np.cross(axis_wk, foot - o2),
        np.cross(axis_wk, foot - o3),
    ])


# Damping λ for dls_ik, in mm/rad. J·Jᵀ has units mm²/rad², so λ shares J's
# units. A conditioning sweep over the whole joint box (scripts/, not shipped)
# gives σ_min ∈ [3.5, 73] mm/rad and cond(J) ≤ ~70 — the leg is well-conditioned
# everywhere in-limits; the pseudo-inverse only degrades for *out-of-reach*
# targets (leg driven straight past its limits). λ=2.0, just under the
# workspace-edge σ_min ≈ 3.5, leaves the step essentially unattenuated inside
# the reachable set yet keeps it bounded (max gain 1/2λ) when a target pushes
# outside it.
_DLS_DAMPING = 2.0


def dls_ik(
    target_xyz,
    q0: tuple[float, float, float] | None = None,
    *,
    damping: float = _DLS_DAMPING,
    tol: float = 1e-3,
    max_iters: int = 100,
) -> tuple[float, float, float]:
    """Damped-least-squares numerical IK — a drop-in for ``ik.joint_angles``.

    Iterates ``q += Jᵀ (J Jᵀ + λ²I)⁻¹ (target − foot(q))`` from a seed until the
    Cartesian error falls under ``tol`` mm (or ``max_iters`` is hit, in which
    case the best-effort last ``q`` is returned — mirroring analytic IK's silent
    degradation on unreachable targets).

    Why this exists alongside the exact analytic solver (this is a square 3×3
    system, so it is *not* resolving redundancy):
      - **Continuity.** Seed ``q0`` from the current pose and the solver tracks
        the same branch between commands, instead of the analytic solver's
        single-branch jumps.
      - **Graceful edges.** Damping bends toward the nearest reachable point
        near a singularity / out-of-reach target rather than snapping like the
        analytic ``sqrt(max(...,0))``.
      - **Shared kernel.** The reusable core for future velocity / whole-body IK.
    """
    target = np.asarray(target_xyz, dtype=float)
    q = np.array(q0 if q0 is not None else joint_angles(*target), dtype=float)
    eye = np.eye(3)
    for _ in range(max_iters):
        err = target - foot_position(*q)
        if np.linalg.norm(err) < tol:
            break
        J = leg_jacobian(*q)
        q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * eye, err)
    return float(q[0]), float(q[1]), float(q[2])


# XL430-W250 gear-output torque constant, N·m/A. Default sourced from
# CONFIG.actuator (configs/robot.yaml); host-only, never in the firmware header.
KT = CONFIG.actuator.torque_constant_nm_per_a


def foot_force(
    q_triple,
    current_triple,
    *,
    kt: float = KT,
    rcond: float = 1e-3,
) -> np.ndarray:
    """Estimate the Cartesian foot force (N, leg-local) from joint currents.

    Statics: joint torque ``τ = kt·current`` balances the foot force ``f`` via
    ``τ = Jᵀ f``, so ``f = (Jᵀ)⁺ τ``. LORIS-style proprioceptive contact sensing
    — no load cell.

    Units: ``leg_jacobian`` is mm/rad, so it is scaled to m/rad (``/1000``)
    before the statics — with ``J`` in m and ``f`` in N, ``Jᵀf`` comes out in
    N·m (radian is dimensionless, so the mm→m conversion is the *entire* unit
    fix). ``pinv`` (not a raw inverse) keeps the estimate bounded near singular
    poses instead of producing phantom forces.

    ``current_triple`` is expected already signed / direction-corrected, as
    ``DynamixelBackend.present_current`` returns it. This is a coarse estimate
    (no friction or gearbox-efficiency model) — good for contact detection and
    order-of-magnitude force, not a calibrated load cell.

    Two caveats to know before trusting the numbers:

    - **No gravity compensation.** ``τ`` is the *total* joint torque, which
      includes whatever the servos draw to hold the leg's own weight against
      gravity — not just the contact reaction. So a lifted, no-contact leg still
      reports a nonzero force. Subtracting the gravity/holding torque is out of
      scope here; for touchdown detection, key off the *change* in the estimate,
      not its absolute value.
    - **Sign convention is not yet hardware-validated.** As written, ``f`` is the
      force the foot exerts *on the environment* (the ground reaction is its
      negative). The true sign also depends on the XL430 present-current
      polarity, which is not confirmed on this robot — treat the direction as
      provisional until checked against a known load during bringup.
    """
    J_m = leg_jacobian(*q_triple) / 1000.0            # mm/rad → m/rad
    tau = kt * np.asarray(current_triple, dtype=float)  # N·m
    return np.linalg.pinv(J_m.T, rcond=rcond) @ tau     # N
