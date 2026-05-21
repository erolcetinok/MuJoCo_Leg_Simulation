"""Inverse kinematics — analytic solver, the exact inverse of fk.foot_position."""
import numpy as np
from quadruped.kinematics.fk import L1, L2, L3, rot_x

PX = L1[0] + L2[0] + L3[0]        # the leg's x is fixed — Rx never changes an x-component
LEN2 = np.hypot(L2[1], L2[2])     # length of L2 projected into the y-z plane
LEN3 = np.hypot(L3[1], L3[2])     # length of L3 projected into the y-z plane
PHI2 = np.arctan2(L2[2], L2[1])   # heading of L2 in the y-z plane
PHI3 = np.arctan2(L3[2], L3[1])   # heading of L3 in the y-z plane

def joint_angles(x: float, y: float, z: float) -> tuple[float, float, float]:
    py = -np.sqrt(max(x * x + y * y - PX * PX, 0.0))
    shoulder = np.arctan2(y, x) - np.arctan2(py, PX)
    p = np.array([PX, py, z])

    w = p - L1
    reach_sq = w[1] * w[1] + w[2] * w[2]
    cos_knee = (reach_sq - LEN2**2 - LEN3**2) / (2 * LEN2 * LEN3)  # law of cosines
    knee = (PHI2 - PHI3) + np.arccos(np.clip(cos_knee, -1.0, 1.0))

    v = L2 + rot_x(knee) @ L3
    wing = np.arctan2(w[2], w[1]) - np.arctan2(v[2], v[1])

    return shoulder, wing, knee
