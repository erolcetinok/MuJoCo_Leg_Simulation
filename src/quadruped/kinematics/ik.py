"""Damped least-squares IK over a MuJoCo model.

Extracted from the original scripts/send_foot_position.py loop. The constants
(alpha, damping, tol) are tuned for the 3-DoF leg at mm scale; tweak in the
constructor if you change scale or model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class IKResult:
    q: np.ndarray            # solved joint angles (rad), in joint_names order
    iterations: int
    final_error_mm: float
    converged: bool


def clamp_to_limits(
    model: mujoco.MjModel,
    jids: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Clamp each joint angle to the MJCF's joint range."""
    out = q.copy()
    for i, jid in enumerate(jids):
        lo, hi = model.jnt_range[jid]
        out[i] = np.clip(out[i], lo, hi)
    return out


class DampedLeastSquaresIK:
    """3D positional IK via DLS using a MuJoCo Jacobian.

    Args:
        model, data: MuJoCo model and data.
        foot_site_name: site whose position we drive (e.g. "foot_site").
        target_site_name: site we read the target position from (typically a mocap site).
        joint_names: joints whose qpos we solve over, in order.
        alpha, damping, tol, max_iter: solver knobs.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        foot_site_name: str,
        target_site_name: str,
        joint_names: tuple[str, ...],
        *,
        alpha: float = 0.6,
        damping: float = 1e-2,
        tol_mm: float = 1e-2,
        max_iter: int = 200,
    ) -> None:
        self.model = model
        self.data = data
        self.foot_site_id = model.site(foot_site_name).id
        self.target_site_id = model.site(target_site_name).id
        self.dof_idxs = np.array(
            [model.joint(n).dofadr[0] for n in joint_names], dtype=int
        )
        self.qpos_idxs = np.array(
            [model.joint(n).qposadr[0] for n in joint_names], dtype=int
        )
        self.jids = np.array([model.joint(n).id for n in joint_names], dtype=int)
        self.alpha = alpha
        self.damping = damping
        self.tol_mm = tol_mm
        self.max_iter = max_iter

    def solve(self, target_world_mm: np.ndarray) -> IKResult:
        """Solve for joint angles so foot_site reaches target_world_mm.

        target_world_mm is written into mocap_pos with the YAML's
        target_site_offset_mm correction (see send_foot CLI for the math).
        """
        q = self.data.qpos[self.qpos_idxs].copy()
        err_norm = float("inf")
        i = 0
        for i in range(1, self.max_iter + 1):
            mujoco.mj_forward(self.model, self.data)
            p_foot = self.data.site_xpos[self.foot_site_id].copy()
            p_target = self.data.site_xpos[self.target_site_id].copy()
            err = p_target - p_foot
            err_norm = float(np.linalg.norm(err))
            if err_norm <= self.tol_mm:
                break
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.foot_site_id)
            J = jacp[:, self.dof_idxs]
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + self.damping * np.eye(3), err)
            q = clamp_to_limits(self.model, self.jids, q + self.alpha * dq)
            self.data.qpos[self.qpos_idxs] = q
        return IKResult(q=q, iterations=i, final_error_mm=err_norm, converged=err_norm <= self.tol_mm)

    def set_mocap_target(self, target_world_mm: np.ndarray, site_offset_mm: np.ndarray, mocap_id: int = 0) -> None:
        """Set mocap_pos so that target_site ends up at target_world_mm."""
        self.data.mocap_pos[mocap_id] = np.asarray(target_world_mm) - np.asarray(site_offset_mm)
