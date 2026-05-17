"""Forward kinematics: thin wrapper around mujoco.mj_forward + site readback.

The MJCF model is the source of geometric truth, so FK is just "set qpos,
call mj_forward, read site_xpos." When/if we switch to analytic FK (no
MuJoCo dependency) it will live alongside this in the same module.
"""
from __future__ import annotations

import numpy as np
import mujoco


def foot_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q: np.ndarray,
    *,
    foot_site_name: str = "foot_site",
    joint_names: tuple[str, ...] = ("shoulder_joint", "wing_joint", "knee_joint"),
) -> np.ndarray:
    """Return (x, y, z) world position of foot_site for joint angles q (rad)."""
    qpos_idxs = np.array(
        [model.joint(n).qposadr[0] for n in joint_names], dtype=int
    )
    data.qpos[qpos_idxs] = q
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site(foot_site_name).id].copy()
