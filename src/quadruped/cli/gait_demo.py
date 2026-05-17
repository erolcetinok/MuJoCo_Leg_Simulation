"""Full quadruped gait demo — walk or trot, IK-driven, live in the viewer.

Replaces scripts/ik_quadruped.py. The IK math is unchanged; the gait math
is now in quadruped.control.gait.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from quadruped.control.gait import (
    TROT_PARAMS,
    TROT_PHASE_OFFSETS,
    WALK_PARAMS,
    WALK_PHASE_OFFSETS,
    compute_gait_offset,
)
from quadruped.sim.env import model_path

LEGS = ("FL", "FR", "BL", "BR")


def _build_leg_info(model: mujoco.MjModel):
    info = {}
    for leg in LEGS:
        info[leg] = {
            "foot_site_id": model.site(f"foot_site_{leg}").id,
            "target_site_id": model.site(f"target_site_{leg}").id,
            "mocap_id": model.body(f"ik_target_{leg}").mocapid[0],
            "dof_idxs": np.array([
                model.joint(f"shoulder_joint_{leg}").dofadr[0],
                model.joint(f"wing_joint_{leg}").dofadr[0],
                model.joint(f"knee_joint_{leg}").dofadr[0],
            ], dtype=int),
            "qpos_idxs": np.array([
                model.joint(f"shoulder_joint_{leg}").qposadr[0],
                model.joint(f"wing_joint_{leg}").qposadr[0],
                model.joint(f"knee_joint_{leg}").qposadr[0],
            ], dtype=int),
            "jids": np.array([
                model.joint(f"shoulder_joint_{leg}").id,
                model.joint(f"wing_joint_{leg}").id,
                model.joint(f"knee_joint_{leg}").id,
            ], dtype=int),
        }
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait", choices=["walk", "trot"], default="walk")
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    params, offsets = (
        (WALK_PARAMS, WALK_PHASE_OFFSETS) if args.gait == "walk"
        else (TROT_PARAMS, TROT_PHASE_OFFSETS)
    )

    model = mujoco.MjModel.from_xml_path(str(model_path("quadruped")))
    data = mujoco.MjData(model)
    info = _build_leg_info(model)

    alpha, damping, tol = 0.3, 1e-2, 0.5

    initial_mocap = {leg: data.mocap_pos[info[leg]["mocap_id"]].copy() for leg in LEGS}
    t = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            phase = 2 * np.pi * params["step_frequency"] * t
            for leg in LEGS:
                lp = (phase + offsets[leg]) % (2 * np.pi)
                data.mocap_pos[info[leg]["mocap_id"]] = (
                    initial_mocap[leg] + compute_gait_offset(leg, lp, params)
                )
            mujoco.mj_forward(model, data)

            for leg in LEGS:
                d = info[leg]
                p_foot = data.site_xpos[d["foot_site_id"]].copy()
                p_target = data.site_xpos[d["target_site_id"]].copy()
                err = p_target - p_foot
                err_norm = float(np.linalg.norm(err))
                if err_norm > tol:
                    jacp = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jacp, None, d["foot_site_id"])
                    J = jacp[:, d["dof_idxs"]]
                    JJt = J @ J.T
                    dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)
                    scale = min(1.0, err_norm / 10.0)
                    q = data.qpos[d["qpos_idxs"]].copy() + alpha * scale * dq
                    for i, jid in enumerate(d["jids"]):
                        lo, hi = model.jnt_range[jid]
                        q[i] = float(np.clip(q[i], lo, hi))
                    data.qpos[d["qpos_idxs"]] = q

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(args.dt)
            t += args.dt
    return 0


if __name__ == "__main__":
    sys.exit(main())
