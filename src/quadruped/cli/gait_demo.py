"""Full quadruped gait demo — walk or trot, IK-driven, live in the viewer.

Drives the four legs through GaitScheduler (cubic Bezier swing trajectories
+ Raibert foothold planning), feeds the resulting per-leg foot positions to
MuJoCo mocap targets, and runs Jacobian IK to track them.

The body itself is fixed in this demo; --vx and --vy command an effective body
velocity that the scheduler uses for foothold planning. With both at 0 the
feet step in place; nonzero velocity makes the feet step forward/back as if
the body were translating beneath them.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from quadruped.control.gait import GaitScheduler, TROT_PRESET, WALK_PRESET
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
    parser.add_argument("--vx", type=float, default=0.0,
                        help="Commanded body forward velocity (units/sec; 0 = in-place stepping)")
    parser.add_argument("--vy", type=float, default=0.0,
                        help="Commanded body lateral velocity (units/sec)")
    args = parser.parse_args()

    preset = WALK_PRESET if args.gait == "walk" else TROT_PRESET

    model = mujoco.MjModel.from_xml_path(str(model_path("quadruped")))
    data = mujoco.MjData(model)
    info = _build_leg_info(model)

    alpha, damping, tol = 0.3, 1e-2, 0.5

    # The body is fixed at the origin in this demo, so the initial mocap
    # positions (world frame) double as the legs' home positions (body frame).
    initial_mocap = {leg: data.mocap_pos[info[leg]["mocap_id"]].copy() for leg in LEGS}
    scheduler = GaitScheduler(**preset, home_positions=initial_mocap)
    body_velocity = (args.vx, args.vy)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            scheduler.advance(args.dt, body_velocity)
            for leg in LEGS:
                data.mocap_pos[info[leg]["mocap_id"]] = scheduler.foot_position(leg)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
