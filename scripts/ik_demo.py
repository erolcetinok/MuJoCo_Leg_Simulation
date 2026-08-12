"""Single-leg IK demo — step or lissajous foot path, solved live in the viewer.

Sweeps a parametric foot path, solves analytic IK each frame, and shows the
result in the MuJoCo passive viewer.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from quadruped.config import CONFIG
from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.ik import joint_angles
from quadruped.sim.env import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["step", "lissajous"], default="step")
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    model, data = load_model()
    # qpos addresses for the leg joints, in CONFIG order (shoulder, wing, knee).
    # Only joints actually present in the loaded MJCF survive — lets this demo
    # work against both single_leg.xml and the full quadruped.
    qpos_idx = []
    for j in CONFIG.joints:
        try:
            qpos_idx.append(int(model.joint(j.mjcf_name).qposadr[0]))
        except KeyError:
            continue

    # Perturb the foot around its neutral (zero-pose) position, in the hip frame.
    target_center = foot_position(0.0, 0.0, 0.0)

    # path knobs
    step_freq, step_amp_x, step_amp_y, lift_amp_z = 0.6, 35.0, 15.0, 25.0
    amp_xyz = np.array([30.0, 20.0, 20.0])
    freq_xyz = np.array([0.55, 0.37, 0.73])
    phase_xyz = np.array([0.0, np.pi / 3, np.pi / 6])

    t = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            if args.path == "lissajous":
                offset = amp_xyz * np.sin(2 * np.pi * freq_xyz * t + phase_xyz)
            else:  # step
                theta = 2 * np.pi * step_freq * t
                offset = np.array([
                    step_amp_x * np.sin(theta),
                    step_amp_y * np.cos(theta),
                    lift_amp_z * 0.5 * (1.0 - np.cos(theta)),
                ])
            target_mm = target_center + offset
            for idx, angle in zip(qpos_idx, joint_angles(*target_mm)):
                data.qpos[idx] = angle
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(args.dt)
            t += args.dt
    return 0


if __name__ == "__main__":
    sys.exit(main())
