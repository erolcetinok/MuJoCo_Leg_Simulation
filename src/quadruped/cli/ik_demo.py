"""Single-leg IK demo — step or lissajous foot path, solved live in the viewer.

Replaces the old scripts/ik_step.py with a thin wrapper around
DampedLeastSquaresIK.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from quadruped.config import CONFIG
from quadruped.kinematics.ik import DampedLeastSquaresIK
from quadruped.sim.env import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["step", "lissajous"], default="step")
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    model, data = load_model()
    ik = DampedLeastSquaresIK(
        model, data,
        foot_site_name=CONFIG.mjcf.foot_site_name,
        target_site_name=CONFIG.mjcf.target_site_name,
        joint_names=CONFIG.mjcf.joint_names,
        alpha=0.6,
        damping=1e-2,
        tol_mm=1e-2,
        max_iter=1,  # one Jacobian step per render frame keeps motion smooth
    )

    target_center = np.array([0.0, 0.0, 150.0])
    site_offset = np.asarray(CONFIG.target_site_offset_mm)

    # path knobs
    step_freq, step_amp_x, step_amp_y, lift_amp_z = 0.6, 35.0, 15.0, 25.0
    amp_xyz = np.array([30.0, 0.0, 0.0])
    freq_xyz = np.array([0.55, 0.37, 0.73])
    phase_xyz = np.array([0.0, np.pi / 3, np.pi / 6])

    t = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            if args.path == "lissajous":
                x = amp_xyz[0] * np.sin(2 * np.pi * freq_xyz[0] * t + phase_xyz[0])
                y = amp_xyz[1] * np.sin(2 * np.pi * freq_xyz[1] * t + phase_xyz[1])
                z = amp_xyz[2] * np.sin(2 * np.pi * freq_xyz[2] * t + phase_xyz[2])
            else:  # step
                theta = 2 * np.pi * step_freq * t
                x = step_amp_x * np.sin(theta)
                y = step_amp_y * np.cos(theta)
                z = lift_amp_z * 0.5 * (1.0 - np.cos(theta))
            target_mm = target_center + np.array([x, y, z])
            ik.set_mocap_target(target_mm, site_offset)
            ik.solve(target_mm)
            viewer.sync()
            time.sleep(args.dt)
            t += args.dt
    return 0


if __name__ == "__main__":
    sys.exit(main())
