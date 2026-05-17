"""Send a Cartesian foot target (x, y, z in mm).

Runs IK against the single-leg MuJoCo model, then dispatches the joint angles
to the chosen backend (sim / hw / mirror).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from quadruped.cli._backends import add_backend_args, build_backend
from quadruped.config import CONFIG
from quadruped.kinematics.ik import DampedLeastSquaresIK
from quadruped.sim.env import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="Target X (mm)")
    parser.add_argument("y", type=float, help="Target Y (mm)")
    parser.add_argument("z", type=float, help="Target Z (mm)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only solve IK and print angles; do not open any backend.")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    model, data = load_model(args.xml)
    ik = DampedLeastSquaresIK(
        model, data,
        foot_site_name=CONFIG.mjcf.foot_site_name,
        target_site_name=CONFIG.mjcf.target_site_name,
        joint_names=CONFIG.mjcf.joint_names,
    )
    target_mm = np.array([args.x, args.y, args.z])
    ik.set_mocap_target(target_mm, np.asarray(CONFIG.target_site_offset_mm))
    result = ik.solve(target_mm)

    print(f"target (mm): {args.x} {args.y} {args.z}")
    print(f"angles (rad): {result.q[0]:.4f} {result.q[1]:.4f} {result.q[2]:.4f}")
    print(f"converged={result.converged} iters={result.iterations} err={result.final_error_mm:.4f} mm")

    if args.dry_run:
        return 0

    q_dict = {name: float(v) for name, v in zip(CONFIG.joint_names, result.q)}
    backend = build_backend(args)
    with backend:
        backend.set_joint_targets(q_dict)
        # For hw / mirror backends print the firmware echo if available.
        for b in getattr(backend, "backends", [backend]):
            reply = getattr(b, "last_reply", "")
            if reply:
                print(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
