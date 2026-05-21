"""Send a Cartesian foot target (x, y, z in mm).

Solves analytic IK for the single leg, then dispatches the joint angles to the
chosen backend (sim / hw / mirror).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from quadruped.cli._backends import add_backend_args, build_backend
from quadruped.config import CONFIG
from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.ik import joint_angles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="Target X (mm)")
    parser.add_argument("y", type=float, help="Target Y (mm)")
    parser.add_argument("z", type=float, help="Target Z (mm)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only solve IK and print angles; do not open any backend.")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    q = joint_angles(args.x, args.y, args.z)
    # FK the solution back: residual is ~0 for a reachable target, larger if the
    # solver clamped an out-of-reach one.
    residual = float(np.linalg.norm(foot_position(*q) - np.array([args.x, args.y, args.z])))

    print(f"target (mm): {args.x} {args.y} {args.z}")
    print(f"angles (rad): {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}")
    print(f"FK residual: {residual:.4e} mm")

    if args.dry_run:
        return 0

    q_dict = {name: float(v) for name, v in zip(CONFIG.joint_names, q)}
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
