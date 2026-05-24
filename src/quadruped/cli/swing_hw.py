"""Play a swing-foot trajectory on hardware (or sim).

Drives the single leg from --from to --to along a SwingFootTrajectory (cubic
Bezier, apex_height above lift_z at the midpoint), sampling at --rate Hz.

Pre-poses to --from and waits 1 s before starting, so the first sample isn't
itself a step input from wherever the leg happened to be.

Examples:
    # Sim dry-run (just prints samples)
    quad-swing-hw --from 0 -175 -50 --to 20 -175 -50 --dry-run

    # Sim with viewer
    quad-swing-hw --from 0 -175 -50 --to 20 -175 -50 --backend sim --viewer

    # Real hardware, single play
    quad-swing-hw --from 0 -175 -50 --to 20 -175 -50 --backend hw

    # Real hardware, loop A<->B until Ctrl+C
    quad-swing-hw --from 0 -175 -50 --to 20 -175 -50 --backend hw --loop
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from quadruped.cli._backends import add_backend_args, build_backend
from quadruped.config import CONFIG
from quadruped.control.trajectory import SwingFootTrajectory
from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.ik import joint_angles


def _angles_dict(x, y, z):
    q = joint_angles(x, y, z)
    return {name: float(v) for name, v in zip(CONFIG.joint_names, q)}, q


def _print_ik_check(label, x, y, z):
    q_dict, q = _angles_dict(x, y, z)
    residual = float(np.linalg.norm(foot_position(*q) - np.array([x, y, z])))
    print(f"  {label:<10} ({x:7.2f}, {y:7.2f}, {z:7.2f}) -> "
          f"angles ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f})  residual {residual:.2e} mm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="lift", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        required=True, help="Lift-off position (mm)")
    parser.add_argument("--to", dest="touch", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        required=True, help="Touchdown position (mm)")
    parser.add_argument("--apex", type=float, default=10.0,
                        help="Apex height above lift_z (mm; default 10)")
    parser.add_argument("--duration", type=float, default=0.3,
                        help="Swing duration in seconds (default 0.3)")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Sample rate in Hz (default 50)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop A->B->A->B... instead of one-shot.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print samples; do not open any backend.")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    lift = tuple(args.lift)
    touch = tuple(args.touch)
    apex_height = lift[2] + args.apex

    # Sanity-check IK at the three corner positions before opening any port.
    print("IK pre-check:")
    _print_ik_check("lift",  *lift)
    _print_ik_check("apex",  (lift[0] + touch[0]) / 2, (lift[1] + touch[1]) / 2, apex_height)
    _print_ik_check("touch", *touch)

    traj_fwd  = SwingFootTrajectory(lift, touch, apex_height, args.duration)
    traj_back = SwingFootTrajectory(touch, lift, apex_height, args.duration)

    dt = 1.0 / args.rate
    n_samples = int(round(args.duration * args.rate)) + 1
    print(f"\nTrajectory: {n_samples} samples over {args.duration:.3f} s "
          f"({args.rate:.0f} Hz, dt = {dt*1000:.1f} ms)")

    if args.dry_run:
        print("\nDry-run samples (fwd):")
        for i in range(n_samples):
            s = i / (n_samples - 1)
            p = traj_fwd.position_at(s)
            print(f"  s={s:.3f}  x={p[0]:7.2f}  y={p[1]:7.2f}  z={p[2]:7.2f}")
        return 0

    # Pass dt through to MujocoBackend so the sim advances real-time per command,
    # otherwise mj_step only ticks 2 ms per call and the viewer barely moves.
    backend = build_backend(args, sim_step_dt=dt)
    with backend:
        print(f"\nPre-pose -> {lift}, settling 1.0 s ...")
        backend.set_joint_targets(_angles_dict(*lift)[0])
        time.sleep(1.0)

        try:
            cycle = 0
            while True:
                direction = "fwd" if cycle % 2 == 0 else "back"
                traj = traj_fwd if cycle % 2 == 0 else traj_back
                print(f"cycle {cycle} ({direction})")
                for i in range(n_samples):
                    s = i / (n_samples - 1)
                    p = traj.position_at(s)
                    backend.set_joint_targets(_angles_dict(float(p[0]), float(p[1]), float(p[2]))[0])
                    time.sleep(dt)
                cycle += 1
                if not args.loop:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
