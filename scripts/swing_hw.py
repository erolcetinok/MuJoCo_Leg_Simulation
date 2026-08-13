"""Play a full gait cycle (swing + stance) on hardware or sim.

Drives the single leg through swing-stance cycles, sampling at --rate Hz:
  - swing:  SwingFootTrajectory (cubic Bezier) from --from to --to, apex
            at --apex above lift_z at the midpoint.
  - stance: StanceFootTrajectory (linear) from --to back to --from at
            ground level.

body_velocity is chosen so the stance ends exactly at the lift position
and the next cycle begins where it left off.

Pre-poses to --from and waits 1 s before starting, so the first sample
isn't itself a step input from wherever the leg happened to be.

Examples:
    # Sim dry-run (prints all samples in one cycle)
    python scripts/swing_hw.py --from 0 -175 -50 --to 20 -175 -50 --dry-run

    # Sim with viewer, looping
    mjpython scripts/swing_hw.py --from -30 -175 -50 --to 30 -175 -50 --apex 30 --duration 0.8 --backend sim --viewer --loop

    # Real hardware, looping cycles
    python scripts/swing_hw.py --from -30 -175 -50 --to 30 -175 -50 --backend hw --loop
"""
import argparse
import sys
import time

import numpy as np

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.cli_args import add_backend_args, build_backend
from quadruped.config import CONFIG
from quadruped.control.trajectory import StanceFootTrajectory, SwingFootTrajectory
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
                        required=True, help="Swing lift-off / stance end position (mm)")
    parser.add_argument("--to", dest="touch", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        required=True, help="Swing touchdown / stance start position (mm)")
    parser.add_argument("--apex", type=float, default=10.0,
                        help="Swing apex height above lift_z (mm; default 10)")
    parser.add_argument("--duration", type=float, default=0.3,
                        help="Swing duration in seconds (default 0.3)")
    parser.add_argument("--stance-duration", type=float, default=None,
                        help="Stance duration in seconds (default = --duration)")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Sample rate in Hz (default 50)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop cycles until Ctrl+C (else one cycle then exit).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print samples; do not open any backend.")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    lift = tuple(args.lift)
    touch = tuple(args.touch)
    # SwingFootTrajectory treats apex_height as a lift ABOVE the foot, so pass
    # --apex straight through. apex_z is the resulting absolute z, for the IK check.
    apex_z = max(lift[2], touch[2]) + args.apex
    T_swing = args.duration
    T_stance = args.stance_duration if args.stance_duration is not None else T_swing
    # Close the cycle: pick body_velocity so stance brings the foot from
    # touch back to lift over T_stance.
    body_velocity = ((touch[0] - lift[0]) / T_stance, (touch[1] - lift[1]) / T_stance)

    print("IK pre-check:")
    _print_ik_check("lift",  *lift)
    _print_ik_check("apex",  (lift[0] + touch[0]) / 2, (lift[1] + touch[1]) / 2, apex_z)
    _print_ik_check("touch", *touch)

    swing = SwingFootTrajectory(lift, touch, args.apex, T_swing, body_velocity)
    stance = StanceFootTrajectory(touch, T_stance, body_velocity)

    dt = 1.0 / args.rate
    n_swing = int(round(T_swing * args.rate)) + 1
    n_stance = int(round(T_stance * args.rate)) + 1
    print(f"\nCycle: swing {n_swing} samples / {T_swing:.3f} s + "
          f"stance {n_stance} samples / {T_stance:.3f} s "
          f"({args.rate:.0f} Hz, dt = {dt*1000:.1f} ms)")
    print(f"body velocity: ({body_velocity[0]:.1f}, {body_velocity[1]:.1f}) mm/s")

    phases = (("swing", swing, n_swing), ("stance", stance, n_stance))

    if args.dry_run:
        for name, traj, n in phases:
            print(f"\nDry-run samples ({name}):")
            for i in range(n):
                s = i / (n - 1)
                p = traj.position_at(s)
                print(f"  s={s:.3f}  x={p[0]:7.2f}  y={p[1]:7.2f}  z={p[2]:7.2f}")
        return 0

    backend = build_backend(args)
    with backend:
        print(f"\nPre-pose -> {lift}, settling 1.0 s ...")
        backend.set_joint_targets(_angles_dict(*lift)[0])
        time.sleep(1.0)

        try:
            cycle = 0
            deadline = time.perf_counter()
            while True:
                print(f"cycle {cycle}")
                for name, traj, n in phases:
                    # Skip i=0: it equals the previous phase's last sample
                    # (or the pre-pose for the very first phase). Sending
                    # it again makes the motor hold for one dt at every
                    # phase boundary.
                    for i in range(1, n):
                        s = i / (n - 1)
                        p = traj.position_at(s)
                        backend.set_joint_targets(_angles_dict(float(p[0]), float(p[1]), float(p[2]))[0])
                        deadline += dt
                        slack = deadline - time.perf_counter()
                        if slack > 0:
                            time.sleep(slack)
                        else:
                            # Fell behind (round-trip > dt). Reset deadline so
                            # we don't sprint indefinitely trying to catch up.
                            deadline = time.perf_counter()
                cycle += 1
                if not args.loop:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
