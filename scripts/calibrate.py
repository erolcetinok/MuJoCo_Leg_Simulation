"""Guided per-leg motor calibration.

For each joint in a leg, find its `direction` and `offset_deg` by jogging it to
the zero pose and confirming the + sense, then write the results into
configs/robot.yaml (run scripts/codegen.py afterwards).

Hardware tool — drives a U2D2 via DynamixelBackend.

    python scripts/calibrate.py --leg FR -p /dev/cu.usbserial-XXXX
    python scripts/calibrate.py --verify --leg FR      # check it took

Have `mjpython scripts/view.py --model quad` open as the visual reference for
"zero pose" and the model's + direction.

Every leg needs this, FL included: the 180.0 / +1 values in robot.yaml are the
"horn mounted at the electrical midpoint" assumption, never a measurement. The
mirrored right-side legs are the ones most likely to come out at direction -1.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.backends import DynamixelBackend
from quadruped.calibration import ask, calibrate_joint, patch_robot_yaml, verify_pose
from quadruped.config import CONFIG

REPO = Path(__file__).resolve().parents[1]
YAML_PATH = REPO / "configs" / "robot.yaml"
CODEGEN = REPO / "scripts" / "codegen.py"


def run_verify(args) -> int:
    """Command the leg to its zero pose and report measured-vs-commanded error.

    A joint whose offset is wrong by 180 deg, or whose direction is inverted,
    still *looks* fine in the config; it shows up here as a large residual.
    """
    joints = list(CONFIG.joints_for_leg(args.leg))
    backend = DynamixelBackend(port=args.port, profile_velocity=30)
    with backend:
        print(f"Commanding {args.leg} to its zero pose. Keep clear.")
        rows = verify_pose(backend, {j.name: 0.0 for j in joints})
        time.sleep(1.0)                      # let the gentle move finish
        rows = verify_pose(backend, {j.name: 0.0 for j in joints})

    print(f"\n{'joint':<12} {'cmd(deg)':>9} {'meas(deg)':>10} {'err(deg)':>9}")
    worst = 0.0
    for name, cmd, meas, err in rows:
        if meas is None:
            print(f"{name:<12} {cmd:>9.1f} {'-':>10} {'NO REPLY':>9}")
            worst = float("inf")
            continue
        print(f"{name:<12} {cmd:>9.1f} {meas:>10.1f} {err:>9.1f}")
        worst = max(worst, abs(err))
    if worst > 5.0:
        print(f"\nFAIL: worst error {worst:.1f} deg. Re-run calibration for this leg.")
        return 1
    print(f"\nOK: worst error {worst:.1f} deg.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leg", choices=CONFIG.legs, required=True)
    parser.add_argument("--port", "-p", default=None, help="U2D2 serial port (or SERIAL_PORT).")
    parser.add_argument("--test-angle", type=float, default=0.3,
                        help="Angle (rad) used for the direction check (default 0.3).")
    parser.add_argument("--verify", action="store_true",
                        help="Skip calibration: command the leg to zero and report the "
                             "measured-vs-commanded error for each joint.")
    args = parser.parse_args()

    if args.verify:
        return run_verify(args)

    backend = DynamixelBackend(port=args.port, profile_velocity=30)  # gentle moves
    results: dict[str, tuple[int, float]] = {}
    try:
        with backend:
            print(f"Calibrating leg {args.leg}. Keep clear — servos move gently. Ctrl-C to abort.")
            for joint in CONFIG.joints_for_leg(args.leg):
                results[joint.name] = calibrate_joint(backend, joint, args.test_angle)
    except KeyboardInterrupt:
        print("\nAborted; no changes written.")
        return 1

    print("\nResults:")
    for name, (d, o) in results.items():
        print(f"  {name}: direction={d:+d}  offset_deg={o:.1f}")
    if ask("\nWrite these into configs/robot.yaml? [y/n]: ") != "y":
        print("  not written.")
        return 0

    YAML_PATH.write_text(patch_robot_yaml(YAML_PATH.read_text(), results))
    print(f"  wrote {YAML_PATH}")
    # Regenerate immediately: calibrate_joint seeds from the GENERATED config,
    # so leaving the two out of step means a second pass silently starts from
    # the old numbers.
    result = subprocess.run([sys.executable, str(CODEGEN)])
    if result.returncode != 0:
        print(f"  codegen FAILED — run  python {CODEGEN}  by hand before using this leg.")
        return 1
    print(f"  next: python scripts/calibrate.py --verify --leg {args.leg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
