"""Guided per-leg motor calibration.

For each joint in a leg, find its `direction` and `offset_deg` by jogging it to
the zero pose and confirming the + sense, then write the results into
configs/robot.yaml (run scripts/codegen.py afterwards).

Hardware tool — drives a U2D2 via DynamixelBackend.

    python scripts/calibrate.py --leg FR -p /dev/cu.usbserial-XXXX

Have `python scripts/view.py --model quad` open as the visual reference for
"zero pose" and the model's + direction. FL is already calibrated; do FR, BL, BR.
"""
import argparse
import sys
from pathlib import Path

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.backends import DynamixelBackend
from quadruped.calibration import ask, calibrate_joint, patch_robot_yaml
from quadruped.config import CONFIG

YAML_PATH = Path(__file__).resolve().parents[1] / "configs" / "robot.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leg", choices=CONFIG.legs, required=True)
    parser.add_argument("--port", "-p", default=None, help="U2D2 serial port (or SERIAL_PORT).")
    parser.add_argument("--test-angle", type=float, default=0.3,
                        help="Angle (rad) used for the direction check (default 0.3).")
    args = parser.parse_args()

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
    if ask("\nWrite these into configs/robot.yaml? [y/n]: ") == "y":
        YAML_PATH.write_text(patch_robot_yaml(YAML_PATH.read_text(), results))
        print(f"  wrote {YAML_PATH}\n"
              f"  next: run  python scripts/codegen.py  to regenerate firmware + config")
    else:
        print("  not written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
