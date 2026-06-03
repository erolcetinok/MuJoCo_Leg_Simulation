"""Guided per-leg motor calibration.

For each joint in a leg, find its `direction` and `offset_deg` by jogging it to
the zero pose and confirming the + sense, then write the results into
configs/robot.yaml (run quad-codegen afterwards).

Hardware tool — drives a U2D2 via DynamixelBackend.

    quad-calibrate --leg FR -p /dev/cu.usbserial-XXXX

Have `quad-view --model quad` open as the visual reference for "zero pose" and
the model's + direction. FL is already calibrated; do FR, BL, BR.
"""
import argparse
import math
import sys
from pathlib import Path

from quadruped.backends import DynamixelBackend
from quadruped.calibration import goal_ticks, patch_robot_yaml
from quadruped.config import CONFIG

YAML_PATH = Path(__file__).resolve().parents[3] / "configs" / "robot.yaml"


def _ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def _calibrate_joint(backend, joint, test_angle: float) -> tuple[int, float]:
    name = joint.name
    direction = joint.direction          # start from the current config guess
    offset = float(joint.offset_deg)
    print(f"\n=== {name} ===")

    # 1) Offset: jog until the joint physically sits at its zero pose.
    print("  OFFSET — jog until the joint is at its ZERO pose (match quad-view).")
    while True:
        backend.write_goal_ticks(name, goal_ticks(direction, offset, 0.0))
        ans = _ask(f"    offset={offset:6.1f}deg  [+N/-N to nudge, Enter re-send, 'ok']: ")
        if ans == "ok":
            break
        if ans == "":
            continue
        try:
            offset += float(ans)
        except ValueError:
            print("      ? type a nudge like +2, -0.5, or 'ok'")

    # 2) Direction: command +test_angle, confirm it matched the model's + sense.
    print(f"  DIRECTION — commanding +{math.degrees(test_angle):.0f}deg.")
    while True:
        backend.write_goal_ticks(name, goal_ticks(direction, offset, test_angle))
        ans = _ask("    moved the model's + direction (vs quad-view)? [y/n]: ")
        backend.write_goal_ticks(name, goal_ticks(direction, offset, 0.0))  # return to zero
        if ans == "y":
            break
        if ans == "n":
            direction = -direction
            print(f"      flipped -> direction={direction}")

    print(f"  -> {name}: direction={direction}  offset_deg={offset:.1f}")
    return direction, offset


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
                results[joint.name] = _calibrate_joint(backend, joint, args.test_angle)
    except KeyboardInterrupt:
        print("\nAborted; no changes written.")
        return 1

    print("\nResults:")
    for name, (d, o) in results.items():
        print(f"  {name}: direction={d:+d}  offset_deg={o:.1f}")
    if _ask("\nWrite these into configs/robot.yaml? [y/n]: ") == "y":
        YAML_PATH.write_text(patch_robot_yaml(YAML_PATH.read_text(), results))
        print(f"  wrote {YAML_PATH}\n  next: run  quad-codegen  to regenerate firmware + config")
    else:
        print("  not written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
