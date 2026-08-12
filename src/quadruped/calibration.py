"""Guided motor calibration: tick math, the YAML edit, and the per-joint routine.

`scripts/calibrate.py` is only argparse plus the run loop — everything with
behaviour worth testing lives here. Prompting is confined to `ask`, so
`calibrate_joint` is driven in tests by monkeypatching `input`.
"""
from __future__ import annotations

import math
import re

TICKS_PER_REV = 4096
TICK_MIN, TICK_MAX = 0, 4095


def goal_ticks(direction: int, offset_deg: float, q_rad: float, *, clamp: bool = True) -> int:
    """Tick goal for a TRIAL (direction, offset_deg) at commanded angle q_rad.

    Same formula as the firmware / DynamixelBackend, but with direction/offset
    passed in so the tool can probe candidate values before they're in config.
    """
    deg = direction * math.degrees(q_rad) + offset_deg
    ticks = int(round(deg * TICKS_PER_REV / 360.0))
    if clamp:
        ticks = max(TICK_MIN, min(TICK_MAX, ticks))
    return ticks


_NUM = r"-?\d+(?:\.\d+)?"


def patch_robot_yaml(text: str, updates: dict[str, tuple[int, float]]) -> str:
    """Return `text` with each joint's `direction` / `offset_deg` replaced.

    updates: {joint_name: (direction, offset_deg)}. Edits only the matched
    numbers on the line whose `name:` is the joint — everything else (motor_id,
    limits, spacing, other joints) is left byte-for-byte intact. Raises KeyError
    if any requested joint name isn't found.
    """
    remaining = dict(updates)
    out = []
    for line in text.splitlines(keepends=True):
        for jname, (direction, offset_deg) in list(remaining.items()):
            if re.search(rf"name:\s*{re.escape(jname)}\b", line):
                line = re.sub(rf"direction:\s*{_NUM}", f"direction: {int(direction)}", line)
                line = re.sub(rf"offset_deg:\s*{_NUM}", f"offset_deg: {offset_deg:.1f}", line)
                del remaining[jname]
                break
        out.append(line)
    if remaining:
        raise KeyError(f"joints not found in YAML: {sorted(remaining)}")
    return "".join(out)


def ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def calibrate_joint(backend, joint, test_angle: float) -> tuple[int, float]:
    name = joint.name
    direction = joint.direction          # start from the current config guess
    offset = float(joint.offset_deg)
    print(f"\n=== {name} ===")

    # 1) Offset: jog until the joint physically sits at its zero pose.
    print("  OFFSET — jog until the joint is at its ZERO pose (match the viewer).")
    while True:
        backend.write_goal_ticks(name, goal_ticks(direction, offset, 0.0))
        ans = ask(f"    offset={offset:6.1f}deg  [+N/-N to nudge, Enter re-send, 'ok']: ")
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
        ans = ask("    moved the model's + direction (vs the viewer)? [y/n]: ")
        backend.write_goal_ticks(name, goal_ticks(direction, offset, 0.0))  # return to zero
        if ans == "y":
            break
        if ans == "n":
            direction = -direction
            print(f"      flipped -> direction={direction}")

    print(f"  -> {name}: direction={direction}  offset_deg={offset:.1f}")
    return direction, offset
