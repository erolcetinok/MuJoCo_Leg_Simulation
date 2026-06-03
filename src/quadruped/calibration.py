"""Pure helpers for the guided motor calibration (quad-calibrate).

Kept separate from the interactive CLI so the math and the YAML edit are unit
tested; only the prompt/loop in cli/calibrate.py is untested.
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
