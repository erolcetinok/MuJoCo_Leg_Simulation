"""Getting into and out of the gait without snapping twelve servos.

The gait loop assumes the robot is already standing in its stance pose. Nothing
put it there: `teleop` and `gait_demo` used to issue a full-amplitude gait pose
on tick 0, from whatever pose the legs happened to be in, into servos with
PROFILE_VELOCITY=0. On hardware that is a full-speed snap.

`ramp_to` closes that gap by interpolating in joint space from where the servos
actually are (read over the bus) to where the gait wants them.

Note there is no true sit-down. The command envelope has no crouch authority
(`body.MIN_HEIGHT_MM == 0`) because the wing joint's +-50 deg is already ~65%
consumed by the gait, so "settle" means returning to the symmetric stance pose
before torque is released, not lowering the body. A real crouch needs
workspace-aware foot clamping, which does not exist yet.
"""
from __future__ import annotations

import time
from typing import Optional


def stance_pose(controller) -> dict:
    """The at-rest joint targets: zero velocity, no body pose, phase unchanged.

    dt=0 leaves the gait phase exactly where it is, so calling this before or
    after a run neither advances nor rewinds the gait.
    """
    return controller.step(0.0)


def ramp_to(
    backend,
    targets: dict,
    *,
    duration: float = 2.0,
    rate: float = 50.0,
    start: Optional[dict] = None,
) -> dict:
    """Interpolate every joint from its present angle to `targets` over `duration`.

    The start pose comes from the backend's own feedback when it has any; a
    backend with no read channel (sim) falls back to `start`, then to `targets`
    itself, which degenerates to a single write. Returns the targets actually
    commanded, so a caller can seed its own last-commanded state.
    """
    if start is None:
        try:
            qpos, _ = backend.read_joint_state()
        except Exception:
            qpos = {}
        start = qpos or {}
    # Any joint we cannot read starts at its target, i.e. it is not ramped.
    begin = {name: float(start.get(name, value)) for name, value in targets.items()}

    steps = max(1, int(round(duration * rate)))
    dt = 1.0 / rate
    deadline = time.perf_counter()
    for i in range(1, steps + 1):
        s = i / steps
        backend.set_joint_targets(
            {name: begin[name] + (value - begin[name]) * s for name, value in targets.items()}
        )
        deadline += dt
        slack = deadline - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            deadline = time.perf_counter()
    return dict(targets)
