"""Teleop command shaping — keystrokes in, smooth body commands out.

A terminal reports key *repeat* but never key *release*, so raw keystrokes can't
drive the robot directly. Every axis therefore carries two numbers:

    setpoint  where the operator has asked the axis to go (moved by STEP per tap)
    actual    where the command actually is (chases setpoint under a SLEW limit)

`gait.py` warns that an abrupt change in body velocity puts a discontinuity in
the middle of a step; slew limiting is the direct mitigation, and it is also
what keeps this usable on real servos later.

Velocity axes **decay to zero** once no key has arrived for `decay_after`, so
letting go coasts the robot to a stop — the closest thing to a deadman a
terminal offers. Pose axes hold, because a commanded attitude should persist.

`apply_key` is a pure function over this state, so the whole key map is testable
without a terminal; `scripts/teleop.py` only does I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from quadruped.control.body import (
    MAX_HEIGHT_MM,
    MAX_PITCH_RAD,
    MAX_ROLL_RAD,
    MAX_YAW_RAD,
    MIN_HEIGHT_MM,
    BodyPose,
)

# Axes whose setpoint decays to zero when the operator stops pressing keys.
VELOCITY_AXES = ("vx", "vy", "yaw_rate")
POSE_AXES = ("roll", "pitch", "body_yaw", "height")


@dataclass
class ShapedAxis:
    """One command axis: a stepped setpoint and a slew-limited actual value."""

    step: float
    slew: float
    lo: float
    hi: float
    unit: str = ""
    setpoint: float = 0.0
    actual: float = 0.0

    def nudge(self, sign: float) -> None:
        self.setpoint = min(self.hi, max(self.lo, self.setpoint + sign * self.step))

    def scale_step(self, factor: float) -> None:
        """Multiply STEP, keeping it usefully bounded."""
        self.step = min(self.hi, max(1e-4, self.step * factor))

    def scale_slew(self, factor: float) -> None:
        self.slew = max(1e-4, self.slew * factor)

    def zero(self) -> None:
        self.setpoint = 0.0

    def hard_zero(self) -> None:
        """Setpoint and actual both to zero, bypassing the slew limit."""
        self.setpoint = 0.0
        self.actual = 0.0

    def update(self, dt: float) -> float:
        """Move `actual` toward `setpoint`, capped at `slew` per second."""
        max_delta = self.slew * dt
        error = self.setpoint - self.actual
        if abs(error) <= max_delta:
            self.actual = self.setpoint
        else:
            self.actual += math.copysign(max_delta, error)
        return self.actual


@dataclass
class TeleopCommand:
    """What the controller consumes for one tick."""

    body_velocity: tuple = (0.0, 0.0)
    yaw_rate: float = 0.0
    pose: BodyPose = field(default_factory=BodyPose)
    gait: str = "walk"


def default_axes(
    *,
    v_step: float = 10.0,
    v_slew: float = 100.0,
    v_max: float = 100.0,
    yaw_step: float = 0.1,
    yaw_slew: float = 0.5,
    yaw_max: float = 1.0,
) -> dict:
    """Axis table with sensible starting STEP/SLEW values.

    Velocities in mm/s and mm/s²; rates in rad/s and rad/s². Pose angles are
    stored in radians but stepped in whole degrees, which is what reads well on
    a keyboard.
    """
    deg = math.radians
    return {
        "vx": ShapedAxis(v_step, v_slew, -v_max, v_max, "mm/s"),
        "vy": ShapedAxis(v_step, v_slew, -v_max, v_max, "mm/s"),
        "yaw_rate": ShapedAxis(yaw_step, yaw_slew, -yaw_max, yaw_max, "rad/s"),
        "roll": ShapedAxis(deg(1.0), deg(30.0), -MAX_ROLL_RAD, MAX_ROLL_RAD, "rad"),
        "pitch": ShapedAxis(deg(1.0), deg(30.0), -MAX_PITCH_RAD, MAX_PITCH_RAD, "rad"),
        "body_yaw": ShapedAxis(deg(2.0), deg(45.0), -MAX_YAW_RAD, MAX_YAW_RAD, "rad"),
        "height": ShapedAxis(3.0, 60.0, MIN_HEIGHT_MM, MAX_HEIGHT_MM, "mm"),
    }


class CommandShaper:
    """Holds every axis, the active gait, and the idle-decay timer."""

    def __init__(self, axes: Optional[dict] = None, *, decay_after: float = 0.3,
                 gait: str = "walk"):
        self.axes = axes if axes is not None else default_axes()
        self.decay_after = decay_after
        self.gait = gait
        self.estopped = False
        self._idle = 0.0

    # --- input ---

    def key(self, k: str) -> bool:
        """Apply one keystroke. Returns False if the key means 'quit'."""
        return apply_key(self, k)

    def note_input(self) -> None:
        self._idle = 0.0

    # --- tick ---

    def update(self, dt: float) -> TeleopCommand:
        """Advance shaping by dt and return the command for this tick."""
        self._idle += dt
        if self._idle >= self.decay_after:
            for name in VELOCITY_AXES:
                self.axes[name].zero()

        for axis in self.axes.values():
            axis.update(dt)

        a = self.axes
        return TeleopCommand(
            body_velocity=(a["vx"].actual, a["vy"].actual),
            yaw_rate=a["yaw_rate"].actual,
            pose=BodyPose(
                roll=a["roll"].actual,
                pitch=a["pitch"].actual,
                yaw=a["body_yaw"].actual,
                z=a["height"].actual,
            ).clamped(),
            gait=self.gait,
        )

    # --- bulk actions ---

    def zero_velocity(self) -> None:
        for name in VELOCITY_AXES:
            self.axes[name].zero()

    def zero_all(self) -> None:
        for axis in self.axes.values():
            axis.zero()

    def estop(self) -> None:
        """Drop every command to zero immediately, ignoring slew limits."""
        for axis in self.axes.values():
            axis.hard_zero()
        self.estopped = True


# Key -> (axis, sign). Movement keys only; the rest are handled explicitly.
_NUDGES = {
    "w": ("vx", +1.0), "s": ("vx", -1.0),
    "a": ("vy", +1.0), "d": ("vy", -1.0),      # +y is body-left
    "q": ("yaw_rate", +1.0), "e": ("yaw_rate", -1.0),
    # Up arrow raises the nose. BodyPose.pitch is right-handed about +y, where
    # positive is nose DOWN, so the intuitive key maps to the negative direction.
    "UP": ("pitch", -1.0), "DOWN": ("pitch", +1.0),
    "LEFT": ("roll", -1.0), "RIGHT": ("roll", +1.0),
    ",": ("body_yaw", +1.0), ".": ("body_yaw", -1.0),
    "r": ("height", +1.0), "f": ("height", -1.0),
}

_STEP_SCALE = 1.25
_SLEW_SCALE = 1.25


def apply_key(shaper: CommandShaper, k: str) -> bool:
    """Apply a keystroke to `shaper`. Returns False iff the key means 'quit'.

    Pure with respect to I/O: `k` is already-decoded — a single character, or
    one of "UP"/"DOWN"/"LEFT"/"RIGHT" for the arrow keys.
    """
    if k in ("\x03", "\x1b", "ESC"):        # Ctrl-C, ESC
        return False

    shaper.note_input()

    if k in _NUDGES:
        # Any movement key clears an e-stop: the operator is driving again.
        shaper.estopped = False
        axis, sign = _NUDGES[k]
        shaper.axes[axis].nudge(sign)
        return True

    if k == " ":
        shaper.zero_all()
    elif k in ("x", "X"):
        shaper.estop()
    elif k == "1":
        shaper.gait = "walk"
    elif k == "2":
        shaper.gait = "trot"
    elif k == "[":
        for name in ("vx", "vy"):
            shaper.axes[name].scale_step(1 / _STEP_SCALE)
    elif k == "]":
        for name in ("vx", "vy"):
            shaper.axes[name].scale_step(_STEP_SCALE)
    elif k == ";":
        for name in ("vx", "vy"):
            shaper.axes[name].scale_slew(1 / _SLEW_SCALE)
    elif k == "'":
        for name in ("vx", "vy"):
            shaper.axes[name].scale_slew(_SLEW_SCALE)
    elif k == "<":
        shaper.axes["yaw_rate"].scale_step(1 / _STEP_SCALE)
    elif k == ">":
        shaper.axes["yaw_rate"].scale_step(_STEP_SCALE)
    elif k == ":":
        shaper.axes["yaw_rate"].scale_slew(1 / _SLEW_SCALE)
    elif k == '"':
        shaper.axes["yaw_rate"].scale_slew(_SLEW_SCALE)
    return True
