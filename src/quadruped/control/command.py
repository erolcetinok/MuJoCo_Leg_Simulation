"""Teleop command shaping — keystrokes in, smooth body commands out.

A terminal reports key *repeat* but never key *release*, so raw keystrokes can't
drive the robot directly. Every axis therefore carries two numbers:

    setpoint  where the operator has asked the axis to go (moved one step per
              tap, or a tenth of a step with shift held)
    actual    where the command actually is (chases setpoint under a slew limit)

Steps are fixed, not tunable: a tap always moves the same amount, so the control
never changes under you mid-drive.

`gait.py` warns that an abrupt change in body velocity puts a discontinuity in
the middle of a step; slew limiting is the direct mitigation, and it is also
what keeps this usable on real servos later.

**Setpoints hold by default.** Set a speed and it stays there until you change
it — that's what makes this usable for driving around, and it doesn't depend on
your terminal's key-repeat behaviour.

Optionally, velocity axes can **decay to zero** once no key has arrived for
`decay_after` seconds (`decay_after=0` disables it, which is the default). That
turns the arrow keys into a hold-to-drive deadman: let go and the robot coasts
to a stop. It is the closest thing to a deadman a terminal offers, so it is
worth switching on for real hardware — a dropped SSH session with persistent
setpoints leaves the robot driving. Pose axes never decay either way.

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
    MIN_HEIGHT_MM,
    BodyPose,
)

# Axes whose setpoint decays to zero when the operator stops pressing keys.
VELOCITY_AXES = ("vx", "vy", "yaw_rate")
POSE_AXES = ("roll", "pitch", "height")

# Holding shift gives a tenth of a step. A terminal can't report modifiers, but
# shift+letter simply arrives as the uppercase letter, which is all we need.
FINE_SCALE = 0.1


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

    def nudge(self, sign: float, *, fine: bool = False) -> None:
        """Move the setpoint one step, or one tenth of a step when `fine`."""
        delta = sign * self.step * (FINE_SCALE if fine else 1.0)
        self.setpoint = min(self.hi, max(self.lo, self.setpoint + delta))

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
    """Axis table with fixed step sizes, one round number per axis.

    A tap is always the same amount and shift+tap is always a tenth of it, so
    the control never changes under you mid-drive. Steps are round in each
    axis's *natural* unit — 1 mm of ride height, 1° of attitude — rather than
    literally 1 everywhere: 1 rad/s of turn would be the entire range in one
    press, and 1 mm/s of drive would need a hundred presses to get moving.

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
        "height": ShapedAxis(1.0, 60.0, MIN_HEIGHT_MM, MAX_HEIGHT_MM, "mm"),
    }


class CommandShaper:
    """Holds every axis, the active gait, and the idle-decay timer."""

    def __init__(self, axes: Optional[dict] = None, *, decay_after: float = 0.0,
                 gait: str = "walk"):
        """decay_after <= 0 means setpoints hold indefinitely (the default)."""
        self.axes = axes if axes is not None else default_axes()
        self.decay_after = decay_after
        self.gait = gait
        self.estopped = False
        self.torque_kill = False
        self._idle = 0.0

    @property
    def deadman(self) -> bool:
        """True when velocity decays to zero on idle rather than holding."""
        return self.decay_after > 0.0

    def toggle_deadman(self, decay_after: float = 0.3) -> None:
        self.decay_after = 0.0 if self.deadman else decay_after

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
        if self.deadman and self._idle >= self.decay_after:
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
                z=a["height"].actual,
            ).clamped(),
            gait=self.gait,
        )

    # --- bulk actions ---

    def adjust_step(self, delta: float) -> None:
        """Change how far one tap moves the drive axes, in mm/s.

        Additive, not multiplicative: `]` always adds exactly 1 mm/s and `}`
        adds 0.1, so the number you're aiming for is the number of presses.
        Applies to vx/vy — the axes measured in mm — while attitude and ride
        height keep their round 1°/1 mm steps.
        """
        for name in ("vx", "vy"):
            axis = self.axes[name]
            axis.step = max(_MIN_STEP, axis.step + delta)

    def adjust_slew(self, delta: float) -> None:
        """Change how fast the drive axes ramp toward their setpoint, in mm/s²."""
        for name in ("vx", "vy"):
            axis = self.axes[name]
            axis.slew = max(_MIN_SLEW, axis.slew + delta)

    def zero_velocity(self) -> None:
        for name in VELOCITY_AXES:
            self.axes[name].zero()

    def zero_all(self) -> None:
        for axis in self.axes.values():
            axis.zero()

    def estop(self) -> None:
        """Drop every command to zero immediately, ignoring slew limits.

        The caller is expected to also stop advancing the gait and re-send the
        last joint targets — zeroing the body command alone leaves the robot
        marching in place, since a stopped gait still lifts each foot 30 mm.
        """
        for axis in self.axes.values():
            axis.hard_zero()
        self.estopped = True

    def request_torque_kill(self) -> bool:
        """Ask for torque to be cut. Only honoured from an e-stopped state.

        Releasing torque drops the robot on its belly, so it takes two
        deliberate keys (`x` then `z`) rather than one mistyped letter.
        Returns whether the request was accepted.
        """
        if not self.estopped:
            return False
        self.torque_kill = True
        return True


# Key -> (axis, sign). One key pair per degree of freedom, nothing else.
#
# Two letter clusters, no arrow keys: WASD drives, IJKL sets posture. Arrows
# would mean parsing multi-byte escape sequences out of the terminal, and a
# half-read sequence looks exactly like the ESC that quits — not worth it for
# keys a left hand can't reach anyway.
_NUDGES = {
    # Drive: the three things you steer with.
    "w": ("vx", +1.0), "s": ("vx", -1.0),
    "a": ("vy", +1.0), "d": ("vy", -1.0),      # +y is body-left
    "q": ("yaw_rate", +1.0), "e": ("yaw_rate", -1.0),
    # Posture: how the chassis is carried while it drives.
    # `i` raises the nose. BodyPose.pitch is right-handed about +y, where
    # positive is nose DOWN, so the intuitive key maps to the negative direction.
    "i": ("pitch", -1.0), "k": ("pitch", +1.0),
    "j": ("roll", -1.0), "l": ("roll", +1.0),
    "r": ("height", +1.0), "f": ("height", -1.0),
}

# Feel keys. Shift gives the fine variant of the same punctuation, so the rule
# is the same everywhere: shift means one tenth as much.
#   [ ]  sensitivity  ∓1 mm/s per tap      { }  ∓0.1
#   ; '  smoothing    ∓1 mm/s² of ramp     : "  ∓0.1
_STEP_KEYS = {"[": -1.0, "]": +1.0, "{": -FINE_SCALE, "}": +FINE_SCALE}
_SLEW_KEYS = {";": -1.0, "'": +1.0, ":": -FINE_SCALE, '"': +FINE_SCALE}

_MIN_STEP = 0.1
_MIN_SLEW = 1.0


def decode_keys(chunk: str) -> list:
    """Split a raw terminal read into logical keys.

    Every binding is a single character, so this is mostly pass-through. The one
    job it does is swallow CSI escape sequences — arrows, function keys, a mouse
    report — so that pressing an unbound arrow does *nothing* instead of being
    read as the leading ESC and quitting the program.

    A genuine lone ESC still means quit.
    """
    keys: list = []
    i = 0
    while i < len(chunk):
        c = chunk[i]
        if c != "\x1b":
            keys.append(c)
            i += 1
            continue
        if chunk[i + 1:i + 2] == "[":
            # CSI: skip the introducer plus parameters up to the final byte.
            i += 2
            while i < len(chunk) and not ("@" <= chunk[i] <= "~"):
                i += 1
            i += 1
            continue
        keys.append("ESC")
        i += 1
    return keys


def apply_key(shaper: CommandShaper, k: str) -> bool:
    """Apply a keystroke to `shaper`. Returns False iff the key means 'quit'.

    Pure with respect to I/O: `k` is already-decoded — a single character, or
    one of "UP"/"DOWN"/"LEFT"/"RIGHT" for the arrow keys.
    """
    if k in ("\x03", "\x1b", "ESC"):        # Ctrl-C, ESC
        return False

    shaper.note_input()

    # Shift+letter arrives as uppercase and means a fine (0.1x) step.
    lower = k.lower()
    if lower in _NUDGES:
        # Any movement key clears an e-stop: the operator is driving again.
        shaper.estopped = False
        axis, sign = _NUDGES[lower]
        shaper.axes[axis].nudge(sign, fine=k.isupper())
        return True

    if k == " ":
        shaper.zero_all()
    elif k in ("x", "X"):
        shaper.estop()
    elif k in ("z", "Z"):
        shaper.request_torque_kill()
    elif k in _STEP_KEYS:
        shaper.adjust_step(_STEP_KEYS[k])
    elif k in _SLEW_KEYS:
        shaper.adjust_slew(_SLEW_KEYS[k])
    elif k in ("g", "G"):
        shaper.toggle_deadman()
    elif k == "1":
        shaper.gait = "walk"
    elif k == "2":
        shaper.gait = "trot"
    return True
