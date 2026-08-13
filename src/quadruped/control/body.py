"""Body pose and body↔leg frame conversion.

The gait scheduler produces foot targets in the **body frame**, assuming a level
body at nominal height. Commanding an attitude (pitch the nose down, roll, spin
the body in place) or a ride height means moving the body relative to the feet
— which, expressed in the body frame, means moving every foot target by the
*inverse* of that transform. `BodyPose.apply` is that inverse.

Pure geometry: no MuJoCo, no config. `leg_poses()` in `quadruped.sim.env` reads
the per-leg shoulder positions this module consumes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# v1 command envelope, measured rather than guessed — see
# tests/test_locomotion.py::test_joint_limits_hold_across_the_whole_command_envelope,
# which sweeps the corners of this box against configs/robot.yaml joint limits.
#
# It is tighter than it looks like it should be, for a real geometric reason:
# the WING joint has only ±50° of travel and it is the binding constraint on
# every axis. At 100 mm/s the gait alone already consumes ~65% of that travel,
# and pitch and roll ADD on the diagonal legs — 6° of each costs about as much
# as 8.5° of either. Hence the elliptical tilt limit below rather than an
# independent cap per axis, and hence no crouching: the nominal stance is
# already low (~70% leg extension), so negative height offsets eat wing travel
# immediately.
#
# Widening any of this wants the real fix — workspace-aware clamping of the foot
# targets (`is_reachable`) instead of a fixed command box.
MAX_TILT_RAD = math.radians(6.0)     # magnitude of combined pitch+roll
MAX_ROLL_RAD = MAX_TILT_RAD
MAX_PITCH_RAD = MAX_TILT_RAD
MAX_YAW_RAD = math.radians(10.0)
MAX_SHIFT_MM = 15.0          # |x| and |y| body translation
MIN_HEIGHT_MM = 0.0          # z offset from nominal stance height (no crouch)
MAX_HEIGHT_MM = 25.0


def rotz_xy(theta: float) -> np.ndarray:
    """2×2 rotation about z, for the xy part of a body-frame vector."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class BodyPose:
    """Commanded body attitude and translation, relative to nominal stance.

    Angles in radians, translations in mm, body frame (x forward, y left, z up).
    `z` is a ride-height offset: **positive z raises the body**, which pushes the
    feet further down in body frame.

    Sign conventions are right-handed about the body axes (ROS REP-103), which
    means **positive pitch is nose DOWN** and positive roll drops the right side.
    That trips people up, so the UI layer (`control.command`, the teleop HUD,
    `gait_demo --pitch`) presents pitch as nose-up-positive and negates once on
    the way in. Keep this class in the standard convention.
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def is_identity(self) -> bool:
        return not any((self.roll, self.pitch, self.yaw, self.x, self.y, self.z))

    def clamped(self) -> "BodyPose":
        """Return this pose inside the command envelope.

        Pitch and roll are limited *together* by their combined magnitude, not
        independently: they add on the diagonal legs, so a box limit that each
        axis passes can still exceed the wing joint's travel when both are near
        their edge. Scaling the tilt vector keeps the commanded direction and
        only shortens it.
        """
        pitch, roll = self.pitch, self.roll
        tilt = math.hypot(pitch, roll)
        if tilt > MAX_TILT_RAD:
            scale = MAX_TILT_RAD / tilt
            pitch *= scale
            roll *= scale
        return BodyPose(
            roll=_clamp(roll, -MAX_ROLL_RAD, MAX_ROLL_RAD),
            pitch=_clamp(pitch, -MAX_PITCH_RAD, MAX_PITCH_RAD),
            yaw=_clamp(self.yaw, -MAX_YAW_RAD, MAX_YAW_RAD),
            x=_clamp(self.x, -MAX_SHIFT_MM, MAX_SHIFT_MM),
            y=_clamp(self.y, -MAX_SHIFT_MM, MAX_SHIFT_MM),
            z=_clamp(self.z, MIN_HEIGHT_MM, MAX_HEIGHT_MM),
        )

    def apply(self, p_body: np.ndarray) -> np.ndarray:
        """Move a body-frame foot target to account for this body pose.

        Inverse transform: the feet stay where they are in the world, so in the
        body's own frame they move opposite to the body. Identity pose returns
        the point unchanged — that is what keeps existing gait behaviour exact.
        """
        if self.is_identity:
            return p_body
        t = np.array([self.x, self.y, self.z], dtype=float)
        return _rot_zyx(self.roll, self.pitch, self.yaw).T @ (np.asarray(p_body, dtype=float) - t)


def body_to_leg(p_body: np.ndarray, shoulder_pos: np.ndarray, yaw: float) -> np.ndarray:
    """Body-frame point → leg-local frame (the frame fk/ik work in).

    `shoulder_pos` is the leg's shoulder *rotation axis* in body frame and `yaw`
    its mounting angle — both from `quadruped.sim.env.leg_poses`.
    """
    delta = np.asarray(p_body, dtype=float) - np.asarray(shoulder_pos, dtype=float)
    xy = rotz_xy(-yaw) @ delta[:2]
    return np.array([xy[0], xy[1], delta[2]])


def leg_to_body(p_local: np.ndarray, shoulder_pos: np.ndarray, yaw: float) -> np.ndarray:
    """Inverse of `body_to_leg` — used to place per-leg home positions."""
    p_local = np.asarray(p_local, dtype=float)
    xy = rotz_xy(yaw) @ p_local[:2]
    return np.asarray(shoulder_pos, dtype=float) + np.array([xy[0], xy[1], p_local[2]])
