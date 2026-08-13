"""The full locomotion tick: body command in, twelve joint angles out.

This is the one place the pipeline lives, so the scripted demo and the
interactive teleop drive identical code:

    gait phase  ->  foot target (body frame)
                ->  body pose applied      (BodyPose.apply)
                ->  leg-local frame        (body_to_leg)
                ->  closed-form IK         (kinematics.ik.joint_angles)
                ->  {joint_name: radians}

Stateless apart from the gait phase held by the scheduler.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quadruped.config import CONFIG
from quadruped.control.body import BodyPose, body_to_leg, leg_to_body
from quadruped.control.gait import GaitScheduler, TROT_PRESET, WALK_PRESET
from quadruped.kinematics.ik import joint_angles

GAIT_PRESETS = {"walk": WALK_PRESET, "trot": TROT_PRESET}


def home_positions(leg_poses: dict) -> dict:
    """Per-leg resting foot position in body frame.

    The canonical rest pose is defined once in leg-local terms by
    `configs/robot.yaml: target_site_offset_mm`; each leg's mounting yaw rotates
    it into the body frame.
    """
    home_local = np.array(CONFIG.target_site_offset_mm, dtype=float)
    return {
        leg: leg_to_body(home_local, shoulder_pos, yaw)
        for leg, (shoulder_pos, yaw) in leg_poses.items()
    }


class LocomotionController:
    """Owns the gait scheduler and turns a body command into joint targets."""

    def __init__(self, leg_poses: dict, gait: str = "walk"):
        if gait not in GAIT_PRESETS:
            raise ValueError(f"unknown gait: {gait!r} (known: {', '.join(GAIT_PRESETS)})")
        self.leg_poses = leg_poses
        self.home = home_positions(leg_poses)
        self.gait = gait
        self.scheduler = GaitScheduler(**GAIT_PRESETS[gait], home_positions=self.home)

    def set_gait(self, gait: str) -> None:
        """Switch gait preset, preserving the current phase so it doesn't jump."""
        if gait == self.gait:
            return
        if gait not in GAIT_PRESETS:
            raise ValueError(f"unknown gait: {gait!r} (known: {', '.join(GAIT_PRESETS)})")
        phase = self.scheduler.phase
        self.gait = gait
        self.scheduler = GaitScheduler(**GAIT_PRESETS[gait], home_positions=self.home)
        self.scheduler.phase = phase

    def foot_targets(self, pose: Optional[BodyPose] = None) -> dict:
        """Current foot target per leg, in that leg's local frame."""
        out = {}
        for leg, (shoulder_pos, yaw) in self.leg_poses.items():
            p_body = self.scheduler.foot_position(leg)
            if pose is not None:
                p_body = pose.apply(p_body)
            out[leg] = body_to_leg(p_body, shoulder_pos, yaw)
        return out

    def step(
        self,
        dt: float,
        *,
        body_velocity: tuple = (0.0, 0.0),
        yaw_rate: float = 0.0,
        pose: Optional[BodyPose] = None,
    ) -> dict:
        """Advance one control tick and return {joint_name: angle_rad} for all 12."""
        self.scheduler.advance(dt, body_velocity, yaw_rate)
        targets: dict = {}
        for leg, p_local in self.foot_targets(pose).items():
            q_sh, q_wg, q_kn = joint_angles(p_local[0], p_local[1], p_local[2])
            targets[f"shoulder_{leg}"] = float(q_sh)
            targets[f"wing_{leg}"] = float(q_wg)
            targets[f"knee_{leg}"] = float(q_kn)
        return targets

    def stance_summary(self) -> str:
        """Compact per-leg stance/swing string for the teleop HUD."""
        return "  ".join(
            f"{leg}:{'sw' if self.scheduler.is_swing(leg) else 'st'}" for leg in CONFIG.legs
        )
