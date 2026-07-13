# AUTO-GENERATED from configs/robot.yaml by scripts/codegen.py.
# Do not edit by hand — re-run codegen after editing the YAML.
"""Static robot configuration generated from configs/robot.yaml.

This module is self-contained — it does not read the YAML at import time, so
runtime has no pyyaml dependency. Regenerate with `python scripts/codegen.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Joint:
    name: str            # e.g. "shoulder_FL"  — canonical key used everywhere
    leg: str             # "FL" | "FR" | "BL" | "BR"
    joint: str           # "shoulder" | "wing" | "knee"
    motor_id: int
    limit_rad: Tuple[float, float]
    offset_deg: float
    direction: int

    @property
    def mjcf_name(self) -> str:
        """MJCF joint name follows `<joint>_joint_<leg>` convention."""
        return f"{self.joint}_joint_{self.leg}"


@dataclass(frozen=True)
class Serial:
    baud_host: int
    baud_dxl: int


@dataclass(frozen=True)
class Actuator:
    torque_constant_nm_per_a: float


@dataclass(frozen=True)
class LinksMM:
    shoulder_to_wing: Tuple[float, float, float]
    wing_to_knee: Tuple[float, float, float]
    knee_to_foot: Tuple[float, float, float]


@dataclass(frozen=True)
class MjcfNames:
    foot_site_name: str
    target_site_name: str
    joint_names: Tuple[str, ...]


@dataclass(frozen=True)
class RobotConfig:
    robot: str
    description_xml: str
    serial: Serial
    actuator: Actuator
    legs: Tuple[str, ...]
    joints: Tuple[Joint, ...]
    links_mm: LinksMM
    foot_site_offset_mm: Tuple[float, float, float]
    target_site_offset_mm: Tuple[float, float, float]
    mjcf: MjcfNames

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"unknown joint: {name!r}")

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    def joints_for_leg(self, leg: str) -> Tuple[Joint, ...]:
        """All joints on `leg`, in canonical YAML order (shoulder, wing, knee)."""
        legs = {j.leg for j in self.joints}
        if leg not in legs:
            raise KeyError(f"unknown leg: {leg!r} (known: {sorted(legs)})")
        return tuple(j for j in self.joints if j.leg == leg)

    def leg_joint_names(self, leg: str) -> Tuple[str, ...]:
        return tuple(j.name for j in self.joints_for_leg(leg))


CONFIG: RobotConfig = RobotConfig(
    robot='quadruped',
    description_xml='description/quadruped.xml',
    serial=Serial(baud_host=57600, baud_dxl=115200),
    actuator=Actuator(torque_constant_nm_per_a=1.077),
    legs=(
'FL',
'FR',
'BL',
'BR',
),
    joints=(
Joint(name='shoulder_FL', leg='FL', joint='shoulder', motor_id=1, limit_rad=(-1.57079632679, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='wing_FL', leg='FL', joint='wing', motor_id=2, limit_rad=(-0.87266462600, 0.87266462600), offset_deg=180.000000, direction=1),
Joint(name='knee_FL', leg='FL', joint='knee', motor_id=3, limit_rad=(-2.00712863979, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='shoulder_FR', leg='FR', joint='shoulder', motor_id=4, limit_rad=(-1.57079632679, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='wing_FR', leg='FR', joint='wing', motor_id=5, limit_rad=(-0.87266462600, 0.87266462600), offset_deg=180.000000, direction=1),
Joint(name='knee_FR', leg='FR', joint='knee', motor_id=6, limit_rad=(-2.00712863979, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='shoulder_BL', leg='BL', joint='shoulder', motor_id=7, limit_rad=(-1.57079632679, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='wing_BL', leg='BL', joint='wing', motor_id=8, limit_rad=(-0.87266462600, 0.87266462600), offset_deg=180.000000, direction=1),
Joint(name='knee_BL', leg='BL', joint='knee', motor_id=9, limit_rad=(-2.00712863979, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='shoulder_BR', leg='BR', joint='shoulder', motor_id=10, limit_rad=(-1.57079632679, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='wing_BR', leg='BR', joint='wing', motor_id=11, limit_rad=(-0.87266462600, 0.87266462600), offset_deg=180.000000, direction=1),
Joint(name='knee_BR', leg='BR', joint='knee', motor_id=12, limit_rad=(-2.00712863979, 1.57079632679), offset_deg=180.000000, direction=1),
),
    links_mm=LinksMM(
        shoulder_to_wing=(-20.25, -23.25, 21.4),
        wing_to_knee=(0.0, -107.5, -23.25),
        knee_to_foot=(21.0, 0.0, -110.0),
    ),
    foot_site_offset_mm=(21, -12, 0),
    target_site_offset_mm=(0, -133, -80),
    mjcf=MjcfNames(
        foot_site_name='foot_site',
        target_site_name='target_site',
        joint_names=(
'shoulder_joint_FL',
'wing_joint_FL',
'knee_joint_FL',
'shoulder_joint_FR',
'wing_joint_FR',
'knee_joint_FR',
'shoulder_joint_BL',
'wing_joint_BL',
'knee_joint_BL',
'shoulder_joint_BR',
'wing_joint_BR',
'knee_joint_BR',
),
    ),
)
