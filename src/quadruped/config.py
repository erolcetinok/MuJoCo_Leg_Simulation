# AUTO-GENERATED from configs/robot.yaml by scripts/codegen.py.
# Do not edit by hand — re-run codegen after editing the YAML.
"""Static robot configuration generated from configs/robot.yaml.

This module is self-contained — it does not read the YAML at import time, so
runtime has no pyyaml dependency. Regenerate with `python scripts/codegen.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Joint:
    name: str
    motor_id: int
    limit_rad: Tuple[float, float]
    offset_deg: float
    direction: int


@dataclass(frozen=True)
class Serial:
    baud_host: int
    baud_dxl: int


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


CONFIG: RobotConfig = RobotConfig(
    robot='single_leg',
    description_xml='description/single_leg.xml',
    serial=Serial(baud_host=57600, baud_dxl=115200),
    joints=(
Joint(name='shoulder', motor_id=1, limit_rad=(-1.57079632679, 1.57079632679), offset_deg=180.000000, direction=1),
Joint(name='wing', motor_id=2, limit_rad=(-0.87266462600, 0.87266462600), offset_deg=180.000000, direction=1),
Joint(name='knee', motor_id=3, limit_rad=(-2.00712863979, 1.57079632679), offset_deg=180.000000, direction=1),
),
    links_mm=LinksMM(
        shoulder_to_wing=(-20.25, -23.25, 21.4),
        wing_to_knee=(0.0, -107.5, -23.25),
        knee_to_foot=(21.0, 0.0, -110.0),
    ),
    foot_site_offset_mm=(21, -12, 0),
    target_site_offset_mm=(20, -175, -50),
    mjcf=MjcfNames(
        foot_site_name='foot_site',
        target_site_name='target_site',
        joint_names=(
'shoulder_joint',
'wing_joint',
'knee_joint',
),
    ),
)
