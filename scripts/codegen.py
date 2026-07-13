#!/usr/bin/env python3
"""Generate firmware header + Python config from configs/robot.yaml.

This is the single source of truth pipeline: edit configs/robot.yaml, run this
script, and both `firmware/leg_controller/robot_config.h` and
`src/quadruped/config.py` are regenerated. The firmware sketch #includes the
header; the Python package imports the config module. Neither artifact should
be hand-edited.

Usage:
  python scripts/codegen.py           # regenerate (writes both files)
  python scripts/codegen.py --check   # exit 1 if either file is stale vs YAML

For pre-commit / CI: `python scripts/codegen.py --check`.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import yaml
from jinja2 import Environment

ENV = Environment(autoescape=False, keep_trailing_newline=True)
ENV.filters["repr"] = repr

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "configs" / "robot.yaml"
HEADER_PATH = ROOT / "firmware" / "leg_controller" / "robot_config.h"
CONFIG_PY_PATH = ROOT / "src" / "quadruped" / "config.py"

GENERATED_BANNER_H = (
    "// AUTO-GENERATED from configs/robot.yaml by scripts/codegen.py.\n"
    "// Do not edit by hand — re-run codegen after editing the YAML.\n"
)
GENERATED_BANNER_PY = (
    "# AUTO-GENERATED from configs/robot.yaml by scripts/codegen.py.\n"
    "# Do not edit by hand — re-run codegen after editing the YAML.\n"
)

HEADER_TMPL = ENV.from_string(
    """{{ banner }}#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#define HOST_BAUD {{ serial.baud_host }}
#define DXL_BAUD {{ serial.baud_dxl }}

#define N_JOINTS {{ joints|length }}
#define N_LEGS {{ legs|length }}

// Joint order is the canonical YAML order: legs FL, FR, BL, BR × joints
// shoulder, wing, knee. The host wire format and the syncWrite packet both
// follow this index order.
static const uint8_t IDS[N_JOINTS] = {
{%- for j in joints %}
  {{ j.motor_id }}{% if not loop.last %},{% endif %}  // [{{ loop.index0 }}] {{ j.name }}
{%- endfor %}
};

static const float OFFSETS_DEG[N_JOINTS] = {
{%- for j in joints %}
  {{ "%.6f"|format(j.offset_deg) }}f{% if not loop.last %},{% endif %}  // [{{ loop.index0 }}] {{ j.name }}
{%- endfor %}
};

static const float DIRS[N_JOINTS] = {
{%- for j in joints %}
  {{ "%.1f"|format(j.direction|float) }}f{% if not loop.last %},{% endif %}  // [{{ loop.index0 }}] {{ j.name }}
{%- endfor %}
};

static const float LIMITS[N_JOINTS][2] = {
{%- for j in joints %}
  { {{ "%.11f"|format(j.limit_rad[0]) }}f, {{ "%.11f"|format(j.limit_rad[1]) }}f }{% if not loop.last %},{% endif %}  // [{{ loop.index0 }}] {{ j.name }}
{%- endfor %}
};

#endif  // ROBOT_CONFIG_H
"""
)

CONFIG_PY_TMPL = ENV.from_string(
    '''{{ banner }}"""Static robot configuration generated from configs/robot.yaml.

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
    robot={{ robot|repr }},
    description_xml={{ description_xml|repr }},
    serial=Serial(baud_host={{ serial.baud_host }}, baud_dxl={{ serial.baud_dxl }}),
    actuator=Actuator(torque_constant_nm_per_a={{ actuator.torque_constant_nm_per_a }}),
    legs=(
{% for leg in legs -%}
        {{ leg|repr }},
{% endfor -%}
    ),
    joints=(
{% for j in joints -%}
        Joint(name={{ j.name|repr }}, leg={{ j.leg|repr }}, joint={{ j.joint|repr }}, motor_id={{ j.motor_id }}, limit_rad=({{ "%.11f"|format(j.limit_rad[0]) }}, {{ "%.11f"|format(j.limit_rad[1]) }}), offset_deg={{ "%.6f"|format(j.offset_deg) }}, direction={{ j.direction }}),
{% endfor -%}
    ),
    links_mm=LinksMM(
        shoulder_to_wing=({{ links_mm.shoulder_to_wing[0] }}, {{ links_mm.shoulder_to_wing[1] }}, {{ links_mm.shoulder_to_wing[2] }}),
        wing_to_knee=({{ links_mm.wing_to_knee[0] }}, {{ links_mm.wing_to_knee[1] }}, {{ links_mm.wing_to_knee[2] }}),
        knee_to_foot=({{ links_mm.knee_to_foot[0] }}, {{ links_mm.knee_to_foot[1] }}, {{ links_mm.knee_to_foot[2] }}),
    ),
    foot_site_offset_mm=({{ foot_site_offset_mm[0] }}, {{ foot_site_offset_mm[1] }}, {{ foot_site_offset_mm[2] }}),
    target_site_offset_mm=({{ target_site_offset_mm[0] }}, {{ target_site_offset_mm[1] }}, {{ target_site_offset_mm[2] }}),
    mjcf=MjcfNames(
        foot_site_name={{ mjcf.foot_site_name|repr }},
        target_site_name={{ mjcf.target_site_name|repr }},
        joint_names=(
{% for jn in mjcf.joint_names -%}
            {{ jn|repr }},
{% endfor -%}
        ),
    ),
)
'''
)


def render(yaml_data: dict) -> tuple[str, str]:
    header = HEADER_TMPL.render(banner=GENERATED_BANNER_H, **yaml_data)
    config_py = CONFIG_PY_TMPL.render(banner=GENERATED_BANNER_PY, **yaml_data)
    return header, config_py


def _diff(actual: str, expected: str, label: str) -> str:
    return "".join(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"{label} (on disk)",
            tofile=f"{label} (regenerated)",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if generated artifacts are stale vs YAML.")
    args = parser.parse_args()

    yaml_data = yaml.safe_load(YAML_PATH.read_text())
    header, config_py = render(yaml_data)

    if args.check:
        drift = False
        for path, content, label in (
            (HEADER_PATH, header, "robot_config.h"),
            (CONFIG_PY_PATH, config_py, "config.py"),
        ):
            on_disk = path.read_text() if path.exists() else ""
            if on_disk != content:
                diff = _diff(on_disk, content, label)
                sys.stderr.write(f"DRIFT in {path.relative_to(ROOT)}:\n{diff}\n")
                drift = True
        if drift:
            sys.stderr.write(
                "\nRun `python scripts/codegen.py` to regenerate.\n"
            )
            return 1
        print("ok: generated artifacts match configs/robot.yaml")
        return 0

    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(header)
    CONFIG_PY_PATH.write_text(config_py)
    print(f"wrote {HEADER_PATH.relative_to(ROOT)}")
    print(f"wrote {CONFIG_PY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
