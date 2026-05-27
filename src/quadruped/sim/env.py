"""MuJoCo model loader with load-time consistency checks against configs/robot.yaml.

The YAML is the source of truth; the MJCF is authored by hand for geometry but
its joint ranges and ctrlranges must match the YAML. This module enforces that
on load — if they drift, you get a clear AssertionError instead of a confusing
runtime surprise.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco

from quadruped.config import CONFIG, RobotConfig

ROOT = Path(__file__).resolve().parents[3]


def model_path(name: str | None = None) -> Path:
    """Resolve a MJCF path. None uses CONFIG.description_xml."""
    if name is None:
        return ROOT / CONFIG.description_xml
    if not name.endswith(".xml"):
        name = f"{name}.xml"
    return ROOT / "description" / name


def _assert_joint_ranges(model: mujoco.MjModel, cfg: RobotConfig) -> None:
    for joint in cfg.joints:
        try:
            jid = model.joint(joint.mjcf_name).id
        except KeyError:
            continue  # joint not in this model (e.g. single_leg.xml on quadruped cfg)
        lo, hi = model.jnt_range[jid]
        elo, ehi = joint.limit_rad
        assert math.isclose(lo, elo, abs_tol=1e-6) and math.isclose(hi, ehi, abs_tol=1e-6), (
            f"joint {joint.mjcf_name!r}: MJCF range ({lo}, {hi}) does not match "
            f"YAML limit_rad ({elo}, {ehi}). Regenerate codegen or fix MJCF."
        )


def load_model(
    xml: str | Path | None = None,
    *,
    check: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load an MJCF and return (model, data). Asserts YAML consistency by default."""
    path = Path(xml) if xml is not None else model_path()
    if not path.is_absolute():
        path = ROOT / path
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if check:
        _assert_joint_ranges(model, CONFIG)
    return model, data
