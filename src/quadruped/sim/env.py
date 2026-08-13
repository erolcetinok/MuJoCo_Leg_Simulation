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
import numpy as np

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


def leg_poses(model: mujoco.MjModel) -> dict[str, tuple[np.ndarray, float]]:
    """Per-leg (shoulder_axis_in_body_frame, mounting_yaw) read from the MJCF.

    The MJCF is the geometric source of truth for leg layout, so this is read
    once at startup even on hardware-only runs.

    The frame FK/IK rotate about is the shoulder *rotation axis*, not the
    shoulder body origin: the joint sits at a local offset inside the body
    (`shoulder_joint` pos in the MJCF). Recover the axis in body frame as
    `body.pos + Rz(yaw)·jnt_pos` so it lands on the design corner (±75, ±75)
    regardless of that offset.
    """
    from quadruped.control.body import rotz_xy

    poses: dict[str, tuple[np.ndarray, float]] = {}
    for leg in CONFIG.legs:
        body = model.body(f"shoulder_{leg}")
        # All four shoulder bodies use a pure z-yaw quaternion, so
        # quat = (cos(θ/2), 0, 0, sin(θ/2)).
        w, _, _, z = body.quat
        yaw = 2.0 * float(math.atan2(z, w))
        jnt_pos = np.asarray(model.joint(f"shoulder_joint_{leg}").pos, dtype=float)
        xy = rotz_xy(yaw) @ jnt_pos[:2]
        axis = np.asarray(body.pos, dtype=float) + np.array([xy[0], xy[1], jnt_pos[2]])
        poses[leg] = (axis, yaw)
    return poses


MJPYTHON_HINT = (
    "This needs an interactive MuJoCo window, which on macOS must run under "
    "`mjpython` (it owns the main thread; plain `python` cannot).\n\n"
    "    mjpython {cmd}\n\n"
    "mjpython ships with the mujoco wheel — it is already at .venv/bin/mjpython.\n"
    "Only windowed commands need it; anything headless runs under plain python, "
    "and `python scripts/gui.py` works too because its default embedded viewer "
    "renders offscreen."
)


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    """`mujoco.viewer.launch_passive`, but with an actionable macOS error.

    MuJoCo's own message says mjpython is required without saying what to type,
    and it surfaces as a traceback — easy to hit repeatedly and still not have
    the command to hand.
    """
    import sys

    import mujoco.viewer as mj_viewer

    try:
        return mj_viewer.launch_passive(model, data)
    except RuntimeError as e:
        if "mjpython" not in str(e):
            raise
        argv = " ".join(sys.argv) if sys.argv and sys.argv[0] else "scripts/<command>.py"
        raise SystemExit(MJPYTHON_HINT.format(cmd=argv)) from None
