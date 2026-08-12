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
