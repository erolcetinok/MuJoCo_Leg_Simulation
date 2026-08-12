from typing import Optional

from quadruped.backends.base import RobotBackend
from quadruped.backends.mujoco_backend import MujocoBackend
from quadruped.backends.arduino_backend import ArduinoBackend
from quadruped.backends.mirror_backend import MirrorBackend
from quadruped.backends.dynamixel_backend import DynamixelBackend

BACKEND_CHOICES = ("sim", "dxl", "hw", "mirror")


def make_backend(
    kind: str,
    *,
    port: Optional[str] = None,
    baud: Optional[int] = None,
    xml: Optional[str] = None,
    viewer: bool = False,
) -> RobotBackend:
    """Build an unconnected backend by name. Caller handles connect/disconnect.

    The single place a `--backend` string becomes an object, so the commands and
    the GUI can never drift apart on what `mirror` means or which backends exist.
    """
    if kind == "sim":
        return MujocoBackend(xml=xml, use_viewer=viewer)
    if kind == "dxl":
        return DynamixelBackend(port=port, baud=baud)
    if kind == "hw":
        return ArduinoBackend(port=port, baud=baud)
    if kind == "mirror":
        sim = MujocoBackend(xml=xml, use_viewer=viewer)
        hw = ArduinoBackend(port=port, baud=baud)
        return MirrorBackend([sim, hw], truth_source=1)
    raise ValueError(f"unknown backend: {kind!r} (known: {', '.join(BACKEND_CHOICES)})")


__all__ = [
    "RobotBackend",
    "MujocoBackend",
    "ArduinoBackend",
    "MirrorBackend",
    "DynamixelBackend",
    "BACKEND_CHOICES",
    "make_backend",
]
