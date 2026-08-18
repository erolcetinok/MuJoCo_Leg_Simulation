from typing import Optional, Sequence

from quadruped.backends.base import RobotBackend
from quadruped.backends.mujoco_backend import MujocoBackend
from quadruped.backends.arduino_backend import ArduinoBackend
from quadruped.backends.mirror_backend import MirrorBackend
from quadruped.backends.dynamixel_backend import DynamixelBackend

# The Arduino/SoftwareSerial bridge is archived: the U2D2 is the one supported
# hardware path. ArduinoBackend stays importable for the old UNO rig, but it is
# no longer reachable from a --backend string.
BACKEND_CHOICES = ("sim", "dxl", "mirror")


def make_backend(
    kind: str,
    *,
    port: Optional[str] = None,
    baud: Optional[int] = None,
    xml: Optional[str] = None,
    viewer: bool = False,
    profile_velocity: Optional[int] = None,
    legs: Optional[Sequence[str]] = None,
) -> RobotBackend:
    """Build an unconnected backend by name. Caller handles connect/disconnect.

    The single place a `--backend` string becomes an object, so the commands and
    the GUI can never drift apart on what `mirror` means or which backends exist.
    """
    dxl_kwargs = {"port": port, "baud": baud}
    if profile_velocity is not None:
        dxl_kwargs["profile_velocity"] = profile_velocity
    if legs is not None:
        dxl_kwargs["legs"] = legs
    if kind == "sim":
        return MujocoBackend(xml=xml, use_viewer=viewer)
    if kind == "dxl":
        return DynamixelBackend(**dxl_kwargs)
    if kind == "mirror":
        sim = MujocoBackend(xml=xml, use_viewer=viewer)
        hw = DynamixelBackend(**dxl_kwargs)
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
