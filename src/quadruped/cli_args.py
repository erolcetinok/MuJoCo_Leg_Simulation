"""Argparse glue shared by the commands in scripts/.

Lives in the package, not in scripts/, for two reasons: scripts must never
import each other (a sibling import only resolves when the script is run
directly, which quietly breaks under `python -m` or any other entry), and
anything with behaviour worth testing belongs where tests can reach it.

Backend construction itself is `quadruped.backends.make_backend` — the GUI uses
that directly, without going through argparse.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

from quadruped.backends import BACKEND_CHOICES, RobotBackend, make_backend


def add_backend_args(parser: argparse.ArgumentParser, *, default: str = "sim") -> None:
    parser.add_argument(
        "--backend",
        choices=list(BACKEND_CHOICES),
        default=default,
        help="Which backend to drive (default: %(default)s). "
             "sim = MuJoCo (kinematic); dxl = U2D2 + DYNAMIXEL SDK; "
             "mirror = sim + dxl together.",
    )
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port for dxl/mirror backends (or SERIAL_PORT env var).")
    parser.add_argument("--baud", "-b", type=int, default=None,
                        help="Override the DXL bus baud (defaults to configs/robot.yaml).")
    parser.add_argument("--profile-velocity", type=int, default=None,
                        help="Servo velocity cap for dxl/mirror (0 = uncapped, which the "
                             "streaming gait needs; try 30 for a first cautious power-on).")
    parser.add_argument("--xml", default=None,
                        help="Override MJCF path (defaults to configs/robot.yaml).")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch the MuJoCo passive viewer when running sim/mirror.")


def build_backend(args: argparse.Namespace) -> RobotBackend:
    """Return an unconnected backend from parsed args. Caller connects."""
    return make_backend(
        args.backend,
        port=args.port or os.environ.get("SERIAL_PORT"),
        baud=args.baud,
        xml=args.xml,
        viewer=args.viewer,
        profile_velocity=getattr(args, "profile_velocity", None),
    )


def install_signal_handlers() -> None:
    """Turn SIGTERM/SIGHUP into KeyboardInterrupt so `with backend` still unwinds.

    Without this, a systemd stop or a dropped SSH session kills the process
    outright: disconnect() never runs, so the servos stay torqued holding their
    last goal and a raw-mode terminal is never restored.
    """
    import signal

    def _raise(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _raise)


def parse_three_floats(line: str) -> Optional[Tuple[float, float, float]]:
    """Parse `0.5 -0.3 0.2` or `0.5, -0.3, 0.2`; None if it isn't three numbers."""
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None
