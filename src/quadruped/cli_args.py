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
             "hw = Arduino/SoftwareSerial; mirror = sim + hw together.",
    )
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port for hw/dxl/mirror backends (or SERIAL_PORT env var).")
    parser.add_argument("--baud", "-b", type=int, default=None,
                        help="Override host baud (defaults to configs/robot.yaml).")
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
    )


def parse_three_floats(line: str) -> Optional[Tuple[float, float, float]]:
    """Parse `0.5 -0.3 0.2` or `0.5, -0.3, 0.2`; None if it isn't three numbers."""
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None
