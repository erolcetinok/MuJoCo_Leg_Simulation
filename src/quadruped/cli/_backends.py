"""Shared CLI helper: parse --backend / --port and build a connected backend."""
from __future__ import annotations

import argparse
import os

from quadruped.backends import ArduinoBackend, MirrorBackend, MujocoBackend, RobotBackend


def add_backend_args(parser: argparse.ArgumentParser, *, default: str = "sim") -> None:
    parser.add_argument(
        "--backend",
        choices=["sim", "hw", "mirror"],
        default=default,
        help="Which backend to drive (default: %(default)s).",
    )
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port for hw/mirror backends (or SERIAL_PORT env var).")
    parser.add_argument("--baud", "-b", type=int, default=None,
                        help="Override host baud (defaults to configs/robot.yaml).")
    parser.add_argument("--xml", default=None,
                        help="Override MJCF path (defaults to configs/robot.yaml).")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch the MuJoCo passive viewer when running sim/mirror.")


def build_backend(args: argparse.Namespace) -> RobotBackend:
    """Return an unconnected backend. Caller is responsible for connect/disconnect."""
    if args.backend == "sim":
        return MujocoBackend(xml=args.xml, use_viewer=args.viewer)
    if args.backend == "hw":
        port = args.port or os.environ.get("SERIAL_PORT")
        return ArduinoBackend(port=port, baud=args.baud)
    if args.backend == "mirror":
        port = args.port or os.environ.get("SERIAL_PORT")
        sim = MujocoBackend(xml=args.xml, use_viewer=args.viewer)
        hw = ArduinoBackend(port=port, baud=args.baud)
        return MirrorBackend([sim, hw], truth_source=1)
    raise ValueError(f"unknown backend: {args.backend!r}")
