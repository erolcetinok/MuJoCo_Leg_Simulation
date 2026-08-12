"""Dear PyGui slider app (Landing 2 + embedded MuJoCo renderer)."""
from __future__ import annotations

import argparse
import os
import sys

from quadruped.backends import BACKEND_CHOICES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(BACKEND_CHOICES), default="sim")
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port for hw/dxl/mirror backends (or SERIAL_PORT env var).")
    parser.add_argument(
        "--viewer", choices=["embedded", "external", "none"], default="embedded",
        help="How to visualize the sim: 'embedded' renders MuJoCo into the GUI "
             "(default, recommended on macOS); 'external' launches mujoco's "
             "passive viewer in a separate window; 'none' is slider-only.",
    )
    parser.add_argument("--hz", type=int, default=100, help="UI loop rate (default 100).")
    args = parser.parse_args()

    try:
        from quadruped.gui.app import run
    except ImportError as e:
        sys.stderr.write(
            f"Dear PyGui not installed: {e}\nInstall with `pip install -e .[gui]`.\n"
        )
        return 2

    port = args.port or os.environ.get("SERIAL_PORT")
    return run(backend=args.backend, port=port, view_mode=args.viewer, hz=args.hz)


if __name__ == "__main__":
    sys.exit(main())
