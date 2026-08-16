"""Send three joint angles (shoulder, wing, knee) to one leg."""
from __future__ import annotations

import argparse
import sys

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.cli_args import add_backend_args, build_backend
from quadruped.config import CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("q1", type=float, help="Shoulder angle (rad)")
    parser.add_argument("q2", type=float, help="Wing angle (rad)")
    parser.add_argument("q3", type=float, help="Knee angle (rad)")
    parser.add_argument("--leg", choices=CONFIG.legs, default="FL",
                        help="Leg to move (default FL). Other legs hold their cached pose.")
    add_backend_args(parser, default="dxl")
    args = parser.parse_args()

    names = CONFIG.leg_joint_names(args.leg)
    q_dict = dict(zip(names, [args.q1, args.q2, args.q3]))
    backend = build_backend(args)
    with backend:
        backend.set_joint_targets(q_dict)
        for b in getattr(backend, "backends", [backend]):
            reply = getattr(b, "last_reply", "")
            if reply:
                print(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
