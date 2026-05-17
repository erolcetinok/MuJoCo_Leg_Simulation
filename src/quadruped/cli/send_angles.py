"""Send three joint angles (shoulder, wing, knee) directly to a backend."""
from __future__ import annotations

import argparse
import sys

from quadruped.cli._backends import add_backend_args, build_backend
from quadruped.config import CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("q1", type=float, help="Shoulder angle (rad)")
    parser.add_argument("q2", type=float, help="Wing angle (rad)")
    parser.add_argument("q3", type=float, help="Knee angle (rad)")
    add_backend_args(parser, default="hw")
    args = parser.parse_args()

    q_dict = dict(zip(CONFIG.joint_names, [args.q1, args.q2, args.q3]))
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
