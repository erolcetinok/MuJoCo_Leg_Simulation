"""Interactive joint-angle jogger. Type three radians per line."""
from __future__ import annotations

import argparse
import sys

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.cli_args import add_backend_args, build_backend, parse_three_floats
from quadruped.config import CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=CONFIG.legs, default="FL",
                        help="Leg to jog (default FL). The three angles map to that "
                             "leg's shoulder, wing, knee; other legs hold their cached pose.")
    add_backend_args(parser, default="hw")
    args = parser.parse_args()

    names = CONFIG.leg_joint_names(args.leg)
    backend = build_backend(args)
    with backend:
        print(f"Connected ({args.backend}). Jogging {args.leg}: type three radians "
              f"({', '.join(names)}). Ctrl-D to exit.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            angles = parse_three_floats(line)
            if angles is None:
                print("  ? need three numbers, e.g.  0.5 -0.3 0.2", file=sys.stderr)
                continue
            q_dict = dict(zip(names, angles))
            try:
                backend.set_joint_targets(q_dict)
            except Exception as e:  # noqa: BLE001 — surface anything serial throws
                print(f"  ! {e}", file=sys.stderr)
                break
            for b in getattr(backend, "backends", [backend]):
                reply = getattr(b, "last_reply", "")
                if reply:
                    print(f"  {reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
