"""Interactive joint-angle jogger. Type three radians per line."""
from __future__ import annotations

import argparse
import sys

from quadruped.cli._backends import add_backend_args, build_backend
from quadruped.config import CONFIG


def parse_three_floats(line: str) -> tuple[float, float, float] | None:
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser, default="hw")
    args = parser.parse_args()

    backend = build_backend(args)
    with backend:
        print(f"Connected ({args.backend}). Type three radians ({', '.join(CONFIG.joint_names)}). Ctrl-D to exit.")
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
            q_dict = dict(zip(CONFIG.joint_names, angles))
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
