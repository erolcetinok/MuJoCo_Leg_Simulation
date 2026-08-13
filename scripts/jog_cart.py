"""Interactive Cartesian foot jogger. Type three mm per line (x y z).

Each line is solved with DLS IK seeded from the previous commanded pose — so the
leg tracks the same branch smoothly between commands instead of jumping — then
dispatched to the chosen backend. Defaults to the U2D2 (dxl) backend and prints
the estimated Cartesian foot force each line (LORIS-style, from the servo
current). On backends without current sensing it prints `n/a` and still jogs.
"""
import argparse
import sys

import numpy as np

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.cli_args import add_backend_args, build_backend, parse_three_floats

from quadruped.config import CONFIG
from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.jacobian import dls_ik


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=CONFIG.legs, default="FL",
                        help="Leg to jog (default FL). x y z are in that leg's "
                             "local frame (mm); other legs hold their cached pose.")
    add_backend_args(parser, default="dxl")
    args = parser.parse_args()

    names = CONFIG.leg_joint_names(args.leg)
    backend = build_backend(args)
    q_prev: tuple[float, float, float] | None = None
    with backend:
        print(f"Connected ({args.backend}). Jogging {args.leg} foot: type three "
              f"mm (x y z) in the leg frame. Ctrl-D to exit.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            target = parse_three_floats(line)
            if target is None:
                print("  ? need three numbers, e.g.  0 -133 -80", file=sys.stderr)
                continue

            q = dls_ik(target, q0=q_prev)
            residual = float(np.linalg.norm(foot_position(*q) - np.array(target)))
            if residual > 1.0:
                print(f"  ~ target likely out of reach (FK residual {residual:.1f} mm)",
                      file=sys.stderr)

            try:
                backend.set_joint_targets(dict(zip(names, q)))
            except Exception as e:  # noqa: BLE001 — surface anything serial throws
                print(f"  ! {e}", file=sys.stderr)
                break
            q_prev = q

            # One walk of the backend(s): print the estimated foot force from
            # the first current-sensing backend, plus any firmware echo (like
            # scripts/jog.py). build_backend's mirror is sim+Arduino, so only the
            # plain dxl backend supplies a foot force today.
            force_shown = False
            for b in getattr(backend, "backends", [backend]):
                if not force_shown and hasattr(b, "foot_force"):
                    try:
                        print(f"  foot force (N): {np.round(b.foot_force(args.leg), 3)}")
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! foot_force: {e}", file=sys.stderr)
                    force_shown = True
                reply = getattr(b, "last_reply", "")
                if reply:
                    print(f"  {reply}")
            if not force_shown:
                print("  foot force: n/a (needs dxl backend)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
