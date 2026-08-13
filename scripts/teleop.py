"""Keyboard teleoperation — drive the quadruped around live.

Drive, turn, and command body attitude from the terminal while the gait runs.
Works with any backend, so the same command flies the sim today and the real
robot over SSH later (no display needed for the control loop itself).

  W/S  forward / back        Up/Down     pitch          1/2    walk / trot
  A/D  strafe left / right   Left/Right  roll           SPACE  zero commands
  Q/E  turn left / right     , / .       body yaw       X      e-stop
                             R/F         ride height    H      toggle help
  [ ]  velocity STEP -/+     ; '         velocity SLEW -/+
  < >  yaw STEP -/+          : "         yaw SLEW -/+   Ctrl-C / ESC  quit

Two knobs shape every axis: STEP is how far one tap moves the setpoint, SLEW is
how fast the command ramps toward it. Velocity decays to zero shortly after you
stop pressing keys, so releasing the keys stops the robot.

Examples:
    mjpython scripts/teleop.py --backend sim --viewer
    python   scripts/teleop.py --backend dxl --rate 33      # real robot, no window
"""
from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty

import mujoco

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.backends import MujocoBackend
from quadruped.cli_args import add_backend_args, build_backend
from quadruped.control.command import CommandShaper, default_axes
from quadruped.control.locomotion import LocomotionController
from quadruped.sim.env import leg_poses, model_path

_ARROWS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}


class RawKeys:
    """Unbuffered, non-blocking single keystrokes from stdin.

    Restores the terminal on the way out no matter how the loop exits — an
    un-restored tty is the classic way a crashed teleop leaves a shell unusable.
    Falls back to a no-op reader when stdin isn't a tty (piped input, CI).
    """

    def __init__(self) -> None:
        self.interactive = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.interactive else None
        self._saved = None

    def __enter__(self) -> "RawKeys":
        if self.interactive:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.interactive and self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self) -> list:
        """Every key pressed since the last call, decoded. Never blocks."""
        if not self.interactive:
            return []
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == "\x1b":
                # Escape alone, or the prefix of an arrow-key sequence (ESC [ A).
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    seq = sys.stdin.read(1)
                    if seq == "[" and select.select([sys.stdin], [], [], 0.01)[0]:
                        keys.append(_ARROWS.get(sys.stdin.read(1), ""))
                        continue
                    continue
                keys.append("ESC")
                continue
            keys.append(ch)
        return keys


def _any_viewer_closed(backend) -> bool:
    for b in getattr(backend, "backends", [backend]):
        if isinstance(b, MujocoBackend) and b._viewer is not None and not b.viewer_alive():
            return True
    return False


HELP_LINES = [
    "  W/S vx   A/D vy   Q/E turn      arrows: pitch/roll    , . body yaw   R/F height",
    "  1 walk  2 trot   SPACE zero   X e-stop   [ ] step   ; ' slew   < > : \" yaw   ESC quit",
]


def _hud(shaper: CommandShaper, controller: LocomotionController, show_help: bool) -> str:
    a = shaper.axes
    deg = math.degrees

    def row(label, axis, scale=1.0, fmt="{:+7.1f}"):
        act = fmt.format(axis.actual * scale)
        sp = fmt.format(axis.setpoint * scale)
        return (f"  {label:<10} {act} -> {sp} {axis.unit:<6}"
                f"step {axis.step * scale:6.2f}  slew {axis.slew * scale:7.2f}")

    lines = [
        row("vx", a["vx"]),
        row("vy", a["vy"]),
        row("yaw rate", a["yaw_rate"], fmt="{:+7.2f}"),
        "",
        # Displayed nose-up-positive; BodyPose.pitch itself is nose-down-positive.
        row("nose up°", a["pitch"], -deg(1.0)),
        row("roll°", a["roll"], deg(1.0)),
        row("body yaw°", a["body_yaw"], deg(1.0)),
        row("height", a["height"]),
        "",
        f"  gait {controller.gait:<6} contact {controller.stance_summary()}"
        + ("   [E-STOP]" if shaper.estopped else ""),
    ]
    if show_help:
        lines += [""] + HELP_LINES
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", choices=["walk", "trot"], default="walk",
                        help="Starting gait (switch live with 1/2).")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Control rate in Hz (default 50; use 33 on hardware)")
    parser.add_argument("--v-step", type=float, default=10.0,
                        help="Velocity change per keypress (mm/s, default 10)")
    parser.add_argument("--v-slew", type=float, default=100.0,
                        help="Velocity ramp rate (mm/s^2, default 100)")
    parser.add_argument("--v-max", type=float, default=100.0,
                        help="Velocity limit (mm/s, default 100)")
    parser.add_argument("--yaw-step", type=float, default=0.1,
                        help="Turn-rate change per keypress (rad/s, default 0.1)")
    parser.add_argument("--yaw-slew", type=float, default=0.5,
                        help="Turn-rate ramp (rad/s^2, default 0.5)")
    parser.add_argument("--yaw-max", type=float, default=1.0,
                        help="Turn-rate limit (rad/s, default 1.0)")
    parser.add_argument("--decay-after", type=float, default=0.3,
                        help="Seconds without a key before velocity decays to zero (default 0.3)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until quit)")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    dt = 1.0 / args.rate
    model = mujoco.MjModel.from_xml_path(str(model_path()))
    controller = LocomotionController(leg_poses(model), gait=args.gait)
    shaper = CommandShaper(
        default_axes(
            v_step=args.v_step, v_slew=args.v_slew, v_max=args.v_max,
            yaw_step=args.yaw_step, yaw_slew=args.yaw_slew, yaw_max=args.yaw_max,
        ),
        decay_after=args.decay_after,
        gait=args.gait,
    )

    backend = build_backend(args)
    show_help = True
    hud_lines = 0
    t0 = time.perf_counter()
    deadline = t0

    with backend, RawKeys() as keys:
        if not keys.interactive:
            print("stdin is not a tty — running with no keyboard input.", file=sys.stderr)
        try:
            while True:
                for k in keys.read():
                    if k == "h":
                        show_help = not show_help
                        shaper.note_input()
                        continue
                    if not shaper.key(k):
                        raise KeyboardInterrupt
                if shaper.gait != controller.gait:
                    controller.set_gait(shaper.gait)

                cmd = shaper.update(dt)
                backend.set_joint_targets(
                    controller.step(dt, body_velocity=cmd.body_velocity,
                                    yaw_rate=cmd.yaw_rate, pose=cmd.pose)
                )

                if keys.interactive:
                    frame = _hud(shaper, controller, show_help)
                    if hud_lines:
                        sys.stdout.write(f"\x1b[{hud_lines}A")
                    sys.stdout.write("\x1b[J" + frame + "\n")
                    sys.stdout.flush()
                    hud_lines = frame.count("\n") + 1

                if _any_viewer_closed(backend):
                    break
                if args.duration is not None and (time.perf_counter() - t0) >= args.duration:
                    break

                deadline += dt
                slack = deadline - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    deadline = time.perf_counter()
        except KeyboardInterrupt:
            pass

    print("\nteleop stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
