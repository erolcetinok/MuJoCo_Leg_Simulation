"""Keyboard teleoperation — drive the quadruped around live.

Drive, turn, and command body attitude from the terminal while the gait runs.
Works with any backend, so the same command flies the sim today and the real
robot over SSH later (no display needed for the control loop itself).

  drive   fwd:w/s   strafe:a/d   turn:q/e
  posture pitch:i/k   roll:j/l   height:r/f
  mode    gait:1/2   stop:SPACE   estop:x   release:z   deadman:g   hide:h   quit:ESC
  feel    sens:[/]   smooth:;/'   fine:SHIFT

  Holding SHIFT on any key gives a tenth as much.

`x` freezes: the gait stops advancing and the last joint targets are re-sent, so
the robot stands still under torque. Any drive key resumes. `z` from that frozen
state cuts torque and the robot drops — two deliberate keys, because it drops.

Each tap moves a setpoint by one step and the command then ramps toward it under
a slew limit, so nothing jumps mid-stride. `sens` sets the step (+/- 1 mm/s per
press) and `smooth` sets the ramp (+/- 1 mm/s^2), both in round units, so the
value you want is just the number of presses. Attitude and ride height step by a
fixed 1 deg and 1 mm.

Setpoints **hold** where you leave them — set a speed and it stays until you
change it. `--decay-after 0.3` (or G) switches to deadman mode instead, where
velocity decays to zero shortly after you stop pressing keys. Worth turning on
for real hardware: with setpoints holding, a dropped SSH session leaves the
robot driving.

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
from quadruped.cli_args import add_backend_args, build_backend, install_signal_handlers
from quadruped.control.command import (
    CommandShaper,
    decode_keys,
    default_axes,
)
from quadruped.control.locomotion import LocomotionController
from quadruped.control.sequencer import ramp_to, stance_pose
from quadruped.sim.env import leg_poses, model_path

# How gently the servos move during the stand-up/settle ramps, and how fast
# during the gait. 0 = uncapped, which streaming needs since consecutive goals
# are already close together.
RAMP_PROFILE_VELOCITY = 30
GAIT_PROFILE_VELOCITY = 0
HEALTH_EVERY_TICKS = 50

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
        """Every key pressed since the last call, decoded. Never blocks.

        Reads raw bytes with os.read rather than sys.stdin.read: select() polls
        the file descriptor, but sys.stdin is buffered, so a three-byte arrow
        sequence lands in Python's buffer where select can't see it — the arrow
        then decodes as a bare ESC and quits the program.
        """
        if not self.interactive:
            return []
        buf = ""
        while select.select([sys.stdin], [], [], 0)[0]:
            chunk = os.read(self._fd, 1024).decode(errors="ignore")
            if not chunk:
                break
            buf += chunk
        if not buf:
            return []
        # A chunk that ends mid-escape would decode as a quit; give the rest of
        # the sequence a moment to land.
        if buf.endswith("\x1b") or buf.endswith("\x1b["):
            if select.select([sys.stdin], [], [], 0.02)[0]:
                buf += os.read(self._fd, 8).decode(errors="ignore")
        return decode_keys(buf)


def _any_viewer_closed(backend) -> bool:
    for b in getattr(backend, "backends", [backend]):
        if isinstance(b, MujocoBackend) and b._viewer is not None and not b.viewer_alive():
            return True
    return False


def _set_profile_velocity(backend, value: int) -> None:
    """Retune the servo velocity cap where the backend has one; ignore where not."""
    for b in getattr(backend, "backends", [backend]):
        setter = getattr(b, "set_profile_velocity", None)
        if setter is not None:
            setter(value)


def _wait_for_go(keys: "RawKeys", skip: bool) -> bool:
    """Block until a key. False means the operator quit instead of starting.

    A deliberate gap between "the legs are in stance" and "the gait is running":
    the stand-up is when you notice a leg is mirrored or a horn is off by 180.
    """
    if skip or not keys.interactive:
        return True
    print("in stance. press any key to start walking, ESC to quit.")
    while True:
        for k in keys.read():
            if k in ("\x03", "\x1b", "ESC"):
                return False
            return True
        time.sleep(0.02)


def _settle(backend, controller, args) -> None:
    """Return to the symmetric stance pose before disconnect() cuts torque.

    Not a sit-down — there is no crouch authority in the command envelope (see
    control/sequencer.py). This just makes the drop happen from a known,
    four-feet-planted pose instead of mid-swing.
    """
    try:
        _set_profile_velocity(backend, RAMP_PROFILE_VELOCITY)
        ramp_to(backend, stance_pose(controller), duration=1.0, rate=args.rate)
    except Exception as exc:
        print(f"settle skipped: {exc}", file=sys.stderr)


# Every line in the panel starts with the same 8-column label, so the whole
# thing reads as one table. `what:keys` pairs put the action first — you scan
# for the thing you want to do, not for a key. Spacing rather than semicolons
# separates them, because `;` is itself the smoothing binding.
_LABEL = 8

HELP_LINES = [
    f"  {'drive':<{_LABEL}}   fwd: w/s   strafe: a/d      turn: q/e",
    f"  {'posture':<{_LABEL}} pitch: i/k     roll: j/l    height: r/f",
    f"  {'feel':<{_LABEL}}  sens: [ ]   smooth: ; '      fine: SHIFT",
    f"  {'mode':<{_LABEL}}  gait: 1/2     stop: SPACE   estop: x   release: z   deadman: g   hide: h   quit: ESC",
]


def _health_line(health: dict) -> str:
    """One line summarising the servo fault/temperature sweep."""
    if not health:
        return f"  {'health':<{_LABEL}}no feedback"
    faults = [name for name, (err, _) in health.items() if err]
    hottest, temp = max(health.items(), key=lambda kv: kv[1][1])
    body = f"{len(health)}/12 replying; hottest {hottest} {temp} C"
    return f"  {'health':<{_LABEL}}{body}" + (f"; FAULT {', '.join(faults)}" if faults else "")


def _hud(shaper: CommandShaper, controller: LocomotionController, show_help: bool,
         health: dict = None) -> str:
    """Two columns mirroring the key groups: what you drive, how it's carried.

    One value per row, and every line shares the same label column so the panel
    reads as a single table. Fixed settings (step sizes, the shift rule) sit on
    the `feel` line rather than being repeated per row.
    """
    a = shaper.axes
    deg = math.degrees

    def row(label, axis, unit, scale=1.0, fmt="{:>6.1f}"):
        """`label  value  unit` — the commanded value, one number.

        The setpoint isn't shown: it only differs from the actual value during
        the brief slew ramp, so a second column was the same number twice
        almost all the time.
        """
        # `or 0.0` collapses negative zero, which the pitch row's sign flip
        # produces at rest and which reads as a fault.
        value = fmt.format((axis.actual * scale) or 0.0)
        return f"{label:<{_LABEL}}{value} {unit:<5}"

    drive = [
        row("vx", a["vx"], "mm/s"),
        row("vy", a["vy"], "mm/s"),
        row("turn", a["yaw_rate"], "rad/s", fmt="{:>6.2f}"),
    ]
    posture = [
        # Shown nose-up-positive; BodyPose.pitch is nose-down-positive.
        row("pitch", a["pitch"], "deg", -deg(1.0)),
        row("roll", a["roll"], "deg", deg(1.0)),
        row("height", a["height"], "mm"),
    ]

    mode = "E-STOP (frozen; z releases torque)" if shaper.estopped else (
        "hold" if not shaper.deadman else f"deadman {shaper.decay_after:.1f}s")
    gutter = 4
    width = len(drive[0]) + gutter

    lines = ["  " + "DRIVE".ljust(width) + "POSTURE"]
    lines += [f"  {d}{' ' * gutter}{p}".rstrip() for d, p in zip(drive, posture)]
    lines += [
        "",
        f"  {'gait':<{_LABEL}}{controller.gait}; {controller.stance_summary()}; {mode}",
        f"  {'odom':<{_LABEL}}x {controller.odom_x / 1000:+.2f} m;"
        f" y {controller.odom_y / 1000:+.2f} m;"
        f" heading {deg(controller.odom_yaw) % 360:.1f} deg",
        f"  {'feel':<{_LABEL}}tap {a['vx'].step:g} mm/s; ramp {a['vx'].slew:g} mm/s^2;"
        f" pitch {deg(a['pitch'].step):g} deg; height {a['height'].step:g} mm; SHIFT = x0.1",
    ]
    if health is not None:
        lines.append(_health_line(health))
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
    parser.add_argument("--decay-after", type=float, default=None,
                        help="Deadman: seconds without a key before velocity decays to zero. "
                             "0 means setpoints hold until you change them (the sim default). "
                             "Defaults to 0.3 on hardware backends; toggle live with G.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until quit)")
    parser.add_argument("--stand-time", type=float, default=2.0,
                        help="Seconds to ramp from the present pose into stance (default 2).")
    parser.add_argument("--no-wait", action="store_true",
                        help="Start the gait as soon as the stand-up ramp finishes, "
                             "without waiting for a keypress.")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    # Setpoints holding is right for a sim window you're watching; on hardware
    # it means a dropped SSH session leaves the robot driving away.
    if args.decay_after is None:
        args.decay_after = 0.0 if args.backend == "sim" else 0.3

    install_signal_handlers()
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

    with backend, RawKeys() as keys:
        if not keys.interactive:
            print("stdin is not a tty — running with no keyboard input.", file=sys.stderr)

        # Stand up before the gait touches the servos. Without this, tick 0 is a
        # full-amplitude gait pose commanded from wherever the legs happen to be.
        _set_profile_velocity(backend, RAMP_PROFILE_VELOCITY)
        print(f"standing up over {args.stand_time:.1f}s — keep clear.")
        last_targets = ramp_to(backend, stance_pose(controller),
                               duration=args.stand_time, rate=args.rate)
        if not _wait_for_go(keys, args.no_wait):
            print("\nquit before the gait started.")
            return 0
        _set_profile_velocity(backend, GAIT_PROFILE_VELOCITY)

        health: dict = {}
        ticks = 0
        t0 = time.perf_counter()   # --duration times the gait, not the stand-up
        deadline = t0
        try:
            while True:
                for k in keys.read():
                    if k == "h":
                        show_help = not show_help
                        shaper.note_input()
                        continue
                    if not shaper.key(k):
                        raise KeyboardInterrupt
                if shaper.torque_kill:
                    print("\ntorque released — the robot is limp.")
                    backend.set_torque_all(False)
                    return 0
                if shaper.gait != controller.gait:
                    controller.set_gait(shaper.gait)

                cmd = shaper.update(dt)
                if shaper.estopped:
                    # Freeze: do NOT advance the gait. Zeroing the body command
                    # alone still leaves each foot lifting 30 mm every cycle.
                    backend.set_joint_targets(last_targets)
                else:
                    last_targets = controller.step(
                        dt, body_velocity=cmd.body_velocity,
                        yaw_rate=cmd.yaw_rate, pose=cmd.pose)
                    backend.set_joint_targets(last_targets)
                    backend.set_base_pose(*controller.base_pose(cmd.pose))

                ticks += 1
                if ticks % HEALTH_EVERY_TICKS == 0:
                    health = backend.health_check()

                if keys.interactive:
                    frame = _hud(shaper, controller, show_help,
                                 health if health else None)
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
        except Exception as exc:
            # Settle rather than unwinding straight into disconnect()'s torque-off.
            print(f"\ncontrol loop failed: {exc}", file=sys.stderr)

        _settle(backend, controller, args)

    print("\nteleop stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
