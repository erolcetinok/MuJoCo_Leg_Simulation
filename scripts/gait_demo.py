"""Full quadruped gait demo — walk or trot through any RobotBackend.

Scripted counterpart to scripts/teleop.py: the command is fixed at launch
instead of coming from the keyboard, which makes this the reproducible one to
reach for when checking a change or recording a clip.

Drives all four legs through LocomotionController each tick:
  1. gait phase -> per-leg foot target in body frame (Raibert footholds)
  2. body pose applied, then converted to each leg's local frame
  3. closed-form IK -> (shoulder, wing, knee)
  4. 12-DOF joint dict -> backend.set_joint_targets()

Sim and hardware differ only in the backend; the joint dict is identical.

Examples:
    # MuJoCo viewer, stepping in place (walk)
    mjpython scripts/gait_demo.py --backend sim --viewer

    # Trot with body forward velocity
    mjpython scripts/gait_demo.py --backend sim --viewer --gait trot --vx 50

    # Turn in place, then an arc
    mjpython scripts/gait_demo.py --backend sim --viewer --yaw-rate 0.5
    mjpython scripts/gait_demo.py --backend sim --viewer --vx 40 --yaw-rate 0.3

    # Real hardware (verify in sim first!)
    python scripts/gait_demo.py --backend hw -p /dev/cu.usbserial-XXXX
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import mujoco

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.backends import MujocoBackend
from quadruped.cli_args import add_backend_args, build_backend, install_signal_handlers
from quadruped.config import CONFIG
from quadruped.control.body import BodyPose
from quadruped.control.locomotion import LocomotionController
from quadruped.control.sequencer import ramp_to, stance_pose
from quadruped.sim.env import leg_poses, model_path

RAMP_PROFILE_VELOCITY = 30
GAIT_PROFILE_VELOCITY = 0


def _set_profile_velocity(backend, value: int) -> None:
    """Retune the servo velocity cap where the backend has one; ignore where not."""
    for b in getattr(backend, "backends", [backend]):
        setter = getattr(b, "set_profile_velocity", None)
        if setter is not None:
            setter(value)


def _settle(backend, controller, args) -> None:
    """Return to the symmetric stance pose before disconnect() cuts torque.

    Not a sit-down — there is no crouch authority in the command envelope (see
    control/sequencer.py). It just makes the drop happen from a known,
    four-feet-planted pose instead of mid-swing.
    """
    try:
        _set_profile_velocity(backend, RAMP_PROFILE_VELOCITY)
        ramp_to(backend, stance_pose(controller), duration=1.0, rate=args.rate)
    except Exception as exc:
        print(f"settle skipped: {exc}", file=sys.stderr)


def _any_viewer_closed(backend) -> bool:
    """True if `backend` owns a passive viewer that has been closed by the user.

    Returns False for backends with no viewer (so hw / sim-without-viewer
    runs proceed). Walks one level of MirrorBackend.backends.
    """
    candidates = getattr(backend, "backends", [backend])
    for b in candidates:
        if isinstance(b, MujocoBackend) and b._viewer is not None and not b.viewer_alive():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", choices=["walk", "trot"], default="walk")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Control rate in Hz (default 50)")
    parser.add_argument("--vx", type=float, default=0.0,
                        help="Body forward velocity in body x (mm/s; 0 = step in place)")
    parser.add_argument("--vy", type=float, default=0.0,
                        help="Body lateral velocity in body y (mm/s)")
    parser.add_argument("--yaw-rate", type=float, default=0.0,
                        help="Body turn rate about z (rad/s, + = left; 0 = straight)")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Body pitch (degrees, + = nose up)")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="Body roll (degrees, + = right side down)")
    parser.add_argument("--height", type=float, default=0.0,
                        help="Ride-height offset from nominal stance (mm, + = taller)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until Ctrl+C / viewer close)")
    parser.add_argument("--stand-time", type=float, default=2.0,
                        help="Seconds to ramp from the present pose into stance (default 2).")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt after the stand-up ramp.")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    install_signal_handlers()
    dt = 1.0 / args.rate

    # MJCF is loaded once just to extract per-leg shoulder pose (the geometric
    # source of truth for leg layout). Cheap even for hw-only mode.
    model = mujoco.MjModel.from_xml_path(str(model_path()))
    poses = leg_poses(model)
    controller = LocomotionController(poses, gait=args.gait)

    # --pitch is nose-up-positive for the operator; BodyPose is right-handed
    # about +y, where positive is nose down. Negate once, here.
    pose = BodyPose(
        roll=math.radians(args.roll),
        pitch=-math.radians(args.pitch),
        z=args.height,
    ).clamped()
    body_velocity = (args.vx, args.vy)

    print(f"gait={args.gait}  rate={args.rate:g} Hz  vx={args.vx:g}  vy={args.vy:g}  "
          f"yaw_rate={args.yaw_rate:g}  backend={args.backend}")
    for leg in CONFIG.legs:
        sp, yaw = poses[leg]
        home = controller.home[leg]
        print(f"  {leg}: shoulder@({sp[0]:6.1f},{sp[1]:6.1f},{sp[2]:6.1f}) "
              f"yaw={math.degrees(yaw):+7.2f} deg  "
              f"home_body=({home[0]:7.1f},{home[1]:7.1f},{home[2]:7.1f})")

    backend = build_backend(args)
    with backend:
        # Stand up first: without this, tick 0 commands a full-amplitude gait
        # pose from wherever the legs physically are, at max servo speed.
        _set_profile_velocity(backend, RAMP_PROFILE_VELOCITY)
        print(f"standing up over {args.stand_time:.1f}s — keep clear.")
        ramp_to(backend, stance_pose(controller), duration=args.stand_time, rate=args.rate)
        if not args.yes and sys.stdin.isatty():
            try:
                input("in stance. press ENTER to start walking, Ctrl-C to quit: ")
            except (EOFError, KeyboardInterrupt):
                print("\nStopped before the gait started.")
                _settle(backend, controller, args)
                return 0
        _set_profile_velocity(backend, GAIT_PROFILE_VELOCITY)

        t0 = time.perf_counter()
        deadline = t0
        try:
            while True:
                targets = controller.step(
                    dt,
                    body_velocity=body_velocity,
                    yaw_rate=args.yaw_rate,
                    pose=pose,
                )
                backend.set_joint_targets(targets)
                backend.set_base_pose(*controller.base_pose(pose))

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
            print("\nStopped by user.")
        except Exception as exc:
            print(f"\ncontrol loop failed: {exc}", file=sys.stderr)
        _settle(backend, controller, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
