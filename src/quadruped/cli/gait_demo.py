"""Full quadruped gait demo — walk or trot through any RobotBackend.

Drives all four legs through GaitScheduler (cubic Bezier swing + linear
stance, Raibert foothold planning). Each tick:
  1. scheduler.advance(dt, body_velocity)
  2. per leg: body-frame foot position → leg-local via inverse shoulder yaw
  3. closed-form IK (kinematics.ik.joint_angles) → (shoulder, wing, knee)
  4. assemble 12-DOF joint target dict and call backend.set_joint_targets()

This unifies sim and hardware: the dict is the same; only the backend differs.

Per-leg shoulder pose is read once from the MJCF at startup (geometric
truth). The canonical resting foot position in any leg's local frame
comes from configs/robot.yaml (target_site_offset_mm).

Examples:
    # MuJoCo viewer, stepping in place (walk)
    quad-gait-demo --backend sim --viewer

    # Trot with small body forward velocity
    quad-gait-demo --backend sim --viewer --gait trot --vx 50

    # Real hardware (verify in sim first!)
    quad-gait-demo --backend hw -p /dev/cu.usbserial-XXXX
"""
import argparse
import math
import sys
import time

import mujoco
import numpy as np

from quadruped.backends import MujocoBackend
from quadruped.cli._backends import add_backend_args, build_backend
from quadruped.config import CONFIG
from quadruped.control.gait import GaitScheduler, TROT_PRESET, WALK_PRESET
from quadruped.kinematics.ik import joint_angles
from quadruped.sim.env import model_path


def _leg_yaw_rad(model: mujoco.MjModel, leg: str) -> float:
    # All four shoulder bodies use a pure z-yaw quaternion, so
    # quat = (cos(θ/2), 0, 0, sin(θ/2)).
    w, _, _, z = model.body(f"shoulder_{leg}").quat
    return 2.0 * float(math.atan2(z, w))


def _rotz_xy(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _build_leg_pose(model: mujoco.MjModel) -> dict:
    # The hip-frame origin FK/IK rotate about is the shoulder *rotation axis*,
    # not the shoulder body origin: the joint sits at a local offset inside the
    # body (shoulder_joint pos in the MJCF). Recover the axis in body frame as
    # body.pos + Rz(yaw)·jnt_pos so it lands on the design corner (±75, ±75)
    # regardless of that offset.
    pose = {}
    for leg in CONFIG.legs:
        body = model.body(f"shoulder_{leg}")
        yaw = _leg_yaw_rad(model, leg)
        jnt_pos = np.asarray(model.joint(f"shoulder_joint_{leg}").pos, dtype=float)
        xy = _rotz_xy(yaw) @ jnt_pos[:2]
        axis = np.asarray(body.pos, dtype=float) + np.array([xy[0], xy[1], jnt_pos[2]])
        pose[leg] = (axis, yaw)
    return pose


def _body_to_local(p_body: np.ndarray, shoulder_pos: np.ndarray, yaw: float) -> np.ndarray:
    delta = p_body - shoulder_pos
    xy = _rotz_xy(-yaw) @ delta[:2]
    return np.array([xy[0], xy[1], delta[2]])


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
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: run until Ctrl+C / viewer close)")
    add_backend_args(parser, default="sim")
    args = parser.parse_args()

    preset = WALK_PRESET if args.gait == "walk" else TROT_PRESET
    dt = 1.0 / args.rate

    # MJCF is loaded once just to extract per-leg shoulder pose (the geometric
    # source of truth for leg layout). Cheap even for hw-only mode.
    model = mujoco.MjModel.from_xml_path(str(model_path()))
    pose = _build_leg_pose(model)

    # Canonical resting foot position in any leg's local frame.
    home_local = np.array(CONFIG.target_site_offset_mm, dtype=float)
    home_body = {}
    for leg in CONFIG.legs:
        shoulder_pos, yaw = pose[leg]
        xy_body = _rotz_xy(yaw) @ home_local[:2]
        home_body[leg] = shoulder_pos + np.array([xy_body[0], xy_body[1], home_local[2]])

    scheduler = GaitScheduler(**preset, home_positions=home_body)
    body_velocity = (args.vx, args.vy)

    print(f"gait={args.gait}  rate={args.rate:g} Hz  vx={args.vx:g}  vy={args.vy:g}  backend={args.backend}")
    for leg in CONFIG.legs:
        sp, yaw = pose[leg]
        print(f"  {leg}: shoulder@({sp[0]:6.1f},{sp[1]:6.1f},{sp[2]:6.1f}) "
              f"yaw={math.degrees(yaw):+7.2f}°  "
              f"home_body=({home_body[leg][0]:7.1f},{home_body[leg][1]:7.1f},{home_body[leg][2]:7.1f})")

    backend = build_backend(args)
    with backend:
        t0 = time.perf_counter()
        deadline = t0
        try:
            while True:
                scheduler.advance(dt, body_velocity)
                targets: dict[str, float] = {}
                for leg in CONFIG.legs:
                    p_body = scheduler.foot_position(leg)
                    shoulder_pos, yaw = pose[leg]
                    p_local = _body_to_local(p_body, shoulder_pos, yaw)
                    q_sh, q_wg, q_kn = joint_angles(p_local[0], p_local[1], p_local[2])
                    targets[f"shoulder_{leg}"] = float(q_sh)
                    targets[f"wing_{leg}"] = float(q_wg)
                    targets[f"knee_{leg}"] = float(q_kn)
                backend.set_joint_targets(targets)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
