#!/usr/bin/env python3
"""
Send a Cartesian foot target (x, y, z in mm) to the leg.
Runs IK with the single_leg MuJoCo model, then sends the resulting joint angles
to the Arduino (same protocol as send_angles.py).

Usage:
  python send_foot_position.py <x> <y> <z>  [--port PORT]
  python send_foot_position.py 20 -175 100 --port /dev/cu.usbmodem14101

Coordinates are in the same world frame as single_leg.xml (mm).
Requires: mujoco, numpy, pyserial
"""

import argparse
import os
import sys

import numpy as np
import mujoco

try:
    import serial
except ImportError:
    print("Need pyserial: pip install pyserial", file=sys.stderr)
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "single_leg.xml")

# target_site is at offset (20, -175, -50) from the mocap body in single_leg.xml
# so mocap_pos = (x,y,z) - (20, -175, -50) to put target at (x,y,z)
TARGET_SITE_OFFSET = np.array([20.0, -175.0, -50.0])


def load_model_and_indices():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    foot_site_id = model.site("foot_site").id
    target_site_id = model.site("target_site").id
    dof_idxs = np.array([
        model.joint("shoulder_joint").dofadr[0],
        model.joint("wing_joint").dofadr[0],
        model.joint("knee_joint").dofadr[0],
    ], dtype=int)
    qpos_idxs = np.array([
        model.joint("shoulder_joint").qposadr[0],
        model.joint("wing_joint").qposadr[0],
        model.joint("knee_joint").qposadr[0],
    ], dtype=int)
    jids = np.array([
        model.joint("shoulder_joint").id,
        model.joint("wing_joint").id,
        model.joint("knee_joint").id,
    ], dtype=int)
    return model, data, foot_site_id, target_site_id, dof_idxs, qpos_idxs, jids


def clamp_to_limits(model, data, qpos_idxs, jids, q):
    q_clamped = q.copy()
    for i, jid in enumerate(jids):
        lo, hi = model.jnt_range[jid]
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)
    return q_clamped


def solve_ik(model, data, foot_site_id, target_site_id, dof_idxs, qpos_idxs, jids,
             target_world_mm, alpha=0.6, damping=1e-2, tol=1e-2, max_iter=200):
    """Run damped least-squares IK so foot_site reaches target_world_mm (x,y,z in mm)."""
    # set mocap so target_site is at target_world_mm
    mocap_pos = target_world_mm - TARGET_SITE_OFFSET
    data.mocap_pos[0] = mocap_pos

    # start from current q (or zero)
    q = data.qpos[qpos_idxs].copy()
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        p_foot = data.site_xpos[foot_site_id].copy()
        p_target = data.site_xpos[target_site_id].copy()
        err = p_target - p_foot
        if np.linalg.norm(err) <= tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, foot_site_id)
        J = jacp[:, dof_idxs]
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)
        q = clamp_to_limits(model, data, qpos_idxs, jids, q + alpha * dq)
        data.qpos[qpos_idxs] = q
    return q


def send_angles(port: str, q1: float, q2: float, q3: float, baud: int = 115200) -> str:
    with serial.Serial(port, baud, timeout=2.0) as ser:
        line = f"q,{q1},{q2},{q3}\n"
        ser.write(line.encode("ascii"))
        reply = ser.readline().decode("ascii", errors="ignore").strip()
    return reply


def main():
    parser = argparse.ArgumentParser(
        description="Send foot target (x,y,z mm) to leg: run IK, then send joint angles."
    )
    parser.add_argument("x", type=float, help="Target X (mm)")
    parser.add_argument("y", type=float, help="Target Y (mm)")
    parser.add_argument("z", type=float, help="Target Z (mm)")
    parser.add_argument("--port", "-p", default=None, help="Serial port")
    parser.add_argument("--baud", "-b", type=int, default=115200, help="Baud rate")
    parser.add_argument("--dry-run", action="store_true", help="Only run IK, print angles, do not send")
    args = parser.parse_args()

    port = args.port or os.environ.get("SERIAL_PORT")
    if not port and not args.dry_run:
        print("Specify --port or SERIAL_PORT (or use --dry-run)", file=sys.stderr)
        sys.exit(1)

    model, data, foot_site_id, target_site_id, dof_idxs, qpos_idxs, jids = load_model_and_indices()
    target_mm = np.array([args.x, args.y, args.z])
    q = solve_ik(model, data, foot_site_id, target_site_id, dof_idxs, qpos_idxs, jids, target_mm)

    print(f"target (mm): {args.x} {args.y} {args.z}")
    print(f"angles (rad): {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}")

    if args.dry_run:
        return
    reply = send_angles(port, float(q[0]), float(q[1]), float(q[2]), args.baud)
    if reply:
        print(reply)


if __name__ == "__main__":
    main()
