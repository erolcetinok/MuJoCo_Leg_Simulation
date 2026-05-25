#!/usr/bin/env python3
"""Run a swing trajectory while printing per-sample round-trip and reply.

This is the actual control loop in swing_hw, instrumented. Any 2+ second
gap will show up clearly.
"""
import time

from quadruped.backends import ArduinoBackend
from quadruped.config import CONFIG
from quadruped.control.trajectory import SwingFootTrajectory
from quadruped.kinematics.ik import joint_angles


def angles_dict(x, y, z):
    q = joint_angles(x, y, z)
    return {name: float(v) for name, v in zip(CONFIG.joint_names, q)}


lift = (-30.0, -175.0, -50.0)
touch = (30.0, -175.0, -50.0)
apex_height = lift[2] + 30.0
T_swing = 0.8
rate = 33.0
dt = 1.0 / rate
n = int(round(T_swing * rate)) + 1

traj = SwingFootTrajectory(lift, touch, apex_height, T_swing)

b = ArduinoBackend()
with b:
    print(f"Pre-pose -> {lift}, settling 1.0 s ...")
    b.set_joint_targets(angles_dict(*lift))
    time.sleep(1.0)

    print(f"\nRunning swing ({n} samples @ {rate} Hz, dt = {dt*1000:.0f} ms):")
    print(f"{'i':>3} {'s':>5} {'rt_ms':>7} {'gap_ms':>7}  reply")
    prev_end = time.perf_counter()
    for i in range(n):
        s = i / (n - 1)
        p = traj.position_at(s)
        t0 = time.perf_counter()
        gap_ms = (t0 - prev_end) * 1000
        b.set_joint_targets(angles_dict(float(p[0]), float(p[1]), float(p[2])))
        rt_ms = (time.perf_counter() - t0) * 1000
        prev_end = time.perf_counter()
        print(f"{i:>3} {s:>5.2f} {rt_ms:>7.1f} {gap_ms:>7.1f}  {b.last_reply!r}")
        slack = dt - rt_ms / 1000
        if slack > 0:
            time.sleep(slack)
