#!/usr/bin/env python3
"""Measure ArduinoBackend round-trip latency (write + readline)."""
import time

from quadruped.backends import ArduinoBackend

b = ArduinoBackend()
with b:
    for _ in range(10):
        t = time.perf_counter()
        b.set_joint_targets({'shoulder': 0.0, 'wing': -0.4, 'knee': 0.05})
        print(f'{(time.perf_counter() - t) * 1000:6.1f} ms  reply={b.last_reply!r}')
