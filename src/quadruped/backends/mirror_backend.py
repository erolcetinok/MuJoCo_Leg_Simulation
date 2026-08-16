"""MirrorBackend: fan a single command stream to multiple backends.

Typical use: drive a sim and the physical leg from the same slider so you can
visually validate sim-to-real. Reads return the designated `truth_source`'s
state (default: the first backend that returns a non-empty qpos dict).
"""
from __future__ import annotations

from typing import Sequence

from quadruped.backends.base import RobotBackend


class MirrorBackend(RobotBackend):
    def __init__(
        self,
        backends: Sequence[RobotBackend],
        *,
        truth_source: int | None = None,
    ) -> None:
        if not backends:
            raise ValueError("MirrorBackend needs at least one backend")
        self.backends: tuple[RobotBackend, ...] = tuple(backends)
        self._truth_idx = truth_source

    def connect(self) -> None:
        for b in self.backends:
            b.connect()

    def disconnect(self) -> None:
        for b in reversed(self.backends):
            try:
                b.disconnect()
            except Exception:
                pass

    def set_joint_targets(self, q: dict[str, float]) -> None:
        for b in self.backends:
            b.set_joint_targets(q)

    def read_joint_state(self) -> tuple[dict[str, float], dict[str, float]]:
        if self._truth_idx is not None:
            return self.backends[self._truth_idx].read_joint_state()
        for b in self.backends:
            qpos, qvel = b.read_joint_state()
            if qpos:
                return qpos, qvel
        return {}, {}

    def set_torque_all(self, on: bool) -> None:
        # Must fan out: inheriting the ABC's no-op would make the teleop
        # torque-kill silently do nothing on --backend mirror.
        for b in self.backends:
            b.set_torque_all(on)

    def health_check(self) -> dict:
        for b in self.backends:
            health = b.health_check()
            if health:
                return health
        return {}
