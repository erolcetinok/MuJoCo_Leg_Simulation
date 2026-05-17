"""RobotBackend ABC.

A backend is anything that can accept joint targets and report current joint
state. Concrete impls: MujocoBackend (sim), ArduinoBackend (serial bridge to
the leg_controller sketch), MirrorBackend (fans out to both),
DynamixelBackend (future U2D2 path).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RobotBackend(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Acquire resources (open port, launch viewer, etc.)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources."""

    @abstractmethod
    def set_joint_targets(self, q: dict[str, float]) -> None:
        """Command joints to the given angles, in radians, keyed by joint name."""

    @abstractmethod
    def read_joint_state(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return (qpos, qvel) dicts keyed by joint name. Either may be empty
        if the backend does not provide that channel."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False
