"""MuJoCo backend: drives data.ctrl, optionally with a passive viewer attached."""
from __future__ import annotations

import time
from typing import Optional

import mujoco
import numpy as np

from quadruped.backends.base import RobotBackend
from quadruped.config import CONFIG
from quadruped.sim.env import load_model


class MujocoBackend(RobotBackend):
    def __init__(
        self,
        xml: Optional[str] = None,
        *,
        use_viewer: bool = False,
        step: bool = True,
    ) -> None:
        self._xml = xml
        self._use_viewer = use_viewer
        self._step = step
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._viewer = None
        self._actuator_idx: dict[str, int] = {}
        self._qpos_idx: dict[str, int] = {}
        self._qvel_idx: dict[str, int] = {}

    def connect(self) -> None:
        self.model, self.data = load_model(self._xml)
        for joint in CONFIG.joints:
            mjcf_name = next(
                (n for n in CONFIG.mjcf.joint_names if n.startswith(joint.name)),
                None,
            )
            if mjcf_name is None:
                continue
            jnt = self.model.joint(mjcf_name)
            self._qpos_idx[joint.name] = int(jnt.qposadr[0])
            self._qvel_idx[joint.name] = int(jnt.dofadr[0])
            # actuator name follows the same prefix convention
            for ai in range(self.model.nu):
                if self.model.actuator(ai).name.startswith(joint.name):
                    self._actuator_idx[joint.name] = ai
                    break
        mujoco.mj_forward(self.model, self.data)
        if self._use_viewer:
            # Import as `mj_viewer`, not `import mujoco.viewer`: the latter
            # rebinds the name `mujoco` as a local, shadowing the module-level
            # import and making every `mujoco.*` call above this line raise
            # UnboundLocalError.
            from mujoco import viewer as mj_viewer  # delayed import
            self._viewer = mj_viewer.launch_passive(self.model, self.data)

    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def set_joint_targets(self, q: dict[str, float]) -> None:
        assert self.data is not None and self.model is not None
        for name, value in q.items():
            ai = self._actuator_idx.get(name)
            if ai is None:
                continue
            self.data.ctrl[ai] = float(value)
        if self._step:
            mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def read_joint_state(self) -> tuple[dict[str, float], dict[str, float]]:
        assert self.data is not None
        qpos = {name: float(self.data.qpos[i]) for name, i in self._qpos_idx.items()}
        qvel = {name: float(self.data.qvel[i]) for name, i in self._qvel_idx.items()}
        return qpos, qvel

    def viewer_alive(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()
