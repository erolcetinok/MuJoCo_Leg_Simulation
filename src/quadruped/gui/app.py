"""Dear PyGui slider app — Landing 2 + Landing 3 (embedded MuJoCo renderer).

Single window, single thread. Three view modes:
  - embedded (default): MuJoCo renders offscreen into a DPG dynamic texture so
    the whole UI lives in one DPG viewport. No second GLFW context — works
    cleanly on macOS where the passive viewer can fight DPG for the main thread.
  - external: launch mujoco.viewer.launch_passive in a separate window. Useful
    if you want MuJoCo's full interactive viewer (mouse-camera control, contact
    visualization). May misbehave on macOS — slider-only / embedded preferred.
  - none: slider-only, no sim visualization.

Modes:
  - joint:     direct sliders per joint (radians, bounds from configs/robot.yaml)
  - Cartesian: x/y/z sliders, runs IK each frame

Backend toggle (sim / hw / mirror) switchable at runtime. Backends control
where commands GO; the embedded renderer always shows the IK model's commanded
pose, which is the most useful visualization in every backend mode.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import mujoco

from quadruped.config import CONFIG
from quadruped.backends import ArduinoBackend, MirrorBackend, MujocoBackend, RobotBackend
from quadruped.kinematics.ik import DampedLeastSquaresIK
from quadruped.sim.env import load_model


VIEW_MODES = ("embedded", "external", "none")


def _build_backend(kind: str, port: Optional[str], external_viewer: bool) -> RobotBackend:
    if kind == "sim":
        return MujocoBackend(use_viewer=external_viewer, step=False)
    if kind == "hw":
        return ArduinoBackend(port=port)
    if kind == "mirror":
        sim = MujocoBackend(use_viewer=external_viewer, step=False)
        hw = ArduinoBackend(port=port)
        return MirrorBackend([sim, hw], truth_source=1)
    raise ValueError(f"unknown backend: {kind!r}")


class GuiApp:
    def __init__(
        self,
        *,
        backend_kind: str = "sim",
        port: Optional[str] = None,
        view_mode: str = "embedded",
        hz: int = 100,
        render_size: tuple[int, int] = (480, 360),  # (width, height)
        render_hz: int = 30,
    ) -> None:
        if view_mode not in VIEW_MODES:
            raise ValueError(f"view_mode must be one of {VIEW_MODES}, got {view_mode!r}")
        self.backend_kind = backend_kind
        self.port = port
        self.view_mode = view_mode
        self.dt = 1.0 / hz
        self.render_period = 1.0 / render_hz
        self._render_w, self._render_h = render_size
        self.backend: Optional[RobotBackend] = None

        # IK / preview model (always owned by the GUI, separate from any sim backend).
        self.ik_model, self.ik_data = load_model()
        self.ik = DampedLeastSquaresIK(
            self.ik_model, self.ik_data,
            foot_site_name=CONFIG.mjcf.foot_site_name,
            target_site_name=CONFIG.mjcf.target_site_name,
            joint_names=CONFIG.mjcf.joint_names,
            max_iter=4,
        )
        self._target_default = np.asarray(CONFIG.target_site_offset_mm, dtype=float)
        self._joint_qpos_idx = {
            j.name: int(self.ik_model.joint(
                next(n for n in CONFIG.mjcf.joint_names if n.startswith(j.name))
            ).qposadr[0])
            for j in CONFIG.joints
        }

        # Embedded renderer state — created lazily after DPG context exists.
        self._renderer: Optional[mujoco.Renderer] = None
        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self._cam)
        mujoco.mjv_defaultOption(self._opt)
        # Reasonable starting view: look at the leg from the front-right, ~600 mm out.
        self._cam.azimuth = 135.0
        self._cam.elevation = -20.0
        self._cam.distance = 600.0
        self._cam.lookat = np.array([0.0, 0.0, 80.0])

        self._tex_tag = "mj_tex"
        self._last_render = 0.0

    # --- backend lifecycle ---

    def _open_backend(self, kind: str) -> None:
        self._close_backend()
        try:
            b = _build_backend(kind, self.port, external_viewer=(self.view_mode == "external"))
            b.connect()
            self.backend = b
            self.backend_kind = kind
            self._status(f"backend = {kind}")
        except Exception as e:  # noqa: BLE001
            self.backend = None
            self._status(f"backend open failed ({kind}): {e}")

    def _close_backend(self) -> None:
        if self.backend is not None:
            try:
                self.backend.disconnect()
            except Exception:
                pass
            self.backend = None

    def _send(self, q: dict[str, float]) -> None:
        if self.backend is None:
            return
        try:
            self.backend.set_joint_targets(q)
        except Exception as e:  # noqa: BLE001
            self._status(f"send failed: {e}")

    def _status(self, msg: str) -> None:
        from dearpygui import dearpygui as dpg
        if dpg.does_item_exist("status"):
            dpg.set_value("status", msg)
        else:
            print(msg, file=sys.stderr)

    def _reset_sliders(self) -> None:
        from dearpygui import dearpygui as dpg
        for joint in CONFIG.joints:
            dpg.set_value(f"joint_{joint.name}", 0.0)
        for axis, default in zip("xyz", self._target_default):
            dpg.set_value(f"cart_{axis}", float(default))

    # --- rendering ---

    def _render_to_texture(self) -> None:
        from dearpygui import dearpygui as dpg
        if self._renderer is None:
            return
        mujoco.mj_forward(self.ik_model, self.ik_data)
        self._renderer.update_scene(self.ik_data, camera=self._cam, scene_option=self._opt)
        pixels = self._renderer.render()  # (H, W, 3) uint8
        # DPG dynamic textures want flat RGBA floats in 0..1.
        rgba = np.empty((pixels.shape[0], pixels.shape[1], 4), dtype=np.float32)
        rgba[..., :3] = pixels.astype(np.float32) * (1.0 / 255.0)
        rgba[..., 3] = 1.0
        dpg.set_value(self._tex_tag, rgba.ravel().tolist())

    # --- main loop ---

    def run(self) -> int:
        from dearpygui import dearpygui as dpg

        dpg.create_context()

        if self.view_mode == "embedded":
            self._renderer = mujoco.Renderer(
                self.ik_model, height=self._render_h, width=self._render_w
            )
            with dpg.texture_registry():
                # Solid-gray placeholder until the first render.
                placeholder = [0.2, 0.2, 0.2, 1.0] * (self._render_w * self._render_h)
                dpg.add_dynamic_texture(
                    width=self._render_w, height=self._render_h,
                    default_value=placeholder, tag=self._tex_tag,
                )

        with dpg.window(label="Quadruped Teleop", tag="main", width=1100, height=620):
            with dpg.group(horizontal=True):
                # Left column: controls
                with dpg.child_window(width=520, height=560, border=False):
                    dpg.add_text(f"Robot: {CONFIG.robot}  |  joints: {', '.join(CONFIG.joint_names)}")
                    dpg.add_separator()

                    with dpg.group(horizontal=True):
                        dpg.add_text("Backend")
                        dpg.add_combo(
                            items=["sim", "hw", "mirror"],
                            default_value=self.backend_kind,
                            width=120, tag="backend_combo",
                            callback=lambda s, v: self._open_backend(v),
                        )
                        dpg.add_button(label="Reset", callback=self._reset_sliders)
                    dpg.add_text("port: " + (self.port or "(SERIAL_PORT)"))
                    dpg.add_separator()

                    with dpg.group(horizontal=True):
                        dpg.add_text("Mode")
                        dpg.add_combo(
                            items=["joint", "cartesian"], default_value="joint",
                            width=120, tag="mode_combo",
                        )
                    dpg.add_separator()

                    dpg.add_text("Joint targets (rad)")
                    for joint in CONFIG.joints:
                        lo, hi = joint.limit_rad
                        dpg.add_slider_float(
                            label=joint.name, tag=f"joint_{joint.name}",
                            min_value=float(lo), max_value=float(hi),
                            default_value=0.0, width=380,
                        )
                    dpg.add_separator()

                    base = self._target_default
                    cart_ranges = {
                        "x": (base[0] - 60.0, base[0] + 60.0),
                        "y": (base[1] - 30.0, base[1] + 30.0),
                        "z": (base[2] - 60.0, base[2] + 80.0),
                    }
                    dpg.add_text("Cartesian foot target (mm)")
                    for axis, (lo, hi) in cart_ranges.items():
                        dpg.add_slider_float(
                            label=axis, tag=f"cart_{axis}",
                            min_value=float(lo), max_value=float(hi),
                            default_value=float(base["xyz".index(axis)]),
                            width=380,
                        )
                    dpg.add_separator()

                    dpg.add_text("Current joint state (rad):")
                    for joint in CONFIG.joints:
                        dpg.add_text(f"  {joint.name}: --", tag=f"state_{joint.name}")
                    dpg.add_separator()
                    dpg.add_text("", tag="status", wrap=480)

                # Right column: embedded MuJoCo viewer (or note if disabled)
                if self.view_mode == "embedded":
                    with dpg.child_window(width=self._render_w + 16, height=560, border=False):
                        dpg.add_text("Sim view (IK model)")
                        dpg.add_image(self._tex_tag)
                        dpg.add_separator()
                        dpg.add_text("Camera")
                        dpg.add_slider_float(
                            label="azimuth", tag="cam_az",
                            min_value=-180.0, max_value=180.0,
                            default_value=self._cam.azimuth, width=300,
                            callback=lambda s, v: setattr(self._cam, "azimuth", float(v)),
                        )
                        dpg.add_slider_float(
                            label="elevation", tag="cam_el",
                            min_value=-89.0, max_value=89.0,
                            default_value=self._cam.elevation, width=300,
                            callback=lambda s, v: setattr(self._cam, "elevation", float(v)),
                        )
                        dpg.add_slider_float(
                            label="distance", tag="cam_dist",
                            min_value=100.0, max_value=1500.0,
                            default_value=self._cam.distance, width=300,
                            callback=lambda s, v: setattr(self._cam, "distance", float(v)),
                        )

        dpg.create_viewport(
            title="Quadruped Teleop",
            width=1150 if self.view_mode == "embedded" else 580,
            height=660,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        self._open_backend(self.backend_kind)

        last_send = 0.0
        try:
            while dpg.is_dearpygui_running():
                t0 = time.time()

                mode = dpg.get_value("mode_combo")
                if mode == "joint":
                    q = {
                        joint.name: float(dpg.get_value(f"joint_{joint.name}"))
                        for joint in CONFIG.joints
                    }
                    # Push slider values into IK model so the preview reflects them.
                    for name, v in q.items():
                        self.ik_data.qpos[self._joint_qpos_idx[name]] = v
                else:
                    target = np.array(
                        [dpg.get_value("cart_x"), dpg.get_value("cart_y"), dpg.get_value("cart_z")],
                        dtype=float,
                    )
                    self.ik.set_mocap_target(target, self._target_default)
                    result = self.ik.solve(target)
                    q = {name: float(v) for name, v in zip(CONFIG.joint_names, result.q)}
                    for name, v in q.items():
                        dpg.set_value(f"joint_{name}", v)

                # Throttle outgoing commands (sim is cheap; hw is ~57600-baud bound).
                if t0 - last_send >= self.dt:
                    self._send(q)
                    last_send = t0

                if self.backend is not None:
                    qpos, _ = self.backend.read_joint_state()
                    for joint in CONFIG.joints:
                        v = qpos.get(joint.name)
                        dpg.set_value(
                            f"state_{joint.name}",
                            f"  {joint.name}: {v:.4f}" if v is not None else f"  {joint.name}: --",
                        )

                # Render embedded viewer at ~render_hz (cheaper than UI loop).
                if self.view_mode == "embedded" and (t0 - self._last_render) >= self.render_period:
                    self._render_to_texture()
                    self._last_render = t0

                dpg.render_dearpygui_frame()

                spare = self.dt - (time.time() - t0)
                if spare > 0:
                    time.sleep(spare)
        finally:
            self._close_backend()
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
            dpg.destroy_context()
        return 0


def run(
    *,
    backend: str = "sim",
    port: Optional[str] = None,
    view_mode: str = "embedded",
    hz: int = 100,
) -> int:
    return GuiApp(backend_kind=backend, port=port, view_mode=view_mode, hz=hz).run()
