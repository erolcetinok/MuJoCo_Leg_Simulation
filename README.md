# MuJoCo Quadruped

A 3-DoF quadruped robot project with a tight simulation ↔ hardware parallel:
MuJoCo simulation, Dynamixel-driven physical leg, and a backend abstraction so
the same command stream can drive either or both in lockstep.

Current state: single physical leg working (Arduino UNO R3 + Dynamixel Shield +
3× XL430). Goal: walking quadruped over the summer, progressing through
kinematics → trajectories → PID → gait → 4-leg coordination → IMU → ROS 2.

> **New to the codebase?** Read **`docs/HANDBOOK.md`** — it's the working
> guide to the project structure, phase-by-phase recipes, and the "I want
> to…" cheat-sheet. This README is the quickstart; the handbook is the manual.

---

## Layout

```
configs/robot.yaml             single source of truth (joints, limits, offsets, bauds)
description/                   MJCF + STL meshes (MuJoCo-Menagerie convention)
firmware/                      Arduino sketches; robot_config.h is codegen'd
src/quadruped/                 Python package
  ├── config.py                generated dataclass mirroring robot.yaml
  ├── kinematics/              IK (damped least-squares) + FK
  ├── control/                 trajectory / PID / gait (Phase 2-4 homes)
  ├── backends/                RobotBackend ABC; Mujoco, Arduino, Mirror impls
  ├── sim/                     MuJoCo loader with YAML-consistency assertions
  ├── gui/                     Dear PyGui slider app (Landing 2 — stub)
  └── cli/                     entry-point modules: send_foot, jog, view, ik_demo…
scripts/                       thin shims + codegen.py
tests/                         pytest: codegen drift + IK→FK round-trip
cad/                           CAD source (e.g. thigh.3mf)
```

## Quickstart

```bash
# Install the package + tooling (creates `quad-*` CLI entry points)
pip install -e .[dev,gui]

# Verify everything works
python scripts/codegen.py --check     # generated artifacts match robot.yaml
pytest                                # IK round-trip + codegen tests

# Simulation only
python scripts/send_foot.py 20 -175 -50 --dry-run     # IK only, prints angles
python scripts/send_foot.py 20 -175 -50 --backend sim --viewer
python scripts/view.py --model single
python scripts/ik_demo.py --path step
python scripts/gait_demo.py --gait trot

# Hardware (needs the leg powered + USB-to-TTL adapter — see firmware/README.md)
export SERIAL_PORT=/dev/cu.usbserial-XXXX
python scripts/send_foot.py 20 -175 -50 --backend hw
python scripts/jog.py --backend hw

# Sim + hardware in lockstep
python scripts/send_foot.py 20 -175 -50 --backend mirror --viewer

# Interactive slider GUI (Dear PyGui + embedded MuJoCo render)
python scripts/gui.py --backend sim                       # default: embedded sim view in the GUI
python scripts/gui.py --backend mirror --viewer embedded  # GUI drives sim + physical leg
python scripts/gui.py --backend hw --viewer none          # slider-only, no sim view
python scripts/gui.py --backend sim --viewer external     # fall back to mujoco passive viewer
```

> **GUI viewer modes:** `embedded` (default) renders MuJoCo offscreen into a
> Dear PyGui dynamic texture — one window, one process, no second GLFW
> context, works reliably on macOS. `external` launches MuJoCo's full
> passive viewer in a separate window (useful for mouse-camera control), but
> on macOS the dual-GLFW setup is fragile. `none` is slider-only.

## Editing the robot configuration

Joint limits, motor IDs, baud rates, zero offsets, and link lengths live in
**`configs/robot.yaml`**. Two artifacts are generated from it and committed:

- `firmware/leg_controller/robot_config.h` — `#include`d by the sketch
- `src/quadruped/config.py` — frozen dataclass imported by the package

After editing the YAML, run `python scripts/codegen.py` (or `quad-codegen`
after install). A `--check` mode is included for pre-commit / CI: it fails if
either artifact has drifted.

To make drift a build error, install the bundled pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

After that every `git commit` runs `scripts/codegen.py --check` and refuses
to commit when the YAML and generated files don't match.

## Backends

`quadruped.backends.RobotBackend` is the abstraction. Three implementations
ship today:

| Backend             | What it does                                          |
| ------------------- | ----------------------------------------------------- |
| `MujocoBackend`     | Drives `data.ctrl` on the MJCF model; optional viewer |
| `ArduinoBackend`    | Serial bridge to `leg_controller.ino` (same wire format) |
| `MirrorBackend`     | Fans the same target to multiple backends             |
| `DynamixelBackend`  | Stub for a future U2D2 + DynamixelSDK direct path     |

## Phase roadmap

1. **Kinematics foundation** — IK exists (✓), refactor into the package (✓)
2. **Smooth motion / trajectories** — `control/trajectory.py` (cubic splines + Bezier swing curves)
3. **PID / feedback control** — `control/pid.py`
4. **Single-leg gait state machine** — `control/gait.py`
5. **4-leg coordination** — wire up `quadruped.xml` in real hardware
6. **Body kinematics & stability**
7. **IMU + state estimation**
8. **ROS 2 integration** — thin wrappers around the library
9. **Advanced locomotion** — trot transitions, terrain
10. **RL / MPC / vision**

The package is intentionally ROS-free so Phase 8 is a thin `ament_python`
wrapper rather than a rewrite.

## Hardware

Full bring-up (parts, wiring, motor configuration, calibration) is in
**`firmware/README.md`**.
