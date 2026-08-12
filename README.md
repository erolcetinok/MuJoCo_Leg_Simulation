# MuJoCo Quadruped

A 3-DoF-per-leg quadruped with a tight simulation ↔ hardware parallel: one
MuJoCo model, one set of kinematics and gait code, and a backend abstraction so
the same command stream drives the simulator, real Dynamixels, or both at once.

**Current state:** the full 12-DOF software stack is written and verified in
simulation. One physical leg has been driven end-to-end on real hardware. The
four-legged robot has not been assembled yet — that's the next job.

> **Coming back to this after a while?** Read **`docs/STATUS.md`** first — it's
> the dated catch-up page (what works, what's only ever run in sim, what's next).
> Then **`docs/HANDBOOK.md`** for how the code is organised.

---

## Layout

```
configs/robot.yaml      single source of truth (joints, limits, offsets, bauds)
description/            MJCF + STL meshes (MuJoCo-Menagerie convention)
firmware/               Arduino sketches; robot_config.h is generated
scripts/                every command you run, plus codegen
src/quadruped/          the importable library
  ├── config.py         generated dataclass mirroring robot.yaml
  ├── kinematics/       fk, ik (closed-form), jacobian (DLS IK + foot force)
  ├── control/          trajectory.py, gait.py
  ├── backends/         RobotBackend ABC: Mujoco, Arduino, Dynamixel, Mirror
  ├── calibration.py    tick math, YAML patching, the guided calibration routine
  ├── cli_args.py       argparse glue shared by scripts/ (backend flags, parsers)
  ├── sim/              MuJoCo loader with YAML-consistency assertions
  └── gui/              Dear PyGui app + embedded MuJoCo renderer
tests/                  pytest
cad/                    CAD source (e.g. thigh.3mf)
```

**`scripts/` holds things you run; `src/quadruped/` holds things you import.**
There are no `quad-*` console scripts — a command is a file, and you run it with
your active interpreter, which is what you want when MuJoCo is involved.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,gui]'      # installs deps + `import quadruped`; no commands
pytest                           # or: PYTHONPATH=src pytest
python scripts/codegen.py --check   # generated artifacts match robot.yaml?
```

```bash
# Simulation only — safe, no hardware needed
python scripts/send_foot.py 20 -175 -50 --dry-run       # IK only, prints angles
python scripts/send_foot.py 20 -175 -50 --backend sim --viewer
python scripts/view.py --model quad
python scripts/ik_demo.py --path step
python scripts/gait_demo.py --gait trot --vx 40

# Hardware — U2D2 path (primary)
export SERIAL_PORT=/dev/cu.usbserial-XXXX
python scripts/calibrate.py --leg FR                    # guided sign/offset wizard
python scripts/jog_cart.py --leg FL                     # Cartesian jog + foot force
python scripts/gait_demo.py --backend dxl --rate 33

# Hardware — Arduino UNO bridge (fallback; see firmware/README.md)
python scripts/jog.py --backend hw
python scripts/swing_hw.py --loop

# Sim + hardware in lockstep
python scripts/send_foot.py 20 -175 -50 --backend mirror --viewer

# Slider GUI (Dear PyGui + embedded MuJoCo render)
python scripts/gui.py                                   # sim, embedded viewer
python scripts/gui.py --backend mirror                  # sim + leg together
python scripts/gui.py --backend hw --viewer none        # sliders only
```

> **GUI viewer modes:** `embedded` (default) renders MuJoCo offscreen into a Dear
> PyGui texture — one window, one process, no second GLFW context, reliable on
> macOS. `external` launches MuJoCo's passive viewer in its own window (mouse
> camera control, but the dual-GLFW setup is fragile on macOS). `none` is
> slider-only.

## Backends

`quadruped.backends.RobotBackend` is the abstraction; pick one with `--backend`
and nothing downstream branches on the choice.

| `--backend` | Class | What it does |
| --- | --- | --- |
| `sim` | `MujocoBackend` | Writes joint angles straight to `data.qpos` and runs `mj_forward`. **Kinematic — physics is bypassed**, so it shows geometry and reach, not balance. |
| `dxl` | `DynamixelBackend` | U2D2 + DYNAMIXEL SDK, direct USB→bus. Sync read/write, present current, `foot_force(leg)`. Written and unit-tested; **not yet run against a real bus.** |
| `hw` | `ArduinoBackend` | Serial bridge to `leg_controller.ino` at 57600. Fire-and-forget: `SoftwareSerial` drops ~15% of host bytes under DXL load. |
| `mirror` | `MirrorBackend` | Fans one command to sim + hardware together. |

## Editing the robot configuration

Joint limits, motor IDs, baud rates, zero offsets, and link lengths live in
**`configs/robot.yaml`**. Two artifacts are generated from it and committed:

- `firmware/leg_controller/robot_config.h` — `#include`d by the sketch
- `src/quadruped/config.py` — frozen dataclass imported by the package

After editing the YAML, run `python scripts/codegen.py`. `--check` mode fails if
either artifact has drifted; install the bundled hook to make drift a commit
error:

```bash
pip install pre-commit && pre-commit install
```

## Documentation

`docs/` is gitignored — it lives on this machine only, not in the repo.

| File | What it's for |
| --- | --- |
| `docs/STATUS.md` | **Start here.** Dated state of the project, what's hardware-proven vs sim-only, open gaps, next action. |
| `docs/HANDBOOK.md` | How the codebase is organised and how to work in it: mental model, repo map, the two pipelines, "I want to…" recipes, debugging playbook. |
| `docs/CLI.md` | Every command, every flag, when to reach for it. |
| `docs/INVERSE_KINEMATICS.md` | Full derivation of the closed-form solver, plus the Jacobian / DLS / foot-force math. |
| `docs/hardware_bringup.md` | BOM and bringup checklist for the full quadruped. |
| `docs/power_and_electronics.md` | Staged power chain, wiring diagrams, part choices. |
| `firmware/README.md` | Arduino UNO bridge path (fallback), end to end. |

## Hardware

The primary path is a Raspberry Pi 5 + U2D2 driving twelve XL430-W250 servos —
see `docs/hardware_bringup.md` and `docs/power_and_electronics.md`. The Arduino
UNO R3 + Dynamixel Shield bridge is kept as a fallback and documented in
`firmware/README.md`.
