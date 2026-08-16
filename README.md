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
firmware/               Arduino sketches (ARCHIVED); robot_config.h is generated
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
pip install -e '.[dev,gui]'      # for the dependencies; commands self-bootstrap
pytest
python scripts/codegen.py --check   # generated artifacts match robot.yaml?
```

> **macOS: windowed commands need `mjpython`, not `python`.** MuJoCo's
> interactive viewer must own the main thread. That means `view.py`, `ik_demo.py`,
> anything with `--viewer`, and `gui.py --viewer external`. Everything else —
> including plain `python scripts/gui.py`, whose embedded viewer renders
> offscreen — runs under normal `python`. `mjpython` ships with the mujoco wheel.

```bash
# Simulation only — safe, no hardware needed
python scripts/send_foot.py 20 -175 -50 --dry-run       # IK only, prints angles
mjpython scripts/send_foot.py 20 -175 -50 --backend sim --viewer
mjpython scripts/view.py --model quad
mjpython scripts/ik_demo.py --path step
python scripts/gait_demo.py --gait trot --vx 40

# Hardware — U2D2, in bringup order (see docs/hardware_bringup.md)
export SERIAL_PORT=/dev/ttyUSB0
python scripts/dxl_scan.py                              # do all 12 enumerate?
python scripts/calibrate.py --leg FL                    # guided sign/offset wizard
python scripts/calibrate.py --verify --leg FL           # did it take?
python scripts/swing_hw.py --leg FL --backend dxl --rate 33 --loop
python scripts/gait_demo.py --backend mirror --rate 33

# Sim + hardware in lockstep
mjpython scripts/send_foot.py 20 -175 -50 --backend mirror --viewer

# Slider GUI (Dear PyGui + embedded MuJoCo render)
python scripts/gui.py                                   # sim, embedded viewer
python scripts/gui.py --backend mirror                  # sim + leg together
python scripts/gui.py --backend dxl --viewer none       # sliders only
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
| `dxl` | `DynamixelBackend` | U2D2 + DYNAMIXEL SDK, direct USB→bus. Pings all 12 on connect and checks every SDK return code. Sync read/write, present current, `foot_force(leg)`, health status. Written and unit-tested; **not yet run against a real bus.** |
| `mirror` | `MirrorBackend` | Fans one command to sim + the U2D2 together. |

`ArduinoBackend` (the UNO/SoftwareSerial bridge that brought up the first leg) is
archived: still importable, no longer reachable from a `--backend` string, and
none of the safety machinery above applies to it. See `firmware/README.md`.

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
| `docs/hardware_bringup.md` | **The build doc.** Components, wiring diagram, gauge per connection, bringup runbook. |
| `docs/power_rationale.md` | Why those parts, what was cut, research sources. |
| `firmware/README.md` | Arduino UNO bridge path (ARCHIVED), end to end. |

## Hardware

The hardware is a Raspberry Pi 5 + U2D2 driving twelve XL430-W250 servos.
`docs/hardware_bringup.md` has the components, the wiring diagram with a gauge per
connection, and the ordered bringup runbook. That stack is final; all remaining
work on this project is software. The Arduino UNO R3 + Dynamixel Shield bridge
that brought up the first leg is archived in `firmware/README.md`.
