"""MuJoCo viewer — collapses the old view_leg / view_quadruped / test_leg.

Modes:
  --model single   single-leg model (default)
  --model quad     quadruped model
  --print-foot     print foot site position(s) every 0.1 s
"""
from __future__ import annotations

import argparse
import sys
import time

import mujoco
import mujoco.viewer

from quadruped.sim.env import model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["single", "quad"], default="single")
    parser.add_argument("--print-foot", action="store_true",
                        help="Stream foot_site x/y/z to stdout while viewer runs.")
    args = parser.parse_args()

    xml = model_path("single_leg" if args.model == "single" else "quadruped")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.model == "single":
        foot_site_ids = [(None, model.site("foot_site").id)]
    else:
        foot_site_ids = [(leg, model.site(f"foot_site_{leg}").id) for leg in ("FL", "FR", "BL", "BR")]

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            if args.print_foot and data.time - last_print > 0.1:
                parts = []
                for tag, sid in foot_site_ids:
                    x, y, z = data.site_xpos[sid]
                    label = f"{tag}=" if tag else ""
                    parts.append(f"{label}({x:.2f},{y:.2f},{z:.2f})")
                print(f"t={data.time:.2f}  ctrl={list(data.ctrl[:])}  foot={' '.join(parts)}")
                last_print = data.time
            viewer.sync()
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    sys.exit(main())
