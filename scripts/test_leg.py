import os
import time
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "single_leg.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Initialize forward kinematics
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Update kinematics when joint positions change (from sliders)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)