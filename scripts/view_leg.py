import mujoco
import mujoco.viewer

# Load the model (path is relative to this script)
model = mujoco.MjModel.from_xml_path("../meshes/single_leg.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data):
    while True:
        mujoco.mj_step(model, data)
