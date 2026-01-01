"""
Simple viewer for the quadruped model.
Allows you to manually control joints via sliders.
"""

import os
import time
import mujoco
import mujoco.viewer

# Get the project root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the quadruped XML model
XML_PATH = os.path.join(ROOT, "meshes", "quadruped.xml")

# Load the model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Get foot site IDs
foot_site_FL_id = model.site("foot_site_FL").id
# foot_site_FR_id = model.site("foot_site_FR").id  # Commented out - front right leg disabled
foot_site_BL_id = model.site("foot_site_BL").id
foot_site_BR_id = model.site("foot_site_BR").id

# Initialize forward kinematics
mujoco.mj_forward(model, data)

# Launch viewer with joint sliders
last_print_time = time.time()
print_interval = 0.5  # Print every 0.5 seconds

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # For position actuators, copy control values to joint positions
        for i in range(model.nu):
            jid = model.actuator(i).trnid[0]
            data.qpos[jid] = data.ctrl[i]
        
        # Update kinematics
        mujoco.mj_forward(model, data)
        
        viewer.sync()

