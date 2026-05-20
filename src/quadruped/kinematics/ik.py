"""Inverse kinematics — solve joint angles for a target foot position.

Implement here. Downstream code that imports from this module
(`cli/send_foot.py`, `cli/ik_demo.py`, `gui/app.py`, `tests/test_kinematics.py`)
will `ImportError` until you re-add your public symbols and re-export them in
`kinematics/__init__.py`.
"""

import numpy as np
from quadrupued.kinematics.fk import L1, L2, L3

def joint_angles (x: float, y: float, z: float) -> [float,float,float]:

