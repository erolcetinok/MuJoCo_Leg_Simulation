"""Forward kinematics — compute foot position (and orientation) from joint angles.

Implement here. `tests/test_kinematics.py` expects something importable from
this module; re-export your public API in `kinematics/__init__.py`.
"""
import numpy as np


# so what we can do is utilize the limb length of the robot
def rot_x(theta):
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array([[1,0,0],[0,cos,-sin],[0,sin,cos]])

def rot_z(theta):
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])
