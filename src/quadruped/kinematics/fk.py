"""Forward kinematics — compute foot position from joint angles.

Implement here. `tests/test_kinematics.py` expects something importable from
this module; re-export your public API in `kinematics/__init__.py`.
"""
import numpy as np
from quadruped.config import CONFIG


def foot_position(theta1, theta2, theta3):
    d1 = np.array(CONFIG.links_mm.shoulder_to_wing)
    d2 = np.array(CONFIG.links_mm.wing_to_knee)
    d3 = np.array(CONFIG.links_mm.knee_to_foot)
    
    foot = rot_z(theta1) @ (d1 + rot_x(theta2) @ (d2 + rot_x(theta3) @ d3))
    
    return foot
    

# rotation matrices x and z
def rot_x(theta):
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array([[1,0,0],[0,cos,-sin],[0,sin,cos]])

def rot_z(theta):
    sin = np.sin(theta)
    cos = np.cos(theta)
    return np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])
