"""Forward kinematics - foot position from joint angles"""

import numpy as np
from quadruped.config import CONFIG

L1 = np.array(CONFIG.links_mm.shoulder_to_wing)
L2 = np.array(CONFIG.links_mm.wing_to_knee)
L3 = np.array(CONFIG.links_mm.knee_to_foot)


def rot_x(theta: float) -> np.ndarray:
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],[0, cos, -sin],[0, sin,  cos]])


def rot_z(theta: float) -> np.ndarray:
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, 0],[sin,  cos, 0],[0,  0, 1]])


def foot_position(shoulder: float, wing: float, knee: float) -> np.ndarray:
    return rot_z(shoulder) @ (L1 + rot_x(wing) @ (L2 + rot_x(knee) @ L3))
