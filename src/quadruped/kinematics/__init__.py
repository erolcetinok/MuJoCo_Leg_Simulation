"""Kinematics package — forward and inverse kinematics for the leg."""
from quadruped.kinematics.fk import foot_position
from quadruped.kinematics.ik import joint_angles

__all__ = ["foot_position", "joint_angles"]
