# MuJoCo Quadruped Simulation

MuJoCo kinematic modeling and inverse kinematics practice for a **3 DoF quadruped robot**, practice for later physical implementation with Dynamixel actuators.

---

## Overview

This repository is used to:
- validate CAD and test joint alignment
- Develop and test **Jacobian-based IK** for 3 DoF leg
- Prototype basic gaits (walk & trot)
- Practice for transition to hardware
- focus is on **kinematics and control logic**

---

## Features

- Single leg and full quadruped MuJoCo models
- Damped least squares IK
- Parametric walk and trot gait
- Millimeter-scale geometry

---

## Tech Stack

- **Simulation:** MuJoCo  
- **Language:** Python  
- **Math:** NumPy
- **CAD:** SolidWorks (rough STL exports for visualization)
