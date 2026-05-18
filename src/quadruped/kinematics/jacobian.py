"""Jacobian computation for the leg.

The Jacobian relates joint velocities to end-effector velocities and is the
core building block for both velocity IK (via J^+) and damped least-squares
(via J^T (J J^T + λI)^-1). Living home for either:

- numerical Jacobian via MuJoCo's `mj_jacSite`, or
- analytic Jacobian derived from link lengths in `CONFIG.links_mm`.

Keeping this separate from `ik.py` lets multiple solvers reuse it.
"""
