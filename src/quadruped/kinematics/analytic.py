"""Closed-form FK/IK for the 3-DoF leg (no MuJoCo dependency at runtime).

A 3-DoF serial chain (shoulder, wing, knee) has a tractable analytic
solution. Implementing it here buys you:

- ~1000× faster than DLS (no Jacobian inversion, no iteration loop)
- no MuJoCo import at runtime — works in environments where MuJoCo isn't installed
- a reference oracle to sanity-check numerical solvers against

Pulls link lengths from `CONFIG.links_mm` so the YAML stays the single
source of truth for geometry.
"""
