"""Reachability and workspace analysis.

Useful for:
- pre-flight check before solving IK (is the target in-bounds at all?)
- generating random sample points inside the reachable shell (for tests)
- visualizing the reachable volume
- giving sane defaults for Cartesian slider ranges in `gui/app.py`

The current code catches unreachable targets only by IK failing to converge,
which is wasteful and silent. A real `is_reachable(target_mm) -> bool` would
short-circuit that.
"""
