"""Phase 4 home — gait state machine.

Seeded from scripts/ik_quadruped.py — these helpers are what the existing
quadruped gait demo uses. Phase 4 will grow them into a real stance/swing
state machine, but the math here already produces walk/trot offsets.
"""
from __future__ import annotations

import numpy as np


def compute_gait_offset(leg: str, leg_phase: float, params: dict) -> np.ndarray:
    """Foot position offset (mm) for a leg at a given gait phase.

    Args:
        leg: leg name (currently unused — kept for future per-leg shaping).
        leg_phase: 0..2π. <π is swing, ≥π is stance.
        params: dict with ``step_length``, ``step_height``, ``step_frequency``.
    """
    diag_component = params["step_length"] / np.sqrt(2.0)

    if leg_phase < np.pi:
        swing_progress = leg_phase / np.pi
        diag_offset = diag_component * (swing_progress - 0.5)
        z_offset = params["step_height"] * np.sin(swing_progress * np.pi)
    else:
        stance_progress = (leg_phase - np.pi) / np.pi
        diag_offset = diag_component * (0.5 - stance_progress)
        z_offset = 0.0

    return np.array([diag_offset, diag_offset, z_offset])


WALK_PARAMS = {
    "step_length": 30.0,
    "step_height": 15.0,
    "step_frequency": 0.6,
}

TROT_PARAMS = {
    "step_length": 30.0,
    "step_height": 15.0,
    "step_frequency": 0.8,
}

# Per-leg phase offsets for the two canonical 4-leg gaits.
WALK_PHASE_OFFSETS = {
    "FL": 0.0,
    "FR": np.pi / 2,
    "BL": np.pi,
    "BR": 3 * np.pi / 2,
}

TROT_PHASE_OFFSETS = {
    "FL": 0.0,
    "BR": 0.0,
    "FR": np.pi,
    "BL": np.pi,
}
