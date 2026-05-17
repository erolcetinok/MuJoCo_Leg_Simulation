"""IK→FK round-trip: pick random foot targets in the workspace, solve IK,
forward-evaluate the solution, assert the foot landed near the target.

Targets that fall outside the leg's reachable workspace are skipped on a
per-trial basis (we count converged trials, require a healthy fraction).
"""
from __future__ import annotations

import numpy as np
import pytest

from quadruped.config import CONFIG
from quadruped.kinematics.ik import DampedLeastSquaresIK
from quadruped.kinematics.fk import foot_position
from quadruped.sim.env import load_model


@pytest.fixture(scope="module")
def model_data():
    return load_model()


def test_ik_fk_roundtrip(model_data):
    model, data = model_data
    ik = DampedLeastSquaresIK(
        model, data,
        foot_site_name=CONFIG.mjcf.foot_site_name,
        target_site_name=CONFIG.mjcf.target_site_name,
        joint_names=CONFIG.mjcf.joint_names,
    )

    # Sample inside the known-reachable shell around the natural foot offset.
    # The natural target_site_offset_mm is (20, -175, -50). Sample within a
    # small box around it so most trials converge.
    rng = np.random.default_rng(seed=42)
    n_trials = 50
    base = np.asarray(CONFIG.target_site_offset_mm, dtype=float)
    site_offset = base.copy()

    converged = 0
    errors_mm = []
    for _ in range(n_trials):
        # Box around the natural offset
        delta = rng.uniform(low=[-25, -15, -25], high=[25, 15, 25], size=3)
        target = base + delta

        # Reset qpos so each trial starts from zero (matches CLI behavior)
        data.qpos[:] = 0.0
        ik.set_mocap_target(target, site_offset)
        result = ik.solve(target)

        if result.converged:
            converged += 1
            achieved = foot_position(
                model, data, result.q,
                foot_site_name=CONFIG.mjcf.foot_site_name,
                joint_names=CONFIG.mjcf.joint_names,
            )
            errors_mm.append(float(np.linalg.norm(achieved - target)))

    assert converged >= n_trials * 0.9, (
        f"only {converged}/{n_trials} IK solves converged"
    )
    assert max(errors_mm) < 1.0, (
        f"FK(IK(target)) drift max={max(errors_mm):.3f} mm exceeds 1 mm tolerance"
    )
