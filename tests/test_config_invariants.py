"""Structural invariants over configs/robot.yaml (via the generated CONFIG).

`robot.yaml` lists all twelve joints explicitly, so the four legs agree only by
hand. A typo in one leg's `limit_rad` is otherwise silent: it survives codegen,
survives the MJCF assertions (which compare per-joint), and shows up only as
one leg clipping strangely mid-gait. These tests make that class of mistake
fail at `pytest` time instead.
"""
from __future__ import annotations

import pytest

from quadruped.config import CONFIG

JOINT_KINDS = ("shoulder", "wing", "knee")


def test_twelve_joints_four_legs():
    assert len(CONFIG.joints) == 12
    assert CONFIG.legs == ("FL", "FR", "BL", "BR")


def test_motor_ids_are_unique_and_cover_1_to_12():
    ids = sorted(j.motor_id for j in CONFIG.joints)
    assert ids == list(range(1, 13)), f"motor IDs must be exactly 1-12, got {ids}"


def test_motor_ids_are_contiguous_per_leg():
    """FL=1-3, FR=4-6, BL=7-9, BR=10-12 — the firmware and wire format assume it."""
    for i, leg in enumerate(CONFIG.legs):
        expected = list(range(1 + 3 * i, 4 + 3 * i))
        got = [j.motor_id for j in CONFIG.joints_for_leg(leg)]
        assert got == expected, f"{leg} should own IDs {expected}, has {got}"


def test_each_leg_has_shoulder_wing_knee_in_order():
    for leg in CONFIG.legs:
        kinds = tuple(j.joint for j in CONFIG.joints_for_leg(leg))
        assert kinds == JOINT_KINDS, f"{leg} joint order is {kinds}"


@pytest.mark.parametrize("kind", JOINT_KINDS)
def test_limits_identical_across_legs(kind):
    """The legs are mechanically identical; their limits must match exactly."""
    limits = {
        j.leg: j.limit_rad
        for j in CONFIG.joints
        if j.joint == kind
    }
    distinct = set(limits.values())
    assert len(distinct) == 1, f"{kind} limits differ across legs: {limits}"


def test_limits_are_ordered_and_nonzero():
    for j in CONFIG.joints:
        lo, hi = j.limit_rad
        assert lo < hi, f"{j.name}: limit_rad is not (min, max): {j.limit_rad}"


def test_directions_are_plus_or_minus_one():
    for j in CONFIG.joints:
        assert j.direction in (1, -1), f"{j.name}: direction={j.direction}"


def test_offsets_reachable_within_single_turn():
    """goal_deg = dir*q*180/pi + offset must stay inside the servo's 0-360 range.

    OP_POSITION is single-turn; a combination that lands outside gets clamped
    silently by the firmware, which is how the knee's negative range was lost
    once before.
    """
    import math

    for j in CONFIG.joints:
        for q in j.limit_rad:
            deg = j.direction * math.degrees(q) + j.offset_deg
            assert 0.0 <= deg <= 360.0, (
                f"{j.name}: q={q:.3f} rad -> {deg:.1f} deg, outside the servo's "
                f"single-turn range (offset_deg={j.offset_deg}, direction={j.direction})"
            )


def test_joint_names_match_leg_joint_names():
    flat = tuple(n for leg in CONFIG.legs for n in CONFIG.leg_joint_names(leg))
    assert flat == CONFIG.joint_names
