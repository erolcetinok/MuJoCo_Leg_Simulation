"""Tests for the pure calibration helpers (goal_ticks + patch_robot_yaml)."""
import math

import pytest

from quadruped.calibration import goal_ticks, patch_robot_yaml

SAMPLE = (
    "joints:\n"
    "  - {name: shoulder_FL, leg: FL, joint: shoulder, motor_id: 1,  "
    "limit_rad: [-1.57,  1.57], offset_deg: 180.0, direction:  1}\n"
    "  - {name: shoulder_FR, leg: FR, joint: shoulder, motor_id: 4,  "
    "limit_rad: [-1.57,  1.57], offset_deg: 180.0, direction:  1}\n"
    "  - {name: wing_FR, leg: FR, joint: wing, motor_id: 5,  "
    "limit_rad: [-0.87, 0.87], offset_deg: 180.0, direction:  1}\n"
)


# --- goal_ticks -------------------------------------------------------------

def test_goal_ticks_zero_is_offset():
    # at q=0 the direction term drops out → ticks = offset * 4096/360
    assert goal_ticks(1, 180.0, 0.0) == 2048
    assert goal_ticks(-1, 180.0, 0.0) == 2048


def test_goal_ticks_direction_flips_sense():
    up = goal_ticks(1, 180.0, math.radians(30))
    dn = goal_ticks(-1, 180.0, math.radians(30))
    assert up > 2048 and dn < 2048
    assert (up - 2048) == (2048 - dn)  # symmetric about centre


def test_goal_ticks_clamps_to_servo_range():
    assert goal_ticks(1, 180.0, math.radians(200)) == 4095   # would exceed
    assert goal_ticks(1, 180.0, math.radians(-200)) == 0


# --- patch_robot_yaml -------------------------------------------------------

def test_patch_changes_only_target_joint():
    out = patch_robot_yaml(SAMPLE, {"shoulder_FR": (-1, 173.2)})
    lines = {ln.split("name: ")[1].split(",")[0]: ln
             for ln in out.splitlines() if "name:" in ln}
    # target updated
    assert "direction: -1" in lines["shoulder_FR"]
    assert "offset_deg: 173.2" in lines["shoulder_FR"]
    # neighbours untouched
    assert "direction:  1" in lines["shoulder_FL"] and "offset_deg: 180.0" in lines["shoulder_FL"]
    assert "direction:  1" in lines["wing_FR"]
    # unrelated fields on the edited line preserved
    assert "motor_id: 4" in lines["shoulder_FR"]
    assert "limit_rad: [-1.57,  1.57]" in lines["shoulder_FR"]


def test_patch_multiple_joints():
    out = patch_robot_yaml(SAMPLE, {"shoulder_FR": (-1, 10.0), "wing_FR": (1, 200.5)})
    assert "offset_deg: 10.0" in out and "offset_deg: 200.5" in out


def test_patch_unknown_joint_raises():
    with pytest.raises(KeyError):
        patch_robot_yaml(SAMPLE, {"nope_FR": (1, 180.0)})


def test_patch_real_robot_yaml_stays_parseable():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "robot.yaml").read_text()
    out = patch_robot_yaml(text, {"shoulder_FR": (-1, 173.2)})
    doc = yaml.safe_load(out)
    fr = next(j for j in doc["joints"] if j["name"] == "shoulder_FR")
    assert fr["direction"] == -1 and fr["offset_deg"] == 173.2
    # an untouched joint is unchanged
    fl = next(j for j in doc["joints"] if j["name"] == "shoulder_FL")
    assert fl["direction"] == 1 and fl["offset_deg"] == 180.0
