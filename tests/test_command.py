"""Teleop command shaping: stepped setpoints, slew limiting, idle decay, key map.

All of this runs without a terminal — `apply_key` is a pure function over the
shaper, which is why the key map lives in the package and `scripts/teleop.py`
only does I/O.
"""
from __future__ import annotations

import math

import pytest

from quadruped.control.command import (
    VELOCITY_AXES,
    CommandShaper,
    ShapedAxis,
    default_axes,
)


# --- ShapedAxis ---------------------------------------------------------------

def test_tap_moves_the_setpoint_by_one_step_not_the_actual():
    axis = ShapedAxis(step=10.0, slew=100.0, lo=-120.0, hi=120.0)
    axis.nudge(+1)
    assert axis.setpoint == pytest.approx(10.0)
    assert axis.actual == 0.0, "actual must ramp, never jump"


def test_actual_ramps_at_the_slew_rate():
    axis = ShapedAxis(step=50.0, slew=100.0, lo=-120.0, hi=120.0)
    axis.nudge(+1)                      # setpoint 50
    axis.update(0.1)                    # 100 mm/s^2 * 0.1 s = 10
    assert axis.actual == pytest.approx(10.0)
    axis.update(0.1)
    assert axis.actual == pytest.approx(20.0)


def test_actual_never_overshoots_the_setpoint():
    axis = ShapedAxis(step=5.0, slew=1000.0, lo=-120.0, hi=120.0)
    axis.nudge(+1)
    axis.update(1.0)                    # slew would allow 1000, setpoint is 5
    assert axis.actual == pytest.approx(5.0)


def test_setpoint_is_clamped_to_the_axis_limits():
    axis = ShapedAxis(step=100.0, slew=100.0, lo=-120.0, hi=120.0)
    for _ in range(10):
        axis.nudge(+1)
    assert axis.setpoint == pytest.approx(120.0)


def test_ramping_down_works_the_same_way():
    axis = ShapedAxis(step=10.0, slew=100.0, lo=-120.0, hi=120.0, setpoint=0.0, actual=50.0)
    axis.update(0.1)
    assert axis.actual == pytest.approx(40.0)


def test_hard_zero_bypasses_the_slew_limit():
    axis = ShapedAxis(step=10.0, slew=1.0, lo=-120.0, hi=120.0, setpoint=90.0, actual=90.0)
    axis.hard_zero()
    assert (axis.setpoint, axis.actual) == (0.0, 0.0)


# --- decay --------------------------------------------------------------------

def test_velocity_decays_to_zero_once_idle():
    s = CommandShaper(default_axes(v_step=40.0, v_slew=1000.0), decay_after=0.3)
    s.key("w")
    s.note_input()
    s.update(0.02)
    assert s.axes["vx"].actual > 0

    for _ in range(30):                 # 0.6 s with no key
        s.update(0.02)
    assert s.axes["vx"].setpoint == 0.0
    assert s.axes["vx"].actual == pytest.approx(0.0)


def test_pose_does_not_decay_when_idle():
    """A commanded attitude should persist; only velocity is deadman-style."""
    s = CommandShaper(default_axes(), decay_after=0.1)
    s.key("UP")
    held = s.axes["pitch"].setpoint
    for _ in range(50):
        s.update(0.02)
    assert s.axes["pitch"].setpoint == pytest.approx(held)


def test_holding_a_key_keeps_it_alive():
    s = CommandShaper(default_axes(v_step=10.0), decay_after=0.3)
    for _ in range(30):                 # key repeat every tick
        s.key("w")
        s.update(0.02)
    assert s.axes["vx"].setpoint > 0


# --- key map ------------------------------------------------------------------

@pytest.mark.parametrize(
    "key,axis,sign",
    [
        ("w", "vx", +1), ("s", "vx", -1),
        ("a", "vy", +1), ("d", "vy", -1),
        ("q", "yaw_rate", +1), ("e", "yaw_rate", -1),
        ("r", "height", +1),
        (",", "body_yaw", +1), (".", "body_yaw", -1),
        ("RIGHT", "roll", +1), ("LEFT", "roll", -1),
    ],
)
def test_movement_keys_move_the_right_axis(key, axis, sign):
    s = CommandShaper(default_axes())
    s.key(key)
    assert math.copysign(1.0, s.axes[axis].setpoint) == sign
    assert s.axes[axis].setpoint != 0.0


def test_height_lowers_back_toward_nominal_but_never_crouches():
    """Nominal stance is the floor — crouching would exceed the wing joint."""
    s = CommandShaper(default_axes())
    for _ in range(3):
        s.key("r")
    raised = s.axes["height"].setpoint
    assert raised > 0
    s.key("f")
    assert s.axes["height"].setpoint < raised
    for _ in range(20):
        s.key("f")
    assert s.axes["height"].setpoint == pytest.approx(0.0)


def test_up_arrow_raises_the_nose():
    """UI is nose-up-positive; BodyPose.pitch is nose-down-positive."""
    s = CommandShaper(default_axes())
    s.key("UP")
    assert s.axes["pitch"].setpoint < 0
    s2 = CommandShaper(default_axes())
    s2.key("DOWN")
    assert s2.axes["pitch"].setpoint > 0


def test_gait_keys_switch_gait():
    s = CommandShaper(default_axes(), gait="walk")
    s.key("2")
    assert s.gait == "trot"
    s.key("1")
    assert s.gait == "walk"


def test_space_zeroes_every_setpoint():
    s = CommandShaper(default_axes())
    for k in ("w", "a", "q", "UP", "r"):
        s.key(k)
    s.key(" ")
    assert all(ax.setpoint == 0.0 for ax in s.axes.values())


def test_estop_zeroes_immediately_and_latches_until_a_move_key():
    s = CommandShaper(default_axes(v_slew=1.0))
    s.key("w")
    s.update(1.0)
    s.key("x")
    assert s.estopped
    assert all(ax.actual == 0.0 for ax in s.axes.values())
    s.key("w")
    assert not s.estopped, "driving again should clear the e-stop"


@pytest.mark.parametrize("key", ["\x03", "\x1b", "ESC"])
def test_quit_keys_report_quit(key):
    assert CommandShaper(default_axes()).key(key) is False


def test_other_keys_do_not_quit():
    s = CommandShaper(default_axes())
    assert s.key("w") is True
    assert s.key("?") is True


def test_step_and_slew_are_tunable_live():
    s = CommandShaper(default_axes(v_step=10.0, v_slew=100.0))
    s.key("]")
    assert s.axes["vx"].step > 10.0
    s.key("[")
    s.key("[")
    assert s.axes["vx"].step < 10.0

    s.key("'")
    assert s.axes["vx"].slew > 100.0
    s.key(";")
    s.key(";")
    assert s.axes["vx"].slew < 100.0


def test_yaw_step_and_slew_have_their_own_keys():
    s = CommandShaper(default_axes(yaw_step=0.1, yaw_slew=0.5))
    s.key(">")
    assert s.axes["yaw_rate"].step > 0.1
    s.key('"')
    assert s.axes["yaw_rate"].slew > 0.5


# --- command assembly ---------------------------------------------------------

def test_update_returns_a_clamped_command():
    s = CommandShaper(default_axes())
    for _ in range(200):                # push pose axes hard against their stops
        s.key("UP")
        s.key("r")
    cmd = s.update(1.0)
    assert cmd.pose == cmd.pose.clamped()
    assert cmd.gait == "walk"


def test_velocity_axes_are_the_decaying_ones():
    assert set(VELOCITY_AXES) == {"vx", "vy", "yaw_rate"}
