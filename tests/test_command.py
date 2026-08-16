"""Teleop command shaping: stepped setpoints, slew limiting, idle decay, key map.

All of this runs without a terminal — `apply_key` is a pure function over the
shaper, which is why the key map lives in the package and `scripts/teleop.py`
only does I/O.
"""
from __future__ import annotations

import math

import pytest

from quadruped.control.command import (
    _NUDGES,
    _SLEW_KEYS,
    _STEP_KEYS,
    FINE_SCALE,
    VELOCITY_AXES,
    CommandShaper,
    ShapedAxis,
    decode_keys,
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

def test_setpoints_hold_by_default():
    """The default is 'set it and it stays' — no decay unless asked for."""
    s = CommandShaper(default_axes(v_step=40.0))
    assert not s.deadman
    s.key("w")
    held = s.axes["vx"].setpoint
    for _ in range(200):                # 4 s of nothing
        s.update(0.02)
    assert s.axes["vx"].setpoint == pytest.approx(held)
    assert s.axes["vx"].actual == pytest.approx(held)


def test_deadman_can_be_toggled_live():
    s = CommandShaper(default_axes(v_step=40.0, v_slew=1000.0))
    s.key("w")
    s.key("g")
    assert s.deadman
    for _ in range(40):
        s.update(0.02)
    assert s.axes["vx"].actual == pytest.approx(0.0)
    s.key("g")
    assert not s.deadman


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
    s.key("i")
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
        ("l", "roll", +1), ("j", "roll", -1),
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


def test_i_raises_the_nose():
    """UI is nose-up-positive; BodyPose.pitch is nose-down-positive."""
    s = CommandShaper(default_axes())
    s.key("i")
    assert s.axes["pitch"].setpoint < 0
    s2 = CommandShaper(default_axes())
    s2.key("k")
    assert s2.axes["pitch"].setpoint > 0


def test_gait_keys_switch_gait():
    s = CommandShaper(default_axes(), gait="walk")
    s.key("2")
    assert s.gait == "trot"
    s.key("1")
    assert s.gait == "walk"


def test_space_zeroes_every_setpoint():
    s = CommandShaper(default_axes())
    for k in ("w", "a", "q", "i", "r"):
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


def test_torque_kill_needs_an_estop_first():
    """Releasing torque drops the robot, so it takes x then z — never one key."""
    s = CommandShaper()
    s.key("z")
    assert not s.torque_kill, "z alone must not cut torque"
    s.key("x")
    s.key("z")
    assert s.torque_kill


def test_pitch_key_is_not_the_torque_kill():
    """`k` is pitch-down and always has been; the release key must not shadow it."""
    s = CommandShaper()
    s.key("x")
    s.key("k")
    assert not s.torque_kill
    assert s.axes["pitch"].setpoint > 0.0


@pytest.mark.parametrize("key", ["\x03", "\x1b", "ESC"])
def test_quit_keys_report_quit(key):
    assert CommandShaper(default_axes()).key(key) is False


def test_other_keys_do_not_quit():
    s = CommandShaper(default_axes())
    assert s.key("w") is True
    assert s.key("?") is True


# --- fixed steps, shift for fine ----------------------------------------------

def test_a_tap_always_moves_one_whole_step():
    """Steps are fixed, so the control can't change under you mid-drive."""
    s = CommandShaper(default_axes(v_step=10.0))
    for n in range(1, 5):
        s.key("w")
        assert s.axes["vx"].setpoint == pytest.approx(10.0 * n)
    assert s.axes["vx"].step == 10.0, "step must not drift"


def test_shift_gives_a_tenth_of_a_step():
    s = CommandShaper(default_axes(v_step=10.0))
    s.key("W")
    assert s.axes["vx"].setpoint == pytest.approx(1.0)
    s.key("W")
    assert s.axes["vx"].setpoint == pytest.approx(2.0)


@pytest.mark.parametrize("coarse,fine", [("w", "W"), ("s", "S"), ("a", "A"), ("d", "D"),
                                         ("q", "Q"), ("e", "E"), ("i", "I"), ("k", "K"),
                                         ("j", "J"), ("l", "L"), ("r", "R"), ("f", "F")])
def test_every_movement_key_has_a_shift_fine_variant(coarse, fine):
    a = CommandShaper(default_axes())
    b = CommandShaper(default_axes())
    a.key(coarse)
    b.key(fine)
    axis = _NUDGES[coarse][0]
    assert b.axes[axis].setpoint == pytest.approx(a.axes[axis].setpoint * FINE_SCALE)


def test_shift_steps_go_the_same_direction_as_coarse():
    s = CommandShaper(default_axes())
    s.key("S")
    assert s.axes["vx"].setpoint < 0, "shift changes size, never direction"


def test_height_steps_are_one_millimetre():
    """Round numbers in each axis's own unit — 1 mm, 1 degree."""
    s = CommandShaper(default_axes())
    s.key("r")
    assert s.axes["height"].setpoint == pytest.approx(1.0)
    s.key("R")
    assert s.axes["height"].setpoint == pytest.approx(1.1)


def test_attitude_steps_are_one_degree():
    s = CommandShaper(default_axes())
    s.key("k")                                   # nose down = +pitch
    assert math.degrees(s.axes["pitch"].setpoint) == pytest.approx(1.0)
    s.key("K")
    assert math.degrees(s.axes["pitch"].setpoint) == pytest.approx(1.1)


# --- terminal key decoding ----------------------------------------------------

def test_ordinary_keys_pass_straight_through():
    assert decode_keys("wasd") == ["w", "a", "s", "d"]


def test_lone_escape_means_quit():
    assert decode_keys("\x1b") == ["ESC"]


@pytest.mark.parametrize("seq", ["\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D"])
def test_arrow_keys_are_ignored_not_treated_as_quit(seq):
    """Arrows aren't bound, but they must not exit the program.

    A terminal sends Up as ESC [ A. Read naively, the leading ESC is the quit
    key — which is exactly what happened before: pressing an arrow killed
    teleop. Nothing is bound to arrows now, so they're swallowed whole.
    """
    assert decode_keys(seq) == []


def test_arrows_mixed_with_real_keys_lose_only_themselves():
    assert decode_keys("w\x1b[Aa\x1b[Cs") == ["w", "a", "s"]


def test_longer_csi_sequences_are_swallowed():
    """Function keys and mouse reports carry parameters before the final byte."""
    assert decode_keys("\x1b[1;5Aw") == ["w"]


def test_arrows_never_quit_through_the_shaper():
    s = CommandShaper(default_axes())
    for k in decode_keys("\x1b[A\x1b[D"):
        assert s.key(k) is not False
    assert all(ax.setpoint == 0.0 for ax in s.axes.values()), "arrows do nothing"


def test_the_key_map_is_only_what_a_driver_needs():
    """Guard against the control surface creeping back up.

    Every bound key must be a degree of freedom, a mode, or one of the two
    feel knobs. Body yaw and the per-axis tuning keys were deliberately cut.
    """
    s = CommandShaper(default_axes())
    bound = (set(_NUDGES) | {" ", "x", "g", "1", "2", "h"}
             | set(_STEP_KEYS) | set(_SLEW_KEYS))
    for dead in (",", ".", "<", ">"):
        assert dead not in bound
    assert set(_NUDGES) == {
        "w", "s", "a", "d", "q", "e", "i", "k", "j", "l", "r", "f"
    }, "one key pair per DOF: WASD drives, IJKL + R/F sets posture"
    assert not any(len(k) > 1 for k in _NUDGES), "single characters only — no escape sequences"
    assert "body_yaw" not in s.axes


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


# --- feel keys: additive, shift for a tenth ------------------------------------

def test_sensitivity_moves_by_exactly_one_per_press():
    """Additive, not multiplicative: the value you want is the press count."""
    s = CommandShaper(default_axes(v_step=10.0))
    s.key("]")
    assert s.axes["vx"].step == pytest.approx(11.0)
    s.key("]")
    assert s.axes["vx"].step == pytest.approx(12.0)
    s.key("[")
    assert s.axes["vx"].step == pytest.approx(11.0)


def test_shift_sensitivity_moves_by_a_tenth():
    s = CommandShaper(default_axes(v_step=10.0))
    s.key("}")
    assert s.axes["vx"].step == pytest.approx(10.1)
    s.key("{")
    assert s.axes["vx"].step == pytest.approx(10.0)


def test_smoothing_moves_by_exactly_one_per_press():
    s = CommandShaper(default_axes(v_slew=100.0))
    s.key("'")
    assert s.axes["vx"].slew == pytest.approx(101.0)
    s.key(";")
    assert s.axes["vx"].slew == pytest.approx(100.0)


def test_shift_smoothing_moves_by_a_tenth():
    s = CommandShaper(default_axes(v_slew=100.0))
    s.key('"')
    assert s.axes["vx"].slew == pytest.approx(100.1)
    s.key(":")
    assert s.axes["vx"].slew == pytest.approx(100.0)


def test_both_drive_axes_stay_in_step():
    s = CommandShaper(default_axes(v_step=10.0, v_slew=100.0))
    s.key("]")
    s.key("'")
    assert s.axes["vy"].step == s.axes["vx"].step
    assert s.axes["vy"].slew == s.axes["vx"].slew


def test_feel_keys_do_not_disturb_attitude_or_height_steps():
    """Those already sit at round 1 deg / 1 mm and shouldn't drift."""
    s = CommandShaper(default_axes())
    for k in "[];'{}":
        s.key(k)
    assert math.degrees(s.axes["pitch"].step) == pytest.approx(1.0)
    assert s.axes["height"].step == pytest.approx(1.0)


def test_sensitivity_and_smoothing_cannot_go_to_zero_or_negative():
    s = CommandShaper(default_axes(v_step=1.0, v_slew=2.0))
    for _ in range(50):
        s.key("[")
        s.key(";")
    assert s.axes["vx"].step > 0
    assert s.axes["vx"].slew > 0
