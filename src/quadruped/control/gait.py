"""Gait scheduler — dispatches swing/stance trajectories per leg.

Drives a quadruped through a periodic gait by:
- tracking a global phase in [0, 1) that advances with time
- offsetting each leg's local phase by a configured amount
- splitting each leg's local phase into stance and swing via duty_factor
- resolving a per-leg ground velocity from the body command, so a yaw rate
  turns the robot: v_leg = v_body + ω × r_leg  (see leg_velocity)
- planning footholds with the Raibert heuristic:
    touch_pos = home + v_leg * T_stance / 2
    lift_pos  = home - v_leg * T_stance / 2
- constructing a StanceFootTrajectory or SwingFootTrajectory per leg per tick
  and querying it at the current local s

Stateless across ticks: each foot_position(leg) call rebuilds the trajectory
from current phase + current body velocity. Simple and responsive; the price
is a small position discontinuity if v_body changes abruptly mid-step.

References:
- Stanford Pupper Controller — gait scheduler and foothold planning:
  https://pupper.readthedocs.io/en/latest/reference/controller.html
- Raibert footstep heuristic (foot-on-the-ground centered about hip):
  Raibert, "Legged Robots That Balance" (MIT Press, 1986)
"""

import numpy as np

from quadruped.control.trajectory import StanceFootTrajectory, SwingFootTrajectory


# Per-leg phase offsets in [0, 1). Diagonal pairs share an offset in trot;
# walk steps each leg in turn.
WALK_PHASE_OFFSETS = {
    "FL": 0.0,
    "FR": 0.25,
    "BL": 0.5,
    "BR": 0.75,
}
TROT_PHASE_OFFSETS = {
    "FL": 0.0,
    "BR": 0.0,
    "FR": 0.5,
    "BL": 0.5,
}

# Gait presets — pass to GaitScheduler(**preset, home_positions=...).
WALK_PRESET = {
    "period": 1.0,
    "duty_factor": 0.75,
    "apex_height": 30.0,
    "phase_offsets": WALK_PHASE_OFFSETS,
}
TROT_PRESET = {
    "period": 0.5,
    "duty_factor": 0.5,
    "apex_height": 30.0,
    "phase_offsets": TROT_PHASE_OFFSETS,
}


class GaitScheduler:
    """Per-leg foot-position dispatcher driven by a periodic gait phase.

    Config (set once at construction):
    - period:         full gait cycle duration (s)
    - duty_factor:    fraction of cycle each leg spends in stance, in (0, 1)
    - phase_offsets:  {leg_name: offset_in_[0,1)}
    - home_positions: {leg_name: np.array([x, y, z])} — foot position in body
                      frame when the robot is standing at rest
    - apex_height:    Z apex during swing (body frame, same units as home)

    State (mutated by advance()):
    - phase:         current global gait phase in [0, 1)
    - body_velocity: (vx, vy) as last passed to advance()
    - yaw_rate:      body rotation rate about z (rad/s) as last passed
    """

    def __init__(self, period, duty_factor, phase_offsets, home_positions, apex_height):
        """Store config; initialize phase to 0 and the command to rest."""
        self.period = period
        self.duty_factor = duty_factor
        self.phase_offsets = phase_offsets
        self.home_positions = home_positions
        self.apex_height = apex_height

        self.phase = 0.0
        self.body_velocity = (0.0, 0.0)
        self.yaw_rate = 0.0

        self.T_stance = period * duty_factor
        self.T_swing = period * (1 - duty_factor)

    def advance(self, dt, body_velocity, yaw_rate=0.0):
        """Advance the global phase by dt (s) and store the body command.

        body_velocity is a 2-tuple (vx, vy) in mm/s; yaw_rate is rad/s about the
        body z-axis, positive counter-clockwise (turn left).
        """
        self.phase = (self.phase + dt / self.period) % 1.0
        self.body_velocity = body_velocity
        self.yaw_rate = yaw_rate

    def leg_velocity(self, leg):
        """Ground velocity this leg's foot must track, as a 2-tuple (vx, vy).

        A body rotating at ω sweeps each leg through a different ground speed
        depending on where that leg is stationed, so velocity is per-leg rather
        than global:

            v_leg = v_body + ω × r_leg
                  = (vx − ω·r_y,  vy + ω·r_x)

        r_leg is the leg's home position in body frame. Pure rotation
        (v_body = 0) therefore sweeps the left legs backward while the right
        legs go forward, which is what turns the robot in place.
        """
        vx, vy = self.body_velocity
        if not self.yaw_rate:
            return vx, vy
        rx, ry = self.home_positions[leg][0], self.home_positions[leg][1]
        return vx - self.yaw_rate * ry, vy + self.yaw_rate * rx

    def is_swing(self, leg):
        """True if `leg` is currently in swing."""
        local_phase = (self.phase + self.phase_offsets[leg]) % 1.0
        return local_phase >= self.duty_factor

    def foot_position(self, leg):
        """Foot position (np.array shape (3,)) for `leg` at the current phase."""
        trajectory, s = self._trajectory_and_s(leg)
        return trajectory.position_at(s)

    def foot_velocity(self, leg):
        """Foot velocity (np.array shape (3,)) for `leg` at the current phase."""
        trajectory, s = self._trajectory_and_s(leg)
        return trajectory.velocity_at(s)

    def _trajectory_and_s(self, leg):
        """Return (active_trajectory, local s) for `leg` at the current phase.

        Foothold planning uses the Raibert heuristic: lift_pos and touch_pos
        are placed symmetrically about home so the leg's range of motion stays
        centered about its home position regardless of body velocity.
        """
        home = self.home_positions[leg]
        v_leg = self.leg_velocity(leg)
        # 3D for adding to home; the trajectory classes still take the 2-tuple.
        v_leg_3d = np.array([v_leg[0], v_leg[1], 0.0])

        touch_pos = home + v_leg_3d * self.T_stance / 2
        lift_pos  = home - v_leg_3d * self.T_stance / 2

        local_phase = (self.phase + self.phase_offsets[leg]) % 1.0
        if local_phase < self.duty_factor:
            # Stance: rescale local_phase from [0, duty_factor) to s in [0, 1).
            s = local_phase / self.duty_factor
            return StanceFootTrajectory(touch_pos, self.T_stance, v_leg), s
        # Swing: rescale local_phase from [duty_factor, 1) to s in [0, 1).
        s = (local_phase - self.duty_factor) / (1 - self.duty_factor)
        return SwingFootTrajectory(lift_pos, touch_pos, self.apex_height, self.T_swing, v_leg), s
