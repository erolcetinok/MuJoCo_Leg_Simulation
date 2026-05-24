"""Phase 2 home — trajectory generation.

Planned contents:
- cubic / quintic joint splines (scipy.interpolate.CubicSpline)
- dual 1D Bezier swing-foot trajectory (see mjbots blog + FootTrajectoryPlanner)
- timing/velocity scaling utilities

Planned contents:
- utilize cubic and quintic joint splines with a 5-3-5 spline trajectory maybe?
- use two 1D Bezier curves for the swing-foot trajectory, use one for x and one for y so they can be configured individually
- timing/velocity scaling utilities
    - doing it based off of the duty factor as opposed to time

Intentionally empty until Phase 2.
"""

import math
import numpy as np

#  - mjbots: Improved swing trajectory (https://blog.mjbots.com/2020/06/05/improved-swing-trajectory/)
#  - Stanford Pupper Controller Description (https://pupper.readthedocs.io/en/latest/reference/controller.html)
#  - Stanford Pupper Lab 7 — Control & Simulation
#  (https://pupper-independent-study.readthedocs.io/en/latest/course-material/lab-7-spr22.html)
#  - High-speed bounding with MIT Cheetah 2 (Park et al., 2017)
#  (https://journals.sagepub.com/doi/10.1177/0278364917694244)
#  - Step Trajectory and Gait Planner from MIT Cheetah (Hackaday) (https://hackaday.io/project/171456-diy-hobby-servos-qu
#  adruped-robot/log/178481-step-trajectory-and-gait-planner-from-mit-cheetah)
#  - pat92fr/FootTrajectoryPlanner (GitHub) (https://github.com/pat92fr/FootTrajectoryPlanner)
#  - Reduced-jerk 5-3-5 spline trajectory planning (https://www.researchgate.net/publication/282267415_Reduced_jerk_joint
#  _space_trajectory_planning_method_using_5-3-5_spline_for_robot_manipulators)
#  - Parameterizable and Jerk-Limited Trajectories with Blending (TUM)
#  (https://mediatum.ub.tum.de/doc/1614584/xs7blx45xiwr7lqfc0z3fr01k.trajectoryBlending.pdf)

class Bezier1D:
    """A 1D Bézier curve evaluated with s on the interval [0, 1]"""

    def __init__(self, control_points):
        self.control_points = np.asarray(control_points, dtype=float)
        self.degree = len(self.control_points) - 1

    def __call__(self, s):
        result = 0
        for i in range(self.degree + 1):
            term = math.comb(self.degree, i) * (1 - s)**(self.degree - i) * s**i * self.control_points[i]
            result += term
        return result

    def derivative(self):
        Q = self.degree * np.diff(self.control_points)
        return Bezier1D(Q)

class SwingFootTrajectory:

    def __init__(self, lift_pos, touch_pos, apex_height, T_swing, body_velocity=(0.0, 0.0)):
        self.T_swing = T_swing

        v_x, v_y = body_velocity
        shift_x = v_x * T_swing / 3
        shift_y = v_y * T_swing / 3

        x_cps = [lift_pos[0], lift_pos[0] - shift_x, touch_pos[0] + shift_x, touch_pos[0],]
        self.x_bezier = Bezier1D(x_cps)

        y_cps = [lift_pos[1], lift_pos[1] - shift_y, touch_pos[1] + shift_y, touch_pos[1],]
        self.y_bezier = Bezier1D(y_cps)

        z_up_cps = [lift_pos[2], lift_pos[2], apex_height, apex_height,]
        self.z_up_bezier = Bezier1D(z_up_cps)

        z_down_cps = [apex_height, apex_height, touch_pos[2], touch_pos[2],]
        self.z_down_bezier = Bezier1D(z_down_cps)

    def position_at(self, s):
        x_pos = self.x_bezier(s)
        y_pos = self.y_bezier(s)
        if s <= 0.5:
            s_local = 2 * s
            z_pos = self.z_up_bezier(s_local)
        else:
            s_local = 2 * s - 1
            z_pos = self.z_down_bezier(s_local)

        return np.array([x_pos, y_pos, z_pos])


    def velocity_at(self, s):
        x_vel = self.x_bezier.derivative()(s) / self.T_swing
        y_vel = self.y_bezier.derivative()(s) / self.T_swing
        if s <= 0.5:
            s_local = 2 * s
            z_vel = self.z_up_bezier.derivative()(s_local) * 2 / self.T_swing
        else:
            s_local = 2 * s - 1
            z_vel = self.z_down_bezier.derivative()(s_local) * 2 / self.T_swing

        return np.array([x_vel, y_vel, z_vel])

class StanceFootTrajectory:

    def __init__(self, touch_pos, T_stance, body_velocity=(0.0, 0.0)):
        self.T_stance = T_stance

        v_x, v_y = body_velocity

        x_cps = [touch_pos[0], touch_pos[0] - v_x * T_stance]
        self.x_bezier = Bezier1D(x_cps)

        y_cps = [touch_pos[1], touch_pos[1] - v_y * T_stance]
        self.y_bezier = Bezier1D(y_cps)

        z_cps = [touch_pos[2], touch_pos[2]]
        self.z_bezier = Bezier1D(z_cps)

    def position_at(self, s):
        x_pos = self.x_bezier(s)
        y_pos = self.y_bezier(s)
        z_pos = self.z_bezier(s)

        return np.array([x_pos, y_pos, z_pos])

    def velocity_at(self, s):
        x_vel = self.x_bezier.derivative()(s) / self.T_stance
        y_vel = self.y_bezier.derivative()(s) / self.T_stance
        z_vel = self.z_bezier.derivative()(s) / self.T_stance

        return np.array([x_vel, y_vel, z_vel])

