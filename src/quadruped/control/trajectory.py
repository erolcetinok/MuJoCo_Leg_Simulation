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

