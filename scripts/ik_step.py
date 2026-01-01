"""
Inverse Kinematics (IK) Solver for MuJoCo Leg Simulation

This script implements an IK solver that moves the foot to follow a target position.
It uses the Jacobian Transpose method with damping regularization to solve for joint angles.

Key concepts:
- IK: Given a desired end-effector (foot) position, find the joint angles that achieve it
- Jacobian: A matrix that relates joint velocities to end-effector velocities
- Damped Least Squares: Regularization technique to avoid singularities and ensure stability
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# ============================================================================
# SETUP: Load the MuJoCo model
# ============================================================================

# Get the project root directory (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Construct path to the XML model file
XML_PATH = os.path.join(ROOT, "meshes", "single_leg.xml")

# Load the MuJoCo model and create simulation data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Verify that we have a mocap body (motion capture target) in the model
print("nmocap =", model.nmocap)  # should be 1 (we have one mocap body: "ik_target")

# ============================================================================
# GET SITE IDs: These are the points we'll track and control
# ============================================================================

# Get the ID of the foot site (where the foot is currently)
foot_site_id = model.site("foot_site").id
# Get the ID of the target site (where we want the foot to go)
target_site_id = model.site("target_site").id

# ============================================================================
# GET JOINT INDICES: These tell us where joint data is stored in arrays
# ============================================================================

# DOF (Degrees of Freedom) addresses: where joint velocities are stored in the velocity vector
# Each joint has a DOF address that tells us its position in the velocity array
dof_shoulder = model.joint("shoulder_joint").dofadr[0]  # Index in velocity array for shoulder
dof_wing = model.joint("wing_joint").dofadr[0]          # Index in velocity array for wing
dof_knee = model.joint("knee_joint").dofadr[0]          # Index in velocity array for knee
# Store all DOF indices in an array for easy access
dof_idxs = np.array([dof_shoulder, dof_wing, dof_knee], dtype=int)

# QPOS (Joint Position) addresses: where joint angles are stored in the position vector
# Each joint has a qpos address that tells us its position in the position array
qpos_shoulder = model.joint("shoulder_joint").qposadr[0]  # Index in position array for shoulder
qpos_wing = model.joint("wing_joint").qposadr[0]          # Index in position array for wing
qpos_knee = model.joint("knee_joint").qposadr[0]          # Index in position array for knee
# Store all qpos indices in an array for easy access
qpos_idxs = np.array([qpos_shoulder, qpos_wing, qpos_knee], dtype=int)

# Joint IDs: used to look up joint properties like limits
jids = np.array([
    model.joint("shoulder_joint").id,  # Joint ID for shoulder
    model.joint("wing_joint").id,      # Joint ID for wing
    model.joint("knee_joint").id       # Joint ID for knee
], dtype=int)

# ============================================================================
# HELPER FUNCTION: Clamp joint angles to their limits
# ============================================================================

def clamp_to_limits(q):
    """
    Ensure joint angles stay within their defined limits.
    
    Args:
        q: Array of joint angles [shoulder, wing, knee]
    
    Returns:
        q_clamped: Same array but with values clipped to joint limits
    """
    q_clamped = q.copy()  # Make a copy so we don't modify the original
    # For each joint, get its min/max limits and clamp the angle
    for i, jid in enumerate(jids):
        lo, hi = model.jnt_range[jid]  # Get lower and upper limits for this joint
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)  # Clamp to [lo, hi]
    return q_clamped

# ============================================================================
# IK SOLVER PARAMETERS: Tune these to adjust solver behavior
# ============================================================================

# Alpha: Step size for gradient descent (how much to move joints each iteration)
# Smaller = more stable but slower, Larger = faster but may overshoot
alpha = 0.6

# Damping: Regularization term to prevent singularities and ensure stability
# Prevents the Jacobian from becoming ill-conditioned (singular)
# Larger damping = more stable but less accurate
damping = 1e-2

# Tolerance: Stop solving when error is below this threshold (in distance units)
# If foot is within this distance of target, we're done
tol = 1e-2

# ============================================================================
# SIMULATION TIMING
# ============================================================================

dt = 0.01  # Time step (seconds) - how long each simulation step takes
t = 0.0    # Current time (seconds) - tracks how long simulation has run

# ============================================================================
# TARGET MOTION PARAMETERS: Define how the target moves in 3D space
# ============================================================================

# Center point around which the target will move
target_center = np.array([0.0, 0.0, 150.0])

# Choose the path type: "lissajous" (smooth 3D curve) or "step" (walking-like)
PATH = "step"   # "lissajous" or "step"

# --- 3D Lissajous: smooth motion in X,Y,Z ---
# Creates a smooth 3D figure-8 pattern by combining sine waves
amp_xyz = np.array([30.0, 0.0, 0.0])   # X, Y, Z amplitudes (how far it moves in each direction)
freq_xyz = np.array([0.55, 0.37, 0.73])  # Different frequencies for 3D motion (creates complex pattern)
phase_xyz = np.array([0.0, np.pi/3, np.pi/6])  # Phase offsets (starts at different points in the cycle)

# --- Step-like path: forward/back + lateral + lift ---
# Creates a walking-like motion: elliptical path in X-Y plane with lifting in Z
step_freq = 0.6        # How fast the step cycle repeats (Hz)
step_amp_x = 35.0      # Forward/backward motion amplitude
step_amp_y = 15.0      # Side-to-side motion amplitude, 15
lift_amp_z = 25.0      # Up/down "lift" amplitude (how high the foot lifts)

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Initialize forward kinematics (compute positions from current joint angles)
    mujoco.mj_forward(model, data)

    while viewer.is_running():

        # ====================================================================
        # STEP 1: Move the mocap target in 3D space
        # ====================================================================
        # The mocap (motion capture) body is a special body that we can directly
        # control - it's used as the target that the foot should follow
        
        if model.nmocap > 0:  # Check if we have any mocap bodies
            if PATH == "lissajous":
                # Lissajous curve: X, Y, Z all vary smoothly with different frequencies
                # This creates a complex 3D figure-8 pattern
                x = amp_xyz[0] * np.sin(2*np.pi*freq_xyz[0]*t + phase_xyz[0])
                y = amp_xyz[1] * np.sin(2*np.pi*freq_xyz[1]*t + phase_xyz[1])
                z = amp_xyz[2] * np.sin(2*np.pi*freq_xyz[2]*t + phase_xyz[2])
                # Set the mocap body position to the target center + the computed offset
                data.mocap_pos[0] = target_center + np.array([x, y, z])

            elif PATH == "step":
                # Step pattern: elliptical motion in X-Y plane with lifting in Z
                theta = 2*np.pi*step_freq*t  # Angle parameter (0 to 2π)
                x = step_amp_x * np.sin(theta)  # Forward/back motion
                y = step_amp_y * np.cos(theta)  # Side-to-side motion

                # Lift profile: 0..1..0 (always non-negative, like a foot lifting)
                # Uses cosine to create smooth lift: starts at 0, peaks at 1, returns to 0
                lift = 0.5 * (1.0 - np.cos(theta))  # 0 at start, 1 mid, 0 end
                z = lift_amp_z * lift  # Scale the lift by the amplitude

                # Set the mocap body position
                data.mocap_pos[0] = target_center + np.array([x, y, z])

        # ====================================================================
        # STEP 2: Update forward kinematics to get current foot position
        # ====================================================================
        # Compute where all bodies are based on current joint angles
        mujoco.mj_forward(model, data)

        # Get the current 3D position of the foot site
        p_foot = data.site_xpos[foot_site_id].copy()  # Where the foot currently is
        # Get the current 3D position of the target site
        p_target = data.site_xpos[target_site_id].copy()  # Where we want the foot to be
        # Calculate the error vector (difference between target and current foot position)
        err = (p_target - p_foot)  # 3D vector pointing from foot to target

        # ====================================================================
        # STEP 3: Solve IK if error is large enough
        # ====================================================================
        # Only solve if the foot is not already close enough to the target
        if np.linalg.norm(err) > tol:  # Check if distance error is above tolerance
            
            # --- Compute the Jacobian matrix ---
            # The Jacobian relates joint velocities to end-effector (foot) velocities
            # jacp: position Jacobian (3xN matrix: 3D position, N joints)
            # jacr: rotation Jacobian (3xN matrix: 3D rotation, N joints) - not used here
            jacp = np.zeros((3, model.nv))  # 3 rows (x,y,z), N columns (one per DOF)
            jacr = np.zeros((3, model.nv))  # Rotation Jacobian (not used, but required)
            # Compute the Jacobian for the foot site
            mujoco.mj_jacSite(model, data, jacp, jacr, foot_site_id)

            # Extract only the columns for our 3 joints (shoulder, wing, knee)
            # This gives us a 3x3 matrix: how foot position changes with each joint
            J = jacp[:, dof_idxs]  # 3x3 matrix: [3D position] x [3 joints]

            # --- Solve for joint angle changes using Damped Least Squares ---
            # We want to solve: J * dq = err
            # Where: J is Jacobian, dq is joint angle change, err is position error
            # 
            # Using Damped Least Squares: dq = J^T * (J*J^T + damping*I)^(-1) * err
            # This is more stable than: dq = J^(-1) * err (which can be singular)
            #
            # JJt = J * J^T (3x3 matrix)
            JJt = J @ J.T
            # Solve: (JJt + damping*I) * x = err, then dq = J^T * x
            # The damping term prevents singularities when the Jacobian is ill-conditioned
            dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)
            # dq is now a 3-element vector: [shoulder_change, wing_change, knee_change]

            # --- Update joint angles ---
            # Get current joint angles
            q = data.qpos[qpos_idxs].copy()  # Current angles: [shoulder, wing, knee]
            # Apply the computed change, scaled by alpha (step size)
            q_new = q + alpha * dq  # New angles = old angles + scaled change
            # Make sure new angles are within joint limits
            q_new = clamp_to_limits(q_new)

            # Set the new joint angles in the simulation
            data.qpos[qpos_idxs] = q_new
            # Recompute forward kinematics with new angles
            mujoco.mj_forward(model, data)

        # ====================================================================
        # STEP 4: Update visualization and advance time
        # ====================================================================
        viewer.sync()  # Update the 3D viewer with current state
        time.sleep(dt)  # Wait for the time step
        t += dt  # Advance simulation time