import jax.numpy as jnp
import jax
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import os
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/go2/go2_mjx.xml"

# Contact frame names and body names for feet (or calves)
contact_frame = ["FL", "FR", "RL", "RR"]
body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

# Time and stage parameters
dt = 0.02
N = 25
mpc_frequency = 100

# Gait parameters
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # galop
duty_factor = 0.65
step_freq = 1.35
step_height = 0.065
initial_height = 0.1
robot_height = 0.27

# Initial base pose and joint configuration
p0 = jnp.array([0.0, 0.0, robot_height])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8], dtype=jnp.float32)
q0_init = q0

# Nominal foot positions (world/body-frame depends on your implementation)
p_legs0 = jnp.array([
     0.192,  0.142, 0.0,   # FL
     0.192, -0.142, 0.0,   # FR
    -0.195,  0.142, 0.0,   # RL
    -0.195, -0.142, 0.0,   # RR
], dtype=jnp.float32)

# Dimensions
n_joints = 12
n_contact = len(contact_frame)

# === Key change: GRF is NOT part of the state ===
grf_as_state = False

# State dimension:
# 13 = [p(3), quat(4), v(3), omega(3)]
# 2*n_joints = [qj(12), dqj(12)]
n = 13 + 2 * n_joints

# Control dimension: joint torques
m = n_joints

u_ref = jnp.zeros(m)

# Cost matrices
Qp     = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
Qrot   = jnp.diag(jnp.array([1000.0, 1000.0, 0.0]))
Qq     = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qdp    = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 5e3
Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e2
Qdq    = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qtau   = jnp.diag(jnp.ones(n_joints)) * 1e-1

# === Key change: remove GRF and leg-contact blocks from W ===
# If you still want foot tracking, it should be implemented as a kinematic residual in the objective,
# not via Qleg/Q_grf blocks tied to GRF states.
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qtau)

use_terrain_estimation = True

# === Key change: objective/hessian flags match "no GRF in state" ===
cost = partial(mpc_objectives.quadruped_wb_obj, False)
hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn, False)

# === Key change: dynamics must be compatible with GRF not in state ===
# Pick the contact-handling dynamics that computes contacts internally.
dynamics = mpc_dyn_model.quadruped_wb_dynamics_explicit_contact
# Alternatively (if your codebase supports it):
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_learned_contact_model

# Torque limits
max_torque = 25
min_torque = -25
