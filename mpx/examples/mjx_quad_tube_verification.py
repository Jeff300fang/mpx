from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax
import mujoco
from functools import partial
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig
from mpx.utils.fast_sls_visual import get_trajectory_tubes
from typing import Callable
import os
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
 
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_go2 as config

from timeit import default_timer as timer
from mpx.utils.render_obstacles import render_static_vertical_cylinder
from mpx.utils.mpc_utils import outside_circle_constraints
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SLSConfig

import imageio
import numpy as np

class VideoWriter:
    def __init__(self, path: str, fps: int = 30):
        self.path = path
        self.fps = fps
        self._writer = imageio.get_writer(path, fps=fps)

    def add(self, frame: np.ndarray):
        # frame: (H, W, 3) uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        self._writer.append_data(frame)

    def close(self):
        self._writer.close()


logdir = "/tmp/jax_trace"

# Define robot and scene parameters
robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])
mpc_frequency = config.mpc_frequency
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables
 
# Initialize simulation environment
sim_frequency = 100.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=0.7,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)

# --------- Define Constraints and Disturbances ---------
def render_obstacles(centers, radii):
    for (cx, cy), radius in zip(centers, radii):
        cyl_xy = np.array([cx, cy])   # x, y
        cyl_height = 1.0
        cyl_radius = radius
        cyl_z_base = 0.0                # sits on ground
        cyl_color = np.array([0.6, 0.6, 0.6, 1.0])

        static_cyl_id = -1
        static_cyl_id = render_static_vertical_cylinder(
            env.viewer,
            center_xy=cyl_xy,
            height=cyl_height,
            radius=(cyl_radius - 0.33), # Obstacles are inflated to capture size of quadruped
            z_base=cyl_z_base,
            color=cyl_color,
            geom_id=static_cyl_id,
        )

def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Enforce box constraints only on x[0] and x[1]:
      x[:2] - x_max[:2] <= 0
      x_min[:2] - x[:2] <= 0
    """
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        x_xy = x[:3]
        return jnp.concatenate(
            [x_xy - x_max[:3], x_min[:3] - x_xy],
            axis=0,
        )

    return constraints

def make_constant_disturbance(
    n: int,
    alpha: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, nc),
    where E[t] = alpha * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]

        diag = jnp.zeros(n, dtype=X_prefix.dtype)
        diag = diag.at[:2].set(alpha)   # first two entries = alpha

        E0 = jnp.diag(diag)              # (n, n)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance

x_max = jnp.array([15.0, 15.0, 1.25])
x_min = jnp.array([-15.0, -15.0, -1.25])

state_box_constraints = make_state_box_constraints(x_min, x_max)

# Predefined obstacles
num_constraints = 7
obstacles = jnp.array([[20.3, 0.35, 0.43]])
centers = obstacles[:, :2]
radii = obstacles[:, 2]

E_mag = 0.1
alpha_sim = E_mag * config.dt
disturbance = make_constant_disturbance(n=config.n, alpha=alpha_sim)
# --------------------------------------


# Define the MPC wrapper
obstacle_cosntraints = partial(outside_circle_constraints, centers=centers, radii=radii)

admm_config = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        condense_block_size=5,
        rho_max=1e5
    )
sls_config = SLSConfig(
    max_sls_iterations = 2,
    sls_primal_tol = 1e-2,
    enable_fastsls=False,
)
sqp_config = SQPConfig(
    max_sqp_iterations=2,
    warm_start=False
)
inital_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
X_in = jnp.tile(inital_state, (config.N + 1, 1))
U_in = jnp.tile(config.u_ref, (config.N, 1))

mpc = mpc_wrapper.MPCControllerWrapper(
    config,
    sls_config, sqp_config, admm_config,
    state_box_constraints, obstacles, num_constraints,
    disturbance,
    X_in, U_in)

env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
env.render()
counter = 0
# Main simulation loop
q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(),env.mjData.qvel.copy())
first = True

for i in range(0):
 
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
    
 
        ref_base_lin_vel = env._ref_base_lin_vel_H
        # ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
        ref_base_lin_vel = np.array([0.3, 0., 0.])
        ref_base_ang_vel =  np.array([0., 0., 0.])

        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
        
        contact_temp, _ = env.feet_contact_state()
        
        contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])
        tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u = mpc.run(qpos,qvel,input,contact)   
        render_obstacles(centers, radii)
    tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])
    state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)

    # time.sleep(0.1)
    counter += 1
    env.render()

sqp_config = SQPConfig(
    max_sqp_iterations=50,
    warm_start=False
)
sls_config = SLSConfig(
    max_sls_iterations = 2,
    sls_primal_tol = 1e-2,
    enable_fastsls=True,
)
inital_state = jnp.concatenate([config.p0, config.quat0,config.q0, jnp.zeros(6+config.n_joints),config.p_legs0,jnp.zeros(3*config.n_contact)])
X_in = jnp.tile(inital_state, (config.N + 1, 1))
U_in = jnp.tile(config.u_ref, (config.N, 1))

mpc = mpc_wrapper.MPCControllerWrapper(
    config,
    sls_config, sqp_config, admm_config,
    state_box_constraints, obstacles, num_constraints,
    disturbance,
    X_in, U_in)

qpos = env.mjData.qpos.copy()
qvel = env.mjData.qvel.copy()

contact_temp, _ = env.feet_contact_state()
contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])

ref_base_lin_vel = env._ref_base_lin_vel_H
ref_base_lin_vel = np.array([0.3, 0., 0.])
ref_base_ang_vel =  np.array([0., 0., 0.])

input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                    ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                    config.robot_height])

tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u, parameter = mpc.run(qpos,qvel,input,contact)

outdir = "mpc_data"
os.makedirs(outdir, exist_ok=True)

save_path = os.path.join(outdir, "go2_mpc_rollout.npz")

np.savez(
    save_path,
    X=np.asarray(X),
    U=np.asarray(U),
    V=np.asarray(V),
    Phi_x=np.asarray(Phi_x),
    Phi_u=np.asarray(Phi_u),
    parameter=np.asarray(parameter),
)

print(f"Saved MPC data to: {save_path}")