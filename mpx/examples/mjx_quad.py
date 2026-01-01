
import jax.numpy as jnp
import jax
import mujoco
from functools import partial
from jax import vmap
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
# Update JAX configuration
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
 
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
 
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_aliengo as config

from timeit import default_timer as timer
from mpx.utils.render_obstacles import render_static_vertical_cylinder
from mpx.utils.mpc_utils import outside_circle_constraints, combine_constraints
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SLSConfig
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

# Define robot and scene parameters
robot_name = "aliengo"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
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


# --------- Define Constraints ---------
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

def random_circles(key, K, radius=0.43, dtype=jnp.float32):
    """
    Uniformly sample K circle centers:
      x ~ U[1, 7], y ~ U[-3, 3]
    All circles share the same radius.

    Returns:
      centers: (K, 2)
      radii:   (K,)
    """
    key_x, key_y = jax.random.split(key, 2)
    x = jax.random.uniform(key_x, (K,), minval=1.0, maxval=7.0, dtype=dtype)
    y = jax.random.uniform(key_y, (K,), minval=-1.0, maxval=1.0, dtype=dtype)
    centers = jnp.stack([x, y], axis=1)
    radii = jnp.full((K,), radius, dtype=dtype)
    return centers, radii

# Example usage
key = jax.random.PRNGKey(1)
# num_constraints = 10
# centers, radii = random_circles(key, K=num_constraints, radius=0.43)

# Predefined obstacles
num_constraints = 1
centers = jnp.array([[2.0, 0.1]], dtype=jnp.float32)

radii = jnp.array([0.43], dtype=jnp.float32)
# --------------------------------------

# --------- Define Disturbance Matrices ---------


def smooth_interval(z, z_min, z_max, k=10.0):
    # ~1 inside [z_min, z_max], ~0 outside, smooth everywhere
    return 0.5 * (jnp.tanh(k * (z - z_min)) - jnp.tanh(k * (z - z_max)))

def gate_2d(px, py,
            x_min=-3.0, x_max=3.0,   # set wide if you only want y-strip
            y_min=-3.0, y_max=3.0,
            k=10.0):
    return smooth_interval(px, x_min, x_max, k) * smooth_interval(py, y_min, y_max, k)


def terrain_E_matrix(x):
    """
    Returns E(x) with shape (nx, nx).
    Interpretable as componentwise disturbance scaling, i.e.
      x_{k+1} = f(x_k,u_k) + E(x_k) w_k,  w_k in [-1,1]^{nx}.

    Assumed state ordering:
      [p(3), quat(4), v_lin(3), omega(3), q(12), dq(12), foot_pos(12), grf(12)]
    """
    n_joints = 12
    n_contact = 4
    nx = 13 + 2 * n_joints + 6 * n_contact  # 61

    # --- terrain patch gate parameters ---
    x_min, x_max = 0.5, 4.0
    y_min, y_max = -0.5, 3.0
    k = 10.0

    # --- disturbance magnitudes (interpret as per-state bounds if w in [-1,1]) ---
    alpha_vz      = 0.5    # affects base v_z
    alpha_omega   = 0.3    # affects base omega_x, omega_y
    alpha_foot_z  = 0.03   # affects each foot z position
    alpha_grf_z   = 50.0   # affects each foot GRF z

    # indices under the assumed layout
    px_idx, py_idx = 0, 1
    v_lin_start = 3 + 4            # 7
    vz_idx = v_lin_start + 2       # 9
    omega_start = v_lin_start + 3  # 10
    omega_x_idx = omega_start + 0  # 10
    omega_y_idx = omega_start + 1  # 11

    foot_pos_start = 13 + 2 * n_joints          # 37
    grf_start      = foot_pos_start + 3*n_contact  # 49

    # smooth activation
    s = gate_2d(x[px_idx], x[py_idx],
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                k=k)

    # build diagonal disturbance scaling
    diag = jnp.zeros((nx,))

    # base disturbances
    diag = diag.at[vz_idx].set(s * alpha_vz)
    diag = diag.at[omega_x_idx].set(s * alpha_omega)
    diag = diag.at[omega_y_idx].set(s * alpha_omega)

    # per-foot z position disturbances
    foot_z_indices = foot_pos_start + 3*jnp.arange(n_contact) + 2
    diag = diag.at[foot_z_indices].set(s * alpha_foot_z)

    # per-foot GRFz disturbances
    grf_z_indices = grf_start + 3*jnp.arange(n_contact) + 2
    diag = diag.at[grf_z_indices].set(s * alpha_grf_z)

    # E is diagonal; sparse + stable for linearization/robust bounds
    return jnp.diag(diag)

def disturbance(X):
    return vmap(terrain_E_matrix)(X)
# --------------------------------------


# Define the MPC wrapper
obstacle_cosntraints = partial(outside_circle_constraints, centers=centers, radii=radii)
cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        condense_block_size=5,
        rho_max=1e5
    )
sls_config = SLSConfig(
    max_sls_iterations = 1,
    sls_primal_tol = 1e-2
)
mpc = mpc_wrapper.MPCControllerWrapper(config, sls_config, cfg, obstacle_cosntraints, num_constraints, disturbance)
env.mjData.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])
env.render()
ids = []
# for i in range(8):
#      ids.append(render_vector(env.viewer,
#               np.zeros(3),
#               np.zeros(3),
#               0.1,
#               np.array([1, 0, 0, 1])))
counter = 0
# Main simulation loop
tau = jnp.zeros(config.n_joints)
tau_old = jnp.zeros(config.n_joints)
delay = int(0.007*sim_frequency)
print('Delay: ',delay)

q = config.q0.copy()
dq = jnp.zeros(config.n_joints)
mpc_time = 0
mpc.robot_height = config.robot_height
mpc.reset(env.mjData.qpos.copy(),env.mjData.qvel.copy())

while env.viewer.is_running():
 
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    if (counter % (sim_frequency / mpc_frequency) == 0 or counter == 0):
    
 
        ref_base_lin_vel = env._ref_base_lin_vel_H
        # ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
        ref_base_lin_vel = np.array([0.2, 0., 0.])
        ref_base_ang_vel =  np.array([0., 0., 0.])

        input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
        
        contact_temp, _ = env.feet_contact_state()
        
        contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])

        if counter != 0:
            for i in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()
                # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
                state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
                counter += 1
        start = timer()
        tau, q, dq = mpc.run(qpos,qvel,input,contact)   
        stop = timer()
        print("Time taken for MPC: ", stop-start)   
        render_obstacles(centers, radii)
        # for i in range(4):
        #     render_sphere(env.viewer,
        #                   collision_point[3*i:3*i+3],
        #                   0.2,
        #                   np.array([1, 0, 0, 0.5]),
        #                   ids[i])

    tau_fb = 10*(q-qpos[7:7+config.n_joints])-2*(qvel[6:6+config.n_joints])
    state, reward, is_terminated, is_truncated, info = env.step(action= tau + tau_fb)

    # time.sleep(0.1)
    counter += 1
    env.render()