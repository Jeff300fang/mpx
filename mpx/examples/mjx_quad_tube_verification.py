from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax
import mujoco
from functools import partial
from jax import vmap
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
from mpx.utils.mpc_utils import outside_circle_constraints, combine_constraints
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SLSConfig
# Set GPU device for JAX
# gpu_device = jax.devices('gpu')[0]
# jax.default_device(gpu_device)

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

x_max = jnp.array([15.0, 15.0, 1.25])
x_min = jnp.array([-15.0, -15.0, 0.25])

state_box_constraints = make_state_box_constraints(x_min, x_max)

# Example usage
key = jax.random.PRNGKey(1)
# num_constraints = 10
# centers, radii = random_circles(key, K=num_constraints, radius=0.43)

# Predefined obstacles
num_constraints = 7
obstacles = jnp.array([[20.3, 0.35, 0.43]])
centers = jnp.array([[2.0, 0.1]])

radii = jnp.array([0.43])
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

def disturbance_terrain(X):
    return vmap(terrain_E_matrix)(X)

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
first = True

# qpos = env.mjData.qpos.copy()
# qvel = env.mjData.qvel.copy()
# contact_temp, _ = env.feet_contact_state()
# contact = np.array([contact_temp[robot_feet_geom_names[leg]] for leg in ['FL','FR','RL','RR']])

# video = VideoWriter("go2_run.mp4", fps=30) 
# ref_base_lin_vel = env._ref_base_lin_vel_H
# # ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
# ref_base_lin_vel = np.array([0.3, 0., 0.])
# ref_base_ang_vel =  np.array([0., 0., 0.])

# input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
#                     ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
#                     config.robot_height])
# tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u = mpc.run(qpos,qvel,input,contact)
# writer = imageio.get_writer("go2_run.mp4", fps=30)
# m = env.mjModel   # sometimes env.model or env._model
# d = env.mjData 
# fb_w = int(getattr(m.vis.global_, "offwidth", 640))
# fb_h = int(getattr(m.vis.global_, "offheight", 480))

# W = min(640, fb_w)
# H = min(480, fb_h)

# renderer = mujoco.Renderer(m, height=H, width=W)
# for i in range(config.N):
#     qpos = env.mjData.qpos.copy()
#     qvel = env.mjData.qvel.copy()
#     tau = U[i, :config.n_joints]
#     q = X[i, 7:config.n_joints + 7]
#     tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
#     print("Time step:", i, "/", config.N, ":", jnp.linalg.norm(qpos[:7] - X[i, :7]))
#     state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
#     env.render()
#     renderer.update_scene(d)
#     frame = renderer.render()          # (H, W, 3), uint8
#     writer.append_data(frame)

# writer.close()

# while env.viewer.is_running():
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

        if counter != 0:
            for i in range(delay):
                qpos = env.mjData.qpos.copy()
                qvel = env.mjData.qvel.copy()
                # tau_fb = K@(x-np.concatenate([qpos,qvel]))

                tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
                state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
                counter += 1
        start = timer()
        tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u = mpc.run(qpos,qvel,input,contact)   
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
# X_in = X
# U_in = U

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

video = VideoWriter("go2_run.mp4", fps=30) 
ref_base_lin_vel = env._ref_base_lin_vel_H
# ref_base_ang_vel =  np.array([0., 0., env._ref_base_ang_yaw_dot])
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

writer = imageio.get_writer("go2_run.mp4", fps=30)
m = env.mjModel   # sometimes env.model or env._model
d = env.mjData 
fb_w = int(getattr(m.vis.global_, "offwidth", 640))
fb_h = int(getattr(m.vis.global_, "offheight", 480))

W = min(640, fb_w)
H = min(480, fb_h)

def get_x_actual_from_mujoco(model, data, contact_id, qpos, qvel, grf_as_state: bool, n_contact: int):
    """
    Build x_actual with the SAME ordering used by MPCControllerWrapper.run():
      x0 = [qpos, qvel, foot_op, (grf or zeros)]
    """
    data.qpos = np.asarray(qpos)
    # data.qvel is not needed for kinematics, but keep consistent if you want:
    # data.qvel = np.asarray(qvel)

    mujoco.mj_kinematics(model, data)

    foot_op = np.array([data.geom_xpos[g] for g in contact_id], dtype=np.float64).reshape(-1)  # (3*n_contact,)

    if grf_as_state:
        grf = np.zeros(3 * n_contact, dtype=np.float64)  # placeholder unless you compute GRF
        x_actual = np.concatenate([np.asarray(qpos), np.asarray(qvel), foot_op, grf], axis=0)
    else:
        x_actual = np.concatenate([np.asarray(qpos), np.asarray(qvel), foot_op], axis=0)

    return x_actual

renderer = mujoco.Renderer(m, height=H, width=W)
diff = np.zeros((config.N, config.n))
for i in range(config.N):
    qpos = env.mjData.qpos.copy()
    qvel = env.mjData.qvel.copy()
    tau = U[i, :config.n_joints]
    q = X[i, 7:config.n_joints + 7]
    tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
    print("Time step:", i, "/", config.N, ":", jnp.linalg.norm(qpos[:7] - X[i, :7]))
    # state, reward, is_terminated, is_truncated, info = env.step(action=tau + tau_fb)
    state, reward, is_terminated, is_truncated, info = env.step(action=tau)
    x_actual = get_x_actual_from_mujoco(
        mpc.model, mpc.data, mpc.contact_id,
        qpos=env.mjData.qpos.copy(),
        qvel=env.mjData.qvel.copy(),
        grf_as_state=config.grf_as_state,
        n_contact=config.n_contact
    )
    x_pred = np.asarray(X[i + 1])
    diff[i] = np.abs(x_actual - x_pred)

    env.render()
    renderer.update_scene(d)
    frame = renderer.render()          # (H, W, 3), uint8
    writer.append_data(frame)

writer.close()


# Plot tubes
import numpy as np
import matplotlib.pyplot as plt
import os
import math
tubes = get_trajectory_tubes(Phi_x)
tube_sizes = tubes[1:]

# Ensure numpy
# diff = np.asarray(diff)                       # (N, nx)
# tube_sizes = np.asarray(tube_sizes)           # (N, nx) if you used tubes[1:]
# N, nx = diff.shape
# assert tube_sizes.shape[0] == N and tube_sizes.shape[1] == nx, (tube_sizes.shape, diff.shape)

# t = np.arange(N) * config.dt

# # Layout
# ncols = 6
# nrows = math.ceil(nx / ncols)

# fig_w = 18
# fig_h = 3.0 * nrows

# outdir = "tube_vs_diff"
# os.makedirs(outdir, exist_ok=True)
# save_path = os.path.join(outdir, f"tube_vs_diff_6perrow_N{N}_nx{nx}.png")

# fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
# axes = np.atleast_2d(axes)

# for j in range(nx):
#     r = j // ncols
#     c = j % ncols
#     ax = axes[r, c]

#     ax.plot(t, tube_sizes[:, j], linewidth=1.2, label="tube")
#     ax.plot(t, diff[:, j],       linewidth=1.2, label="|x_actual - x_pred|")

#     ax.set_title(f"x[{j}]", fontsize=9)
#     ax.grid(True)

#     # Optional: log scale helps when magnitudes vary a lot
#     # ax.set_yscale("log")

# # Turn off unused axes
# for j in range(nx, nrows * ncols):
#     r = j // ncols
#     c = j % ncols
#     axes[r, c].axis("off")

# # Only label bottom row to reduce clutter
# for ax in axes[-1, :]:
#     ax.set_xlabel("Time [s]")

# # Single legend for the whole figure
# handles, labels = axes[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper right")

# fig.suptitle("Tube size vs actual deviation (per state dimension)", y=0.995)
# plt.tight_layout()

# fig.savefig(save_path, dpi=250, bbox_inches="tight")
# plt.close(fig)

# print(f"Saved: {save_path}")
diff = np.asarray(diff)                 # (N, nx)
tube_sizes = np.asarray(tube_sizes)     # (N, nx)
N, nx = diff.shape
assert tube_sizes.shape == (N, nx), (tube_sizes.shape, diff.shape)

t = np.arange(N) * config.dt
nx_no_contact_grf = nx - 12
def minmax_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map 1D array to [0,1]. If nearly-constant, return zeros."""
    x = np.asarray(x)
    x_min = np.min(x)
    x_max = np.max(x)
    denom = x_max - x_min
    if denom < eps:
        return np.zeros_like(x)
    return (x - x_min) / denom

# Layout
ncols = 6
nrows = math.ceil(nx / ncols)

fig_w = 18
fig_h = 3.0 * nrows

outdir = "tube_vs_diff"
os.makedirs(outdir, exist_ok=True)
save_path = os.path.join(outdir, f"tube_vs_diff_norm01_6perrow_N{N}_nx{nx}.png")

fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
axes = np.atleast_2d(axes)

for j in range(nx_no_contact_grf):
    r = j // ncols
    c = j % ncols
    ax = axes[r, c]

    tube_j = tube_sizes[:, j]
    diff_j = diff[:, j]

    tube_n = minmax_01(tube_j)
    diff_n = minmax_01(diff_j)

    ax.plot(t, tube_n, linewidth=1.2, label="tube (norm)")
    ax.plot(t, diff_n, linewidth=1.2, label="diff (norm)")

    ax.set_title(f"x[{j}]", fontsize=9)
    ax.grid(True)
    ax.set_ylim(-0.05, 1.05)  # keep consistent scale across subplots

# Turn off unused axes
for j in range(nx_no_contact_grf, nrows * ncols):
    r = j // ncols
    c = j % ncols
    axes[r, c].axis("off")

for ax in axes[-1, :]:
    ax.set_xlabel("Time [s]")

# Single legend for the whole figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

fig.suptitle("Tube vs deviation (per-dimension, independently normalized to [0,1])", y=0.995)
plt.tight_layout()
fig.savefig(save_path, dpi=250, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {save_path}")