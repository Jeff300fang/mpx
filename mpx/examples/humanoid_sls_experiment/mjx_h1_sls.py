from jax import config
config.update("jax_enable_x64", False)

import os
import sys
from timeit import default_timer as timer
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

import mujoco
import mujoco.viewer

# -----------------------------
# Path setup
# -----------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))

# -----------------------------
# JAX cache config
# -----------------------------
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# -----------------------------
# MPC imports
# -----------------------------
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig, SLSConfig

import mpx.utils.mpc_wrapper_reference_shifter as mpc_wrapper
import mpx.config.config_h1_sls_expe as config


# -----------------------------
# Constraints / disturbance
# -----------------------------
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
        x_xy = x[:2]
        return jnp.concatenate(
            [x_xy - x_max[:2], x_min[:2] - x_xy],
            axis=0,
        )

    return constraints


def make_constant_disturbance(n: int, alpha: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, n),
    where E[t] = diag([alpha, alpha, 0, ..., 0]) for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        diag = jnp.zeros(n, dtype=X_prefix.dtype)
        diag = diag.at[0:3].set(alpha)
        diag = diag.at[26:29].set(0.01)
        E0 = jnp.diag(diag)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance


# -----------------------------
# Viewer marker helpers (cylinder pillar)
# -----------------------------
def clear_user_geoms(viewer: mujoco.viewer.Handle) -> None:
    # Clear any previously injected geoms
    viewer.user_scn.ngeom = 0


def add_cylinder_pillar(
    viewer: mujoco.viewer.Handle,
    pos_xyz: np.ndarray,
    radius: float,
    height: float,
    rgba=(1.0, 0.2, 0.2, 0.6),
) -> None:
    """
    Inject a cylinder geom into the viewer scene.

    MuJoCo cylinder sizes:
      size[0] = radius
      size[1] = half-height
    """
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        # Scene geom buffer full; skip to avoid overflow
        return

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([float(radius), float(height) / 2.0, 0.0], dtype=np.float64),
        pos=np.array(pos_xyz, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


# -----------------------------
# Problem setup
# -----------------------------
E_mag = 0.025
alpha_sim = E_mag * config.dt
disturbance = make_constant_disturbance(n=config.n, alpha=alpha_sim)

x_max = jnp.array([15.0, 15.0])
x_min = -x_max
state_box_constraints = make_state_box_constraints(x_min, x_max)

# Obstacles: each row is [x, y, radius]
obstacles = jnp.array([[0.6, 0.1, 0.43]])

# Cylinder pillar height (visual only)
OBSTACLE_HEIGHT = 2  # meters

# -----------------------------
# MuJoCo model
# -----------------------------
model = mujoco.MjModel.from_xml_path(
    os.path.join(dir_path, "..", "..", "data", "unitree_h1", "mjx_scene_h1_walk.xml")
)
data = mujoco.MjData(model)

mpc_frequency = 250.0
sim_frequency = 500.0
model.opt.timestep = 1.0 / sim_frequency

# -----------------------------
# MPC configs
# -----------------------------
admm_config = ADMMConfig(
    eps_abs=1e-2,
    eps_rel=1e-2,
    condense_block_size=5,
    rho_max=1e5,
)
sls_config = SLSConfig(
    max_sls_iterations=2,
    sls_primal_tol=1e-1,
    enable_fastsls=True,
)
sqp_config = SQPConfig(
    max_sqp_iterations=10,
)

num_constraints = 5

initial_state = jnp.concatenate(
    [
        config.p0,
        config.quat0,
        config.q0,
        jnp.zeros(6 + config.n_joints),
        config.p_legs0,
        jnp.zeros(3 * config.n_contact),
    ]
)

X_in = jnp.tile(initial_state, (config.N + 1, 1))
U_in = jnp.tile(config.u_ref, (config.N, 1))

mpc = mpc_wrapper.MPCControllerWrapper(
    config,
    sls_config,
    sqp_config,
    admm_config,
    state_box_constraints,
    obstacles,
    num_constraints,
    disturbance,
    X_in,
    U_in,
)

# Initialize sim state
data.qpos = np.array(jnp.concatenate([config.p0, config.quat0, config.q0]), dtype=np.float64)

ref_base_lin_vel = jnp.array([0.75, 0.0, 0.0])
ref_base_ang_vel = jnp.array([0.0, 0.0, 0.0])

inp = np.array(
    [
        float(ref_base_lin_vel[0]),
        float(ref_base_lin_vel[1]),
        float(ref_base_lin_vel[2]),
        float(ref_base_ang_vel[0]),
        float(ref_base_ang_vel[1]),
        float(ref_base_ang_vel[2]),
        1.0,
    ],
    dtype=np.float64,
)
qpos = data.qpos.copy()
qvel = data.qvel.copy()
# Set this to the current contact state to use blind step adaptation
contact = np.zeros(config.n_contact, dtype=np.float64)
import time
start = time.perf_counter()
tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u, parameter = mpc.run(qpos, qvel, inp, contact, use_xin_reference=False)
for zi in range(1):
    start_policy = time.perf_counter()
    tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u, parameter = mpc.run(qpos, qvel, inp, contact, use_xin_reference=True)
end_comp = time.perf_counter()

print("Compilation Time:", (end_comp - start))
print("Policy time:", (end_comp - start_policy))
outdir = "mpc_data"
os.makedirs(outdir, exist_ok=True)

save_path = os.path.join(outdir, "h1_mpc_rollout.npz")

np.savez(
    save_path,
    X=np.asarray(X),
    U=np.asarray(U),
    V=np.asarray(V),
    Phi_x=np.asarray(Phi_x),
    Phi_u=np.asarray(Phi_u),
    parameter=np.asarray(parameter),
    obstacles=np.asarray(obstacles)
)