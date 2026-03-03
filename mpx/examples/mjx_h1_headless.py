import os
import sys
from timeit import default_timer as timer
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

import mujoco

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

import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_h1 as config

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
        return jnp.concatenate([x_xy - x_max[:2], x_min[:2] - x_xy], axis=0)

    return constraints


def make_constant_disturbance(n: int, alpha: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, n),
    where E[t] = diag([alpha, alpha, 0, ..., 0]) for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        diag = jnp.zeros(n, dtype=X_prefix.dtype).at[:2].set(alpha)
        E0 = jnp.diag(diag)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance


# -----------------------------
# Problem setup
# -----------------------------
E_mag = 0.2
alpha_sim = E_mag * config.dt
disturbance = make_constant_disturbance(n=config.n, alpha=alpha_sim)

x_max = jnp.array([15.0, 15.0])
x_min = -x_max
state_box_constraints = make_state_box_constraints(x_min, x_max)

# Obstacles: each row is [x, y, radius]
obstacles = jnp.array([[2.0, 0.2, 0.43], [1.7, -1.1, 0.43]])

# -----------------------------
# MuJoCo model
# -----------------------------
model = mujoco.MjModel.from_xml_path(
    os.path.join(dir_path, "..", "data", "unitree_h1", "mjx_scene_h1_walk.xml")
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
    max_sls_iterations=1,
    sls_primal_tol=1e-1,
    enable_fastsls=True,
)
sqp_config = SQPConfig(
    max_sqp_iterations=1,
)

num_constraints = 6

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

# -----------------------------
# Run loop (HEADLESS)
# -----------------------------
tau = jnp.zeros(config.n_joints)
mpc_update_idx = 0

# Prime simulation once (optional but mirrors viewer version)
mujoco.mj_step(model, data)

delay = int(0 * sim_frequency)
print("Delay:", delay)

mpc.robot_height = config.robot_height
mpc.reset(np.array(data.qpos.copy()), np.array(data.qvel.copy()))

counter = 0
T_STEPS = 320
total_time = 0.0
i = 0

while i < T_STEPS:
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    # MPC update
    if counter % int(sim_frequency / config.mpc_frequency) == 0 or counter == 0:
        if counter != 0:
            for _ in range(delay):
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()
                tau_fb = -3.0 * (qvel[6 : 6 + config.n_joints])
                data.ctrl = np.array(tau, dtype=np.float64) + np.array(tau_fb, dtype=np.float64)
                mujoco.mj_step(model, data)
                counter += 1

        ref_base_lin_vel = jnp.array([0.6, 0.0, 0.0])
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

        # Set this to the current contact state to use blind step adaptation
        contact = np.zeros(config.n_contact, dtype=np.float64)

        start = timer()
        tau, q, dq, X, U, V, backoffs, Phi_x, Phi_u, parameter = mpc.run(qpos, qvel, inp, contact)

        now = timer()          # wall-clock
        sim_t = data.time      # MuJoCo simulation time
        mpc_update_idx += 1

        stop = timer()
        # print(f"Time elapsed: {stop - start}")
        if i != 0:
            total_time += (stop - start)
        i += 1
        # print(i)

    # Apply control at sim rate (simple damping term)
    counter += 1
    data.ctrl = np.array(tau, dtype=np.float64) - 3.0 * qvel[6 : 6 + config.n_joints]
    mujoco.mj_step(model, data)

print(total_time / (T_STEPS - 1))
mpc_update_idx += 1
