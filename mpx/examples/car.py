"""
dubins_car_mpc_experiment.py

End-to-end experiment: use mpx.utils.generic_mpc_wrapper.GenericMPCControllerWrapper
to control a Dubins car with box constraints on controls.

State:      x = [px, py, theta]
Control:    u = [v, omega]

Assumptions (matches your mpc() usage):
  - dynamics(x, u, t, *, parameter=...) returns x_{t+1} (discrete-time)
  - cost(W, reference, x, u, t) returns scalar stage cost
  - constraints(x, u, t) returns g(x,u,t) with g <= 0 (inequality)
  - disturbance(X_prefix) returns E used by get_controller(..., E, eta)
"""

from __future__ import annotations
from functools import partial
from jax import config
config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Any, Callable

import time
import jax
import jax.numpy as jnp

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper
from mpx.utils.mpc_utils import outside_circle_constraints, combine_constraints
from mpx.utils.fast_sls_visual import get_trajectory_tubes
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# -----------------------------
# Angle wrapping
# -----------------------------
def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles elementwise to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# -----------------------------
# Dubins car dynamics
# x = [px, py, theta], u = [v, omega]
# -----------------------------
@jax.jit
def dubins_step(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    px, py, th = x[0], x[1], x[2]
    v, om = u[0], u[1]

    px_next = px + dt * v * jnp.cos(th)
    py_next = py + dt * v * jnp.sin(th)
    th_next = wrap_to_pi(th + dt * om)

    return jnp.array([px_next, py_next, th_next], dtype=x.dtype)

@jax.jit
def dubins_step_with_disturbance(
    key: jax.Array,          # PRNGKey
    x: jnp.ndarray,          # (3,)
    u: jnp.ndarray,          # (2,)
    E: jnp.ndarray,          # (3,3)
    dt: float,
) -> tuple[jax.Array, jnp.ndarray]:
    """
    Simulates: x_{k+1} = f(x_k,u_k) + E w,   with ||w||_2 <= 1
    where w is sampled uniformly from the unit ball in R^{nx}.

    Returns (key_next, x_next).
    """
    px, py, th = x
    v, om = u

    # Nominal Dubins step
    px_next = px + dt * v * jnp.cos(th)
    py_next = py + dt * v * jnp.sin(th)
    th_next = wrap_to_pi(th + dt * om)
    x_nom = jnp.array([px_next, py_next, th_next], dtype=x.dtype)

    # Sample w ~ Uniform(unit l2 ball in R^3)
    key, key_dir, key_rad = jax.random.split(key, 3)
    z = jax.random.normal(key_dir, (x.shape[0],), dtype=x.dtype)
    z_norm = jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype)
    z = z / z_norm

    n = jnp.asarray(x.shape[0], dtype=x.dtype)  # n = 3, but keep generic
    r = jax.random.uniform(key_rad, (), minval=0.0, maxval=1.0, dtype=x.dtype) ** (1.0 / n)
    # w = r * z  # ||w||_2 <= 1
    w = jnp.array([0.0, 1.0, 0])

    # Additive disturbance
    x_next = x_nom + E @ w
    return key, x_next, w


def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    """Discrete-time dynamics required by your model evaluator."""
    dt = parameter
    return dubins_step(x, u, dt)


# -----------------------------
# Cost and Constraints matching your mpc() expectations
# -----------------------------
def cost(W, reference, x, u, t):
    """
    W = [wx, wy, wtheta, wv, womega]
    """
    wx, wy, wtheta, wv, womega = W
    xref = reference[t]

    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    dth = wrap_to_pi(x[2] - xref[2])

    v, om = u[0], u[1]

    return (
        wx * (dx * dx)
        + wy * (dy * dy)
        + wtheta * (dth * dth)
        + wv * (v * v)
        + womega * (om * om)
    )


def make_control_box_constraints(u_min: jnp.ndarray, u_max: jnp.ndarray) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for control bounds:
      u - u_max <= 0
      u_min - u <= 0
    """
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints


def make_zero_disturbance(n: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Conservative default: returns E with shape (T, n, nc).
    If your get_controller expects a different E shape, change this function.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        return jnp.zeros((T, n, n), dtype=X_prefix.dtype)

    return disturbance

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
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)  # (n, nc)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance


# -----------------------------
# Config
# -----------------------------
@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray

@partial(jax.jit, static_argnames=("N",))
def build_forward_reference(
    x0: jnp.ndarray,   # (3,)
    N: int,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a horizon reference assuming constant controls:
      v = v_ref, omega = om_ref
    Returns:
      X_ref: (N+1, 3)
      U_ref: (N, 2)
    """
    v_ref: float = 1.0
    om_ref: float = 0.0
    u = jnp.array([v_ref, om_ref], dtype=x0.dtype)
    U_ref = jnp.broadcast_to(u, (N, 2))

    def step(carry, _):
        x = carry
        x_next = dubins_step(x, u, dt)
        return x_next, x_next

    # produce x1..xN, then prepend x0
    _, xs = jax.lax.scan(step, x0, xs=None, length=N)
    X_ref = jnp.concatenate([x0[None, :], xs], axis=0)
    X_ref = X_ref.at[:, 1].set(0.0)
    return X_ref, U_ref


def dyn_defect_from_predictions(
    X_pred: jnp.ndarray,   # (N+1, 3)
    U_pred: jnp.ndarray,   # (N, 2)
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      err      : (N, 3)  raw defect X[k+1] - f(X[k], U[k])
      err_wrapped : (N, 3) where theta component is wrapped to (-pi, pi]
    """
    # predicted next via dynamics applied to predicted (x,u)
    X_next_hat = jax.vmap(lambda x, u: dubins_step(x, u, dt))(X_pred[:-1], U_pred)

    err = X_pred[1:] - X_next_hat

    # wrap theta error (component 2)
    err_wrapped = err.at[:, 2].set(wrap_to_pi(err[:, 2]))
    return err, err_wrapped


def summarize_dyn_defect(err_wrapped: jnp.ndarray) -> dict:
    """
    Computes norms per step and summary stats.
    """
    # per-step L2 norm (over state components)
    step_l2 = jnp.linalg.norm(err_wrapped, axis=1)  # (N,)
    # per-component max abs
    comp_max = jnp.max(jnp.abs(err_wrapped), axis=0)  # (3,)

    return {
        "step_l2": step_l2,
        "max_step_l2": jnp.max(step_l2),
        "mean_step_l2": jnp.mean(step_l2),
        "median_step_l2": jnp.median(step_l2),
        "comp_max_abs": comp_max,
    }

def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for state bounds:
      x - x_max <= 0
      x_min - x <= 0

    Note: This constrains theta directly in [-pi, pi] if you set that as bounds.
    Your dynamics already wrap theta, so it's consistent.
    """
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)

    return constraints

# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Dimensions
    n = 3      # [px, py, theta]
    nu = 2     # [v, omega]

    # Horizon and dt
    N = 100
    dt = 0.025

    # Weights: (x, y, theta, v, omega)
    W = jnp.array([5.0, 5.0, 0.1, 0.1, 0.1])

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,)),
    )

    admm_cfg = ADMMConfig(
        eps_abs=4e-2,
        eps_rel=0,
        rho_max=1e10,
        max_iterations=1000,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=7,
        sls_primal_tol=1e-2,
        enable_fastsls=False
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations = 54,
    )

    parameter = dt

    v_max = 5.0
    om_max = 10.0

    u_min = jnp.array([-v_max, -om_max])
    u_max = jnp.array([v_max,  om_max])

    constraints_u = make_control_box_constraints(u_min, u_max)
    x_max = jnp.array([10.0, 10.0, jnp.pi], dtype=jnp.float64)   # [px, py, theta]
    x_min = -x_max                                                # symmetric box; adjust if desired

    constraints_x = make_state_box_constraints(x_min, x_max)
    
    centers = jnp.array([[1.0, 0.04]])   # (K,2)
    radii   = jnp.array([0.15])         # (K,)
    K = centers.shape[0]

    # Inflate a bit if you want “safety margin”
    obstacle_constraints = partial(
        outside_circle_constraints,
        centers=centers,
        radii=radii,
    )  # returns (K,)

    # Combine: first control bounds, then obstacles
    # constraints_all = combine_constraints(constraints_u, obstacle_constraints)
    # constraints_all = combine_constraints(
    #     combine_constraints(constraints_u, constraints_x),
    #     obstacle_constraints,
    # )
    constraints_all = combine_constraints(constraints_u, constraints_x)

    # Total constraint count:
    nc = 2 * nu + 2 * n
    # nc = 2 * nu + K

    # disturbance = make_zero_disturbance(n=n)
    E_mag = 0.4
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    controller = GenericMPCControllerWrapper(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
        X_in=jnp.zeros((cfg.N + 1, cfg.n)),
        U_in=jnp.zeros((cfg.N, cfg.nu))
    )

    # -----------------------------
    # Initial condition
    # -----------------------------
    x = jnp.array([0.0, 0.0, 0.0])
    # X_ref, U_ref = build_forward_reference(x, N, dt)
    # print(X_ref)
    x_goal = jnp.array([2.0, 2.0, 0.0])  # shape (nx,)
    X_ref = jnp.tile(x_goal[None, :], (N + 1, 1))  # shape (N+1, nx)
    reference = X_ref
    # Closed-loop rollout
    T_steps = min(int(5.0 / dt), 100)  # simulate ~5s (cap)
    T_steps = N
    xs = []
    us = []

    total_time = 0.0
    min_time = float("inf")

    # Compilation warmup
    # controller.run(x0=x, reference=reference, parameter=parameter)

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(3)

    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    # Fully mpc closed loop
    # for k in range(T_steps):
    #     print(f"sim iteration {k}")
    #     start = time.perf_counter()
    #     u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(x0=x, reference=reference, parameter=parameter)
    #     u0.block_until_ready()
    #     u0 = jnp.clip(u0, u_min * 2, u_max * 2)
    #     end = time.perf_counter()
    #     tube = get_trajectory_tubes(Phi_x)
    #     print(tube)
    #     plan_xy = X_pred[:, :2]                     # (N+1, 2)
    #     lower = plan_xy - tube[:, :2]
    #     upper = plan_xy + tube[:, :2]

    #     plans_xy.append(plan_xy)
    #     lowers_xy.append(lower)
    #     uppers_xy.append(upper)
        
    #     jax.debug.print("u = {}", u0)
    #     dt_run = end - start
    #     total_time += dt_run
    #     min_time = min(min_time, dt_run)

    #     # apply
    #     key, x, w = dubins_step_with_disturbance(key, x, u0, E_sim, dt)
    #     # x = dubins_step(x, u0, dt)
    #     # key, x_disturb = dubins_step_with_disturbance(key, x, u0, E_sim, dt)
    #     jax.debug.print("x = {} y = {}", x[0], x[1])
    #     # jax.debug.print("x_disturb = {} y_disturb = {}", x_disturb[0], x_disturb[1])
    #     jax.debug.print("Distance to obstacle {}", ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5)
    #     # if ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5 < radii[0]:
    #     if ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5 < radii[0]:
    #         jax.debug.print("Crashed!")
    #         # break
    #     xs.append(x)
    #     us.append(u0)

    jax.debug.print("Warmup complete.")
    start = time.perf_counter()
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(x0=x, reference=reference, parameter=parameter)
    u0.block_until_ready()
    end = time.perf_counter()
    jax.debug.print("Nominal trajectory done")
    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=0,
        rho_max=1e10,
        max_iterations=800,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=10,
        sls_primal_tol=1e-2,
        enable_fastsls=True
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations = 1,
    )
    controller = GenericMPCControllerWrapper(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
        X_in=X_pred,
        U_in=U_pred,
    )
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(x0=x, reference=reference, parameter=parameter)
    
    disturbance_history = []
    disturbed_distance = []
    disturbed_distance_y = []
    tube_size = []
    tube_size_y = []
    for k in range(T_steps):
        print(f"sim iteration {k}")
        tube = get_trajectory_tubes(Phi_x)
        tube_size.append(tube[k + 1, 0])
        tube_size_y.append(tube[k + 1, 1])
        plan_xy = X_pred[:, :2]                     # (N+1, 2)
        lower = plan_xy - tube[:, :2]
        upper = plan_xy + tube[:, :2]

        plans_xy.append(plan_xy)
        lowers_xy.append(lower)
        uppers_xy.append(upper)
        
        dt_run = end - start
        total_time += dt_run
        min_time = min(min_time, dt_run)
        disturbance_feedback = jnp.zeros((nu,))
        # for j in range(k):
        #     disturbance_feedback += Phi_u[k, j] @ disturbance_history[j]
        u0 = U_pred[k] + disturbance_feedback
        # apply
        key, x, w = dubins_step_with_disturbance(key, x, u0, E_sim, dt)
        disturbed_distance.append(abs(X_pred[k + 1, 0] - x[0]))
        disturbed_distance_y.append(abs(X_pred[k + 1, 1] - x[1]))
        disturbance_history.append(w)
        # x = dubins_step(x, u0, dt)
        # key, x_disturb = dubins_step_with_disturbance(key, x, u0, E_sim, dt)
        jax.debug.print("x = {} y = {}", x[0], x[1])
        # jax.debug.print("x_disturb = {} y_disturb = {}", x_disturb[0], x_disturb[1])
        jax.debug.print("Distance to obstacle {}", ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5)
        if ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5 < radii[0]:
            jax.debug.print("Crashed!")
            # break
        xs.append(x)
        us.append(u0)

    plans_xy = jnp.stack(plans_xy, axis=0)   # (T_steps, N+1, 2)
    lowers_xy = jnp.stack(lowers_xy, axis=0) # (T_steps, N+1, 2)
    uppers_xy = jnp.stack(uppers_xy, axis=0) # (T_steps, N+1, 2)
    xs = jnp.stack(xs, axis=0)               # (T_steps, 3) (post-step states)
    us = jnp.stack(us, axis=0)
    print("Average Time (ms):", total_time / T_steps* 1000)
    print("Min time (ms):", min_time * 1000)
    save_replay(
        xs=xs,                         # note: post-step states (length T_steps)
        centers=centers,
        radii=radii,
        plans_xy=plans_xy,
        lowers_xy=lowers_xy,
        uppers_xy=uppers_xy,
        filename="replay.mp4",
        dt=dt,
        fps=int(round(1.0 / dt)),
        box_stride=1,
    )

    import numpy as np
    import matplotlib.pyplot as plt

    # convert to numpy
    dx_np = np.asarray(jnp.stack(disturbed_distance))
    dy_np = np.asarray(jnp.stack(disturbed_distance_y))
    tube_x_np = np.asarray(jnp.stack(tube_size))
    tube_y_np = np.asarray(jnp.stack(tube_size_y))

    t = np.arange(dx_np.shape[0]) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # ---- X direction ----
    ax1.plot(t, dx_np, label="|x - x_nominal|")
    ax1.plot(t, tube_x_np, label="tube size (x)")
    ax1.set_ylabel("meters")
    ax1.set_title("X-direction: Deviation vs Tube Size")
    ax1.grid(True)
    ax1.legend()

    # ---- Y direction ----
    ax2.plot(t, dy_np, label="|y - y_nominal|")
    ax2.plot(t, tube_y_np, label="tube size (y)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("meters")
    ax2.set_title("Y-direction: Deviation vs Tube Size")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        "disturbance_vs_tube_size_xy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


    print("Saved replay to replay.mp4")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def save_replay(
    xs,
    centers,
    radii,
    plans_xy,
    lowers_xy,
    uppers_xy,
    filename: str = "replay.mp4",
    dt: float = 0.1,
    fps: int | None = None,
    box_stride: int = 1,
    margin: float = 0.5,
):
    """
    Dubins replay that shows, per MPC step t:
      - executed trajectory up through x_{t+1}
      - planned trajectory at time t (plans_xy[t])
      - tube rectangles from lowers_xy[t], uppers_xy[t]
      - obstacle circles

    Expected shapes:
      xs        : (n_steps+1, 3) or (n_steps, 3)
      plans_xy  : (n_steps, N+1, 2)
      lowers_xy : (n_steps, N+1, 2)
      uppers_xy : (n_steps, N+1, 2)
      centers   : (K, 2)
      radii     : (K,)
    """
    xs = np.asarray(xs)
    centers = np.asarray(centers)
    radii = np.asarray(radii)

    plans_xy = np.asarray(plans_xy)
    lowers_xy = np.asarray(lowers_xy)
    uppers_xy = np.asarray(uppers_xy)

    n_steps = plans_xy.shape[0]
    if n_steps == 0:
        raise ValueError("plans_xy is empty; nothing to replay.")

    # --- fps / interval ---
    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))

    # --- Determine executed trajectory length convention ---
    # We want to show executed up through x_{t+1} for frame t.
    # If xs is only length n_steps, we clamp.
    xs_len = xs.shape[0]

    # --- Axis limits from executed + plans + tubes ---
    all_px = np.concatenate([
        xs[:, 0].ravel(),
        plans_xy[:, :, 0].ravel(),
        lowers_xy[:, :, 0].ravel(),
        uppers_xy[:, :, 0].ravel(),
        centers[:, 0].ravel() if centers.size else np.array([], dtype=float),
    ])
    all_py = np.concatenate([
        xs[:, 1].ravel(),
        plans_xy[:, :, 1].ravel(),
        lowers_xy[:, :, 1].ravel(),
        uppers_xy[:, :, 1].ravel(),
        centers[:, 1].ravel() if centers.size else np.array([], dtype=float),
    ])

    xmin, xmax = float(all_px.min() - margin), float(all_px.max() + margin)
    ymin, ymax = float(all_py.min() - margin), float(all_py.max() + margin)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dubins MPC Replay: Plans + Tube Boxes")

    # Obstacles
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="red", alpha=0.35))

    # Artists
    executed_line, = ax.plot([], [], lw=2, alpha=0.8, label="Executed (closed-loop)")
    planned_line,  = ax.plot([], [], lw=2, ls="--", alpha=0.9, label="Planned (open-loop)")
    cur_pt = ax.scatter([], [], marker="o", s=50, label="Current state")
    end_pt = ax.scatter([], [], marker="x", s=60, label="End of plan")

    tube_boxes = PatchCollection([], alpha=0.20, match_original=False, label="Tube boxes")
    ax.add_collection(tube_boxes)

    ax.grid(True)
    ax.legend(loc="best")

    title = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left"
    )

    def init():
        executed_line.set_data([], [])
        planned_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        end_pt.set_offsets(np.zeros((0, 2)))
        tube_boxes.set_paths([])
        title.set_text("")
        return executed_line, planned_line, cur_pt, end_pt, tube_boxes, title

    def update(t: int):
        # executed: show through x_{t+1} (post-step), clamped to available xs
        t_next = min(t + 1, xs_len - 1)

        ex_px = xs[: t_next + 1, 0]
        ex_py = xs[: t_next + 1, 1]
        executed_line.set_data(ex_px, ex_py)

        # planned trajectory for frame t
        pl_px = plans_xy[t, :, 0]
        pl_py = plans_xy[t, :, 1]
        planned_line.set_data(pl_px, pl_py)

        # tube rectangles for frame t
        lo_px = lowers_xy[t, :, 0]
        lo_py = lowers_xy[t, :, 1]
        up_px = uppers_xy[t, :, 0]
        up_py = uppers_xy[t, :, 1]

        rects = []
        stride = max(int(box_stride), 1)
        for k in range(0, lo_px.shape[0], stride):
            w = up_px[k] - lo_px[k]
            h = up_py[k] - lo_py[k]
            if not np.isfinite(w) or not np.isfinite(h):
                continue
            if w < 0.0 or h < 0.0:
                continue
            rects.append(Rectangle((lo_px[k], lo_py[k]), w, h))
        tube_boxes.set_paths(rects)

        # current marker at x_{t+1} (post-step)
        cur_pt.set_offsets(np.array([[xs[t_next, 0], xs[t_next, 1]]]))

        # end of plan marker
        end_pt.set_offsets(np.array([[pl_px[-1], pl_py[-1]]]))

        title.set_text(f"MPC step {t}/{n_steps-1} (showing x_{t_next})")
        return executed_line, planned_line, cur_pt, end_pt, tube_boxes, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    ext = filename.lower().split(".")[-1]
    if ext == "mp4":
        if animation.FFMpegWriter.isAvailable():
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(filename, writer=writer, dpi=200)
        else:
            raise RuntimeError(
                "Requested .mp4 but ffmpeg is not available. Install ffmpeg or save as .gif."
            )
    elif ext == "gif":
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
    else:
        raise ValueError(f"Unsupported extension .{ext}. Use .mp4 or .gif.")

    plt.close(fig)

if __name__ == "__main__":
    main()
