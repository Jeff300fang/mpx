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
config.update("jax_enable_x64", False)

from dataclasses import dataclass
from typing import Any, Callable

import time
import jax
import jax.numpy as jnp

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper
from mpx.utils.mpc_utils import outside_circle_constraints, combine_constraints


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
    w = r * z  # ||w||_2 <= 1
    # w = jnp.array([0.2, 0.96, 0])

    # Additive disturbance
    x_next = x_nom + E @ w
    return key, x_next


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
    # dth = wrap_to_pi(x[2] - xref[2])

    dth = jnp.arctan2(
        jnp.sin(x[2] - xref[2]),
        jnp.cos(x[2] - xref[2]),
    )


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
        E0 = E0.at[2, 2].set(0.0)
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


# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Dimensions
    n = 3      # [px, py, theta]
    nu = 2     # [v, omega]

    # Horizon and dt
    N = 50
    dt = 0.025
    # N = 100
    # dt = 0.01

    # Weights: (x, y, theta, v, omega)
    W = jnp.array([5.0, 0.3, 0.1, 0.1, 0.1])

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,)),
    )

    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=5e-2,
        rho_max=1e3,
        max_iterations=400,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2
    )

    parameter = dt

    v_max = 5.0
    om_max = 2.5

    u_min = jnp.array([-v_max, -om_max])
    u_max = jnp.array([v_max,  om_max])

    constraints_u = make_control_box_constraints(u_min, u_max)
    centers = jnp.array([[1.0, 0.08]])   # (K,2)
    radii   = jnp.array([0.15])         # (K,)
    K = centers.shape[0]

    # Inflate a bit if you want “safety margin”
    obstacle_constraints = partial(
        outside_circle_constraints,
        centers=centers,
        radii=radii,
    )  # returns (K,)

    # Combine: first control bounds, then obstacles
    constraints_all = combine_constraints(constraints_u, obstacle_constraints)

    # Total constraint count:
    # nc = 2 * nu 
    nc = 2 * nu + K

    # disturbance = make_zero_disturbance(n=n)
    alpha_sim = 0.001
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    controller = GenericMPCControllerWrapper(
        sls_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
    )

    # -----------------------------
    # Initial condition
    # -----------------------------
    x = jnp.array([0.0, 0.0, 0.0])
    X_ref, U_ref = build_forward_reference(x, N, dt)
    reference = X_ref
    # Closed-loop rollout
    T_steps = min(int(5.0 / dt), 100)  # simulate ~5s (cap)
    xs = []
    us = []

    total_time = 0.0
    min_time = float("inf")

    # Compilation warmup
    _ = controller.run(x0=x, reference=reference, parameter=parameter)

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(3)
    E_sim = E_sim.at[2, 2].set(0)

    for k in range(T_steps):
        print(f"sim iteration {k}")
        X_ref, U_ref = build_forward_reference(x, N, dt)
        reference = X_ref  # your cost() expects (N+1, 3)
        start = time.perf_counter()
        u0, X_pred, U_pred, V_pred = controller.run(x0=x, reference=reference, parameter=parameter)
        u0.block_until_ready()
        u0 = jnp.clip(u0, u_min * 2, u_max * 2)
        end = time.perf_counter()
        jax.debug.print("u = {}", u0)
        dt_run = end - start
        total_time += dt_run
        min_time = min(min_time, dt_run)

        # apply
        key, x = dubins_step_with_disturbance(key, x, u0, E_sim, dt)
        # x = dubins_step(x, u0, dt)
        # key, x_disturb = dubins_step_with_disturbance(key, x, u0, E_sim, dt)
        jax.debug.print("x = {} y = {}", x[0], x[1])
        # jax.debug.print("x_disturb = {} y_disturb = {}", x_disturb[0], x_disturb[1])
        jax.debug.print("Distance to obstacle {}", ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5)
        if ((x[0] - centers[0][0]) ** 2 + (x[1] - centers[0][1]) ** 2) ** 0.5 < radii[0]:
            jax.debug.print("Crashed!")
            break
        xs.append(x)
        us.append(u0)

    xs = jnp.stack(xs, axis=0)
    us = jnp.stack(us, axis=0)
    print("Average Time (ms):", total_time / T_steps* 1000)
    print("Min time (ms):", min_time * 1000)
    save_replay(
        xs=xs,
        centers=centers,
        radii=radii,
        filename="replay.mp4",
        dt=dt,
    )

    print("Saved replay to replay.mp4")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def save_replay(
    xs,
    centers,
    radii,
    filename: str = "replay.mp4",
    dt: float = 0.1,
):
    xs = np.asarray(xs)
    centers = np.asarray(centers)
    radii = np.asarray(radii)

    T = xs.shape[0]
    if T == 0:
        raise ValueError("xs is empty; nothing to replay.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="red", alpha=0.4))

    margin = 0.5
    xmin, xmax = xs[:, 0].min() - margin, xs[:, 0].max() + margin
    ymin, ymax = xs[:, 1].min() - margin, xs[:, 1].max() + margin
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))

    traj_line, = ax.plot([], [], "b-", lw=2, alpha=0.7)
    car_dot, = ax.plot([], [], "bo", markersize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dubins Car MPC Replay")

    def init():
        traj_line.set_data([], [])
        car_dot.set_data([], [])
        return traj_line, car_dot

    def update(k: int):
        traj_line.set_data(xs[:k+1, 0], xs[:k+1, 1])
        car_dot.set_data([xs[k, 0]], [xs[k, 1]])  # sequences
        return traj_line, car_dot

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init, interval=dt * 1000.0, blit=True
    )

    fps = max(1, int(round(1.0 / dt)))

    # --- Writer selection ---
    ext = filename.lower().split(".")[-1]

    if ext == "mp4":
        # Force ffmpeg writer if available
        if animation.FFMpegWriter.isAvailable():
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(filename, writer=writer)
        else:
            raise RuntimeError(
                "Requested .mp4 but Matplotlib cannot find ffmpeg. "
                "Install ffmpeg or save as .gif instead."
            )
    elif ext == "gif":
        # Pillow works for GIF
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
    else:
        raise ValueError(f"Unsupported extension .{ext}. Use .mp4 or .gif.")

    plt.close(fig)


if __name__ == "__main__":
    main()
