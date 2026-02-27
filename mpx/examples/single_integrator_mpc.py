from __future__ import annotations

# =========================
# Full script: Receding-horizon MPC + MP4
# =========================

from jax import config
config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse

import jax
import jax.numpy as jnp

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper
from mpx.utils.mpc_utils import combine_constraints
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig


# -------------------------
# Config / dataclasses
# -------------------------
@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray


# -------------------------
# Constraints
# -------------------------
def make_control_box_constraints(
    u_min: jnp.ndarray, u_max: jnp.ndarray
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
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


def make_circle_constraints(xc, yc, r):
    # NOTE: This is "outside circle" as g(x) = r^2 - ||x-c||^2 <= 0
    def constraint(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([r**2 - (x[0] - xc) ** 2 - (x[1] - yc) ** 2], dtype=x.dtype)

    return constraint


def make_ellipsoid_constraints(
    xc: float,
    yc: float,
    a: float,
    b: float,
    *,
    theta: float = 0.0,
):
    """
    Inequality constraint g(x) <= 0 representing OUTSIDE an ellipsoid.

    Axis-aligned (theta=0):
        g(x) = 1 - ((x-xc)/a)^2 - ((y-yc)/b)^2

    If theta != 0, we rotate coordinates by theta (radians) before evaluating.
    """
    a2 = a * a
    b2 = b * b
    cth = jnp.cos(theta)
    sth = jnp.sin(theta)

    def constraint(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        dx = x[0] - xc
        dy = x[1] - yc

        # rotate into ellipsoid frame
        xrp = cth * dx + sth * dy
        yrp = -sth * dx + cth * dy

        val = 1.0 - (xrp * xrp) / a2 - (yrp * yrp) / b2
        return jnp.array([val], dtype=x.dtype)

    return constraint


# -------------------------
# Dynamics / Cost
# -------------------------
def dynamics(x, u, t, *, parameter):
    # single integrator
    dt = parameter
    px, py = x
    vx, vy = u
    px_next = px + dt * vx
    py_next = py + dt * vy
    return jnp.array([px_next, py_next], dtype=x.dtype)


def cost(W, reference, x, u, t):
    wx, wy, wvx, wvy = W
    xref = reference[t]
    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    vx, vy = u
    return wx * dx * dx + wy * dy * dy + wvx * vx * vx + wvy * vy * vy


# -------------------------
# Disturbance (optional)
# -------------------------
def make_constant_disturbance(n: int, alpha: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, n),
    where E[t] = alpha * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)  # (n, n)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance


# -------------------------
# Reference generation
# -------------------------
def make_straight_line_reference(
    x0: jnp.ndarray,
    x_goal: jnp.ndarray,
    N: int,
    dt: float,
    v_des: float,
) -> jnp.ndarray:
    """
    Straight-line reference from x0 to x_goal at constant speed v_des (>=0).
    Returns X_ref with shape (N+1, 2). Once goal is reached, it stays there.
    """
    x0 = jnp.asarray(x0)
    x_goal = jnp.asarray(x_goal)

    d = x_goal - x0
    dist = jnp.linalg.norm(d)
    dir_vec = jnp.where(dist > 1e-12, d / dist, jnp.zeros_like(d))

    t = jnp.arange(N + 1, dtype=x0.dtype) * dt
    s = jnp.minimum(v_des * t, dist)  # clip so we don't pass the goal

    X_ref = x0[None, :] + s[:, None] * dir_vec[None, :]
    return X_ref


def get_reference_window(reference_full: jnp.ndarray, k: int, N: int) -> jnp.ndarray:
    """
    Returns (N+1,2) reference for the horizon starting at time k.
    Pads with the last point once we run past the end.
    """
    Tfull = reference_full.shape[0]
    idx = jnp.arange(k, k + N + 1)
    idx = jnp.clip(idx, 0, Tfull - 1)
    return reference_full[idx]


# -------------------------
# Nominal projection init
# -------------------------
def project_strictly_outside_ellipsoid(
    X: jnp.ndarray,  # (T,2)
    *,
    xc: float,
    yc: float,
    a: float,
    b: float,
    theta: float = 0.0,
    eps: float = 1e-3,          # strict margin: s >= 1+eps
    keep_endpoints: bool = True # keep X[0], X[-1] unchanged if desired
) -> jnp.ndarray:
    """
    Projects points that violate s(x) >= 1+eps to the (1+eps)-level set of the ellipsoid,
    by scaling in the rotated ellipsoid frame (radial projection in that metric).
    """
    a2 = a * a
    b2 = b * b
    cth = jnp.cos(theta)
    sth = jnp.sin(theta)

    def proj_one(x):
        dx = x[0] - xc
        dy = x[1] - yc

        # rotate into ellipsoid frame
        xrp = cth * dx + sth * dy
        yrp = -sth * dx + cth * dy

        s = (xrp * xrp) / a2 + (yrp * yrp) / b2  # want >= 1+eps
        s_safe = jnp.maximum(s, 1e-12)

        scale = jnp.sqrt((1.0 + eps) / s_safe)
        do_scale = s < (1.0 + eps)

        xrp2 = jnp.where(do_scale, xrp * scale, xrp)
        yrp2 = jnp.where(do_scale, yrp * scale, yrp)

        # rotate back
        dx2 = cth * xrp2 - sth * yrp2
        dy2 = sth * xrp2 + cth * yrp2

        return jnp.array([xc + dx2, yc + dy2], dtype=x.dtype)

    Xp = jax.vmap(proj_one)(X)

    if keep_endpoints and X.shape[0] >= 2:
        Xp = Xp.at[0].set(X[0])
        Xp = Xp.at[-1].set(X[-1])
    return Xp


# -------------------------
# Warm-start shifting
# -------------------------
def shift_warm_start(X_pred: jnp.ndarray, U_pred: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shift predicted traj/controls forward by 1 step to use as next initial guess.
    """
    X_next = jnp.concatenate([X_pred[1:], X_pred[-1:]], axis=0)  # (N+1,2)
    U_next = jnp.concatenate([U_pred[1:], U_pred[-1:]], axis=0)  # (N,2)
    return X_next, U_next


# -------------------------
# Simulation step (closed-loop)
# -------------------------
def sim_step(x: jnp.ndarray, u: jnp.ndarray, k: int, dt: float, sigma: float = 0.0) -> jnp.ndarray:
    """
    One closed-loop simulation step using the same dynamics; optional additive Gaussian noise.
    """
    xnext = dynamics(x, u, k, parameter=dt)
    if sigma > 0.0:
        w = jnp.asarray(np.random.randn(*x.shape)) * sigma
        xnext = xnext + w
    return xnext


# -------------------------
# Main
# -------------------------
def main():
    # --- Define Cost ---
    W = jnp.array([30.0, 30.0, 1.0, 1.0])

    nx = 2
    nu = 2
    N = 110
    dt = 0.1

    cfg = MPCConfig(
        n=nx,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,)),
    )

    admm_cfg = ADMMConfig(
        eps_abs=1e-3,
        eps_rel=0,
        rho_max=1e2,
        max_iterations=2000,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=3,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        warm_start=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=15,
        warm_start=True,
        line_search=True,
    )

    # --- Constraints ---
    obstacles = jnp.array([])

    u_min = jnp.array([-1.0, -1.0])
    u_max = jnp.array([1.0, 1.0])
    constraints_u = make_control_box_constraints(u_min, u_max)

    ellipsoid_constraint = make_ellipsoid_constraints(
        xc=0.0, yc=0.0,
        a=0.45, b=0.25,
        theta=jnp.deg2rad(90.0),
    )

    constraints_all = combine_constraints(constraints_u, ellipsoid_constraint)
    nc = 2 * nu + 1

    # disturbance for your solver internals (as in your snippet)
    E_mag = 0.075
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=nx, alpha=alpha_sim)

    # --- Initial condition and goal ---
    x0 = jnp.array([-0.75, -0.75])
    x_goal = jnp.array([0.7, 1.0])

    # --- Build a longer reference so windowing makes sense in receding horizon ---
    # You can make this longer than N+1; otherwise the window will just clamp to the last point quickly.
    T_ref = 400
    v_des = 0.6
    reference_full = make_straight_line_reference(x0=x0, x_goal=x_goal, N=N, dt=dt, v_des=v_des)

    # --- Nominal init (project reference to be strictly outside ellipsoid) ---
    theta = float(jnp.deg2rad(90.0))
    X_in = project_strictly_outside_ellipsoid(
        reference_full[: (N + 1)],
        xc=0.0, yc=0.0,
        a=0.45, b=0.25,
        theta=theta,
        eps=1e-3,
        keep_endpoints=False,
    )

    # --- Build controller ---
    controller = GenericMPCControllerWrapper(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
        X_in=X_in.copy(),
        U_in=jnp.zeros((cfg.N, cfg.nu)),
    )

    # =========================
    # Closed-loop receding-horizon MPC
    # =========================
    T_sim = 8         # number of MPC steps to execute
    dist_sigma = 0.0    # set e.g. 0.002 to see robustness effects

    xk = jnp.array(x0, dtype=jnp.float64)

    X_warm = X_in.copy()
    U_warm = jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64)

    X_exec = [np.asarray(xk)]
    U_exec = []
    X_preds = []  # predicted horizon each MPC iteration

    for k in range(T_sim):
        ref_k = get_reference_window(reference_full, k, cfg.N)

        # Try to feed warm starts back into the controller if attributes exist.
        # (If your wrapper ignores these, it still works.)
        if hasattr(controller, "X_in"):
            controller.X_in = X_warm
        if hasattr(controller, "U_in"):
            controller.U_in = U_warm

        u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
            x0=xk,
            reference=ref_k,
            parameter=dt,
        )

        X_preds.append(np.asarray(X_pred))

        # apply first control and step
        uk = u0
        xk = sim_step(xk, uk, k, dt=float(dt), sigma=dist_sigma)

        U_exec.append(np.asarray(uk))
        X_exec.append(np.asarray(xk))

        # shift warm start
        X_warm, U_warm = shift_warm_start(X_pred, U_pred)

    X_exec = np.asarray(X_exec)  # (T_sim+1,2)
    U_exec = np.asarray(U_exec)  # (T_sim,2)

    print("Closed-loop rollout done.")
    print("Final state:", X_exec[-1])

    # =========================
    # Animation to MP4
    # =========================
    mp4_path = "mpc_rollout.mp4"

    fig, ax = plt.subplots(figsize=(6, 6))

    # Ellipsoid patches
    ell_fill = Ellipse(
        (0.0, 0.0),
        width=2 * 0.45,
        height=2 * 0.25,
        angle=np.rad2deg(np.deg2rad(90.0)),
        fill=True,
        alpha=0.15,
    )
    ell_bd = Ellipse(
        (0.0, 0.0),
        width=2 * 0.45,
        height=2 * 0.25,
        angle=np.rad2deg(np.deg2rad(90.0)),
        fill=False,
        linewidth=2,
    )
    ax.add_patch(ell_fill)
    ax.add_patch(ell_bd)

    # Static full reference
    ref_np = np.asarray(reference_full)
    ax.plot(ref_np[:, 0], ref_np[:, 1], "--", linewidth=2, label="reference (full)")

    # dynamic lines
    (exec_line,) = ax.plot([], [], "-", linewidth=2, label="executed")
    (pred_line,) = ax.plot([], [], "-", linewidth=2, label="predicted (current solve)")

    # start/goal
    ax.scatter([X_exec[0, 0]], [X_exec[0, 1]], s=60, marker="o", label="start")
    ax.scatter([ref_np[-1, 0]], [ref_np[-1, 1]], s=60, marker="*", label="goal")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    # view limits
    pad = 0.25
    xmin = min(ref_np[:, 0].min(), X_exec[:, 0].min(), -0.45) - pad
    xmax = max(ref_np[:, 0].max(), X_exec[:, 0].max(),  0.45) + pad
    ymin = min(ref_np[:, 1].min(), X_exec[:, 1].min(), -0.25) - pad
    ymax = max(ref_np[:, 1].max(), X_exec[:, 1].max(),  0.25) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def init():
        exec_line.set_data([], [])
        pred_line.set_data([], [])
        ax.set_title("Receding-horizon MPC")
        return exec_line, pred_line

    def update(frame):
        # frame: 0..T_sim-1
        xe = X_exec[: frame + 2]
        exec_line.set_data(xe[:, 0], xe[:, 1])

        xp = X_preds[frame]
        pred_line.set_data(xp[:, 0], xp[:, 1])

        ax.set_title(f"Receding-horizon MPC | step {frame+1}/{T_sim}")
        return exec_line, pred_line

    anim = FuncAnimation(fig, update, frames=T_sim, init_func=init, blit=True, interval=50)

    # Requires ffmpeg installed. On Ubuntu: sudo apt-get install ffmpeg
    writer = FFMpegWriter(fps=20, bitrate=1800)
    anim.save(mp4_path, writer=writer)

    plt.close(fig)
    print("Saved:", mp4_path)


if __name__ == "__main__":
    main()