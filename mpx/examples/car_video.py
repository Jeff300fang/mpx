"""
dubins_car_mpc_experiment.py

End-to-end experiment: use mpx.utils.generic_mpc_wrapper.GenericMPCControllerWrapper
to control a Dubins car with box constraints on controls.

State:      x = [px, py, theta]
Control:    u = [omega]   (nu=1)

Assumptions (matches your mpc() usage):
  - dynamics(x, u, t, *, parameter=...) returns x_{t+1} (discrete-time)
  - cost(W, reference, x, u, t) returns scalar stage cost
  - constraints(x, u, t) returns g(x,u,t) with g <= 0 (inequality)
  - disturbance(X_prefix) returns E used by get_controller(..., E, eta)
"""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from typing import Any, Callable

import os
import time

import jax
import jax.numpy as jnp
from jax import config

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper
from mpx.utils.mpc_utils import combine_constraints
from mpx.utils.fast_sls_visual import get_trajectory_tubes
import matplotlib as mpl
config.update("jax_enable_x64", True)

# --- Styling palette (muted, readable) ---
PALETTE = {
    "plan":      "#1f77b4",   # blue
    "random":    "#ff7f0e",   # orange
    "adversary": "#d62728",   # red
    "tube_face": "#2ca02c",   # green
    "tube_edge": "#1b7f1b",   # darker green edge
    "obs_face":  "#7f7f7f",   # gray
    "obs_edge":  "#4d4d4d",   # dark gray edge
}

NUM_RANDOM = 1
NUM_ADV = 0
mpl.rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,        # keep False unless you want full LaTeX
})
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.2  # meters (XY distance)

def reached_goal_xy(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dxy = x[:2] - x_goal[:2]
    return (dxy @ dxy) <= (tol * tol)

# -----------------------------
# Angle wrapping
# -----------------------------
def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles elementwise to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

# -----------------------------
# Dubins car dynamics
# x = [px, py, theta], u = [omega]
# -----------------------------
V_CONST = 0.2

@jax.jit
def dubins_step(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    px, py, th = x[0], x[1], x[2]
    om = u[0]

    px_next = px + dt * V_CONST * jnp.cos(th)
    py_next = py + dt * V_CONST * jnp.sin(th)
    th_next = wrap_to_pi(th + dt * om)

    return jnp.array([px_next, py_next, th_next], dtype=x.dtype)

def dubins_step_with_disturbance(
    key: jax.Array,          # PRNGKey
    x: jnp.ndarray,          # (3,)
    u: jnp.ndarray,          # (1,)
    E: jnp.ndarray,          # (3,3)
    dt: float,
    i: int
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    Simulates: x_{k+1} = f(x_k,u_k) + E w,   with ||w||_2 <= 1
    where w is sampled from a unit-ball-ish distribution (plus some deterministic cases).

    Returns (key_next, x_next, w).
    """
    px, py, th = x
    om = u[0]

    # Nominal Dubins step
    px_next = px + dt * V_CONST * jnp.cos(th)
    py_next = py + dt * V_CONST * jnp.sin(th)
    th_next = wrap_to_pi(th + dt * om)
    x_nom = jnp.array([px_next, py_next, th_next], dtype=x.dtype)

    # Stronger disturbance sampling
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (x.shape[0],), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(x.shape[0], dtype=x.dtype)
    a = jnp.asarray(1.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = r * z

    # Optional deterministic set of w's for "adversarial" rollouts
    jax.debug.print("{}", w)
    start = i - NUM_RANDOM + 5
    w = jnp.array([0.0, 1.0, 0.0], dtype=x.dtype)
    # if start == 5:
    #     w = jnp.array([0.0, 1.0, 0.0], dtype=x.dtype)
    # if start == 6:
    #     w = jnp.array([0.0, -1.0, 0.0], dtype=x.dtype)
    # if start == 7:
    #     w = jnp.array([1.0, 0.0, 0.0], dtype=x.dtype)
    # if start == 8:
    #     w = jnp.array([-1.0, 0.0, 0.0], dtype=x.dtype)
    # if start == 9:
    #     w = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)
    # if start == 10:
    #     w = jnp.array([0.0, 0.0, -1.0], dtype=x.dtype)
    # if start == 11:
    #     w = jnp.array([0.707, 0.707, 0.0], dtype=x.dtype)
    # if start == 12:
    #     w = jnp.array([-0.707, 0.707, 0.0], dtype=x.dtype)
    # if start == 13:
    #     w = jnp.array([0.707, -0.707, 0.0], dtype=x.dtype)
    # if start == 14:
    #     w = jnp.array([-0.707, -0.707, 0.0], dtype=x.dtype)
    # if start == 15:
    #     w = jnp.array([0.707, 0.0, 0.707], dtype=x.dtype)
    # if start == 16:
    #     w = jnp.array([-0.707, 0.0, 0.707], dtype=x.dtype)
    # if start == 17:
    #     w = jnp.array([0.707, 0.0, -0.707], dtype=x.dtype)
    # if start == 18:
    #     w = jnp.array([-0.707, 0.0, -0.707], dtype=x.dtype)
    # if start == 19:
    #     w = jnp.array([0.0, 0.707, 0.707], dtype=x.dtype)
    # if start == 20:
    #     w = jnp.array([0.0, -0.707, 0.707], dtype=x.dtype)
    # if start == 21:
    #     w = jnp.array([0.0, 0.707, -0.707], dtype=x.dtype)
    # if start == 22:
    #     w = jnp.array([0.0, -0.707, -0.707], dtype=x.dtype)
    # if start == 23:
    #     w = jnp.array([0.577, 0.577, 0.577], dtype=x.dtype)
    # if start == 24:
    #     w = jnp.array([-0.577, 0.577, 0.577], dtype=x.dtype)
    # if start == 25:
    #     w = jnp.array([0.577, -0.577, 0.577], dtype=x.dtype)
    # if start == 26:
    #     w = jnp.array([0.577, 0.577, -0.577], dtype=x.dtype)
    # if start == 27:
    #     w = jnp.array([-0.577, -0.577, 0.577], dtype=x.dtype)
    # if start == 28:
    #     w = jnp.array([0.577, -0.577, -0.577], dtype=x.dtype)
    # if start == 29:
    #     w = jnp.array([-0.577, 0.577, -0.577], dtype=x.dtype)
    # if start == 30:
    #     w = jnp.array([-0.577, -0.577, -0.577], dtype=x.dtype)

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
    W = [wx, wy, wtheta, womega]
    """
    wx, wy, wtheta, womega = W
    xref = reference[t]

    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    dth = x[2] - xref[2]
    theta_cost = 1 - jnp.cos(dth)

    om = u[0]

    return (
        wx * (dx * dx)
        + wy * (dy * dy)
        + wtheta * theta_cost
        + womega * (om * om)
    )

def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray
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

def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for state bounds:
      x - x_max <= 0
      x_min - x <= 0
    """
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)

    return constraints

def make_constant_disturbance(
    n: int,
    alpha: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, n),
    where E[t] = alpha * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)  # (n, n)
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

# -----------------------------
# Visualization helpers
# -----------------------------
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

    xmin, xmax = float(np.nanmin(all_px) - margin), float(np.nanmax(all_px) + margin)
    ymin, ymax = float(np.nanmin(all_py) - margin), float(np.nanmax(all_py) + margin)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dubins MPC Replay: Plans + Tube Boxes")

    # Obstacles
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="tab:red", alpha=0.35))

    # Artists
    executed_line, = ax.plot([], [], lw=2, alpha=0.8, label="Executed (closed-loop)")
    planned_line,  = ax.plot([], [], lw=2, ls="--", alpha=0.9, label="Planned (open-loop)")
    cur_pt = ax.scatter([], [], marker="o", s=50, label="Current state")
    end_pt = ax.scatter([], [], marker="x", s=60, label="End of plan")

    tube_boxes = PatchCollection(
        [],
        alpha=0.20,
        match_original=False,
        label="Robust tubes (state uncertainty)"
    )
    ax.add_collection(tube_boxes)

    ax.grid(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.1, -0.35),
        framealpha=0.9,
    )

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        executed_line.set_data([], [])
        planned_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        end_pt.set_offsets(np.zeros((0, 2)))
        tube_boxes.set_paths([])
        title.set_text("")
        return executed_line, planned_line, cur_pt, end_pt, tube_boxes, title

    def update(t: int):
        t_next = min(t + 1, xs_len - 1)

        ex_px = xs[: t_next + 1, 0]
        ex_py = xs[: t_next + 1, 1]
        executed_line.set_data(ex_px, ex_py)

        pl_px = plans_xy[t, :, 0]
        pl_py = plans_xy[t, :, 1]
        planned_line.set_data(pl_px, pl_py)

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

        if np.isfinite(xs[t_next, 0]) and np.isfinite(xs[t_next, 1]):
            cur_pt.set_offsets(np.array([[xs[t_next, 0], xs[t_next, 1]]]))
        else:
            cur_pt.set_offsets(np.zeros((0, 2)))

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
            raise RuntimeError("Requested .mp4 but ffmpeg is not available. Install ffmpeg or save as .gif.")
    elif ext == "gif":
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
    else:
        raise ValueError(f"Unsupported extension .{ext}. Use .mp4 or .gif.")

    plt.close(fig)

def plot_rollouts_tubes_centers(
    xs,
    centers=None,
    radii=None,
    plans_xy=None,
    lowers_xy=None,
    uppers_xy=None,
    step_idx: int | None = 0,
    tube_stride: int = 2,
    tube_alpha: float = 0.15,
    rollout_alpha: float = 0.35,
    show_plan: bool = True,
    margin: float = 0.5,
    filename: str | None = "rollouts_tubes_centers.png",
    dpi: int = 300,
):
    """
    Static plot with obstacle centers, tube rectangles, and all rollout trajectories.

    Expected shapes:
      xs:        (n_rollouts, T, 3) OR (T, 3)
      plans_xy:  (n_steps, N+1, 2)  (optional)
      lowers_xy: (n_steps, N+1, 2)  (optional)
      uppers_xy: (n_steps, N+1, 2)  (optional)
      centers:   (K, 2)             (optional)
      radii:     (K,)               (optional)
    """
    xs = np.asarray(xs)

    # Normalize xs to (n_rollouts, T, 3)
    if xs.ndim == 2 and xs.shape[1] == 3:
        xs = xs[None, :, :]
    elif xs.ndim == 2 and xs.shape[1] != 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected last dim=3.")
    elif xs.ndim == 3 and xs.shape[2] != 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected xs[...,2] to be theta.")
    elif xs.ndim != 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected 2D or 3D array.")

    n_rollouts, T, _ = xs.shape

    if plans_xy is not None:
        plans_xy = np.asarray(plans_xy)
    if lowers_xy is not None:
        lowers_xy = np.asarray(lowers_xy)
    if uppers_xy is not None:
        uppers_xy = np.asarray(uppers_xy)

    if centers is not None:
        centers = np.asarray(centers)
        if centers.ndim == 1:
            centers = centers[None, :]
    if radii is not None:
        radii = np.asarray(radii).reshape(-1)

    # pick tube/plan frame
    if lowers_xy is not None and uppers_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, lowers_xy.shape[0] - 1))
        lo = lowers_xy[step_idx]
        up = uppers_xy[step_idx]
    else:
        lo = up = None

    # axis limits (use nan-aware because rollouts may be padded with NaN)
    all_x = [xs[:, :, 0].ravel()]
    all_y = [xs[:, :, 1].ravel()]

    if plans_xy is not None:
        all_x.append(plans_xy[:, :, 0].ravel())
        all_y.append(plans_xy[:, :, 1].ravel())
    if lo is not None and up is not None:
        all_x.append(lo[:, 0].ravel())
        all_x.append(up[:, 0].ravel())
        all_y.append(lo[:, 1].ravel())
        all_y.append(up[:, 1].ravel())
    if centers is not None and centers.size:
        all_x.append(centers[:, 0].ravel())
        all_y.append(centers[:, 1].ravel())

    all_x = np.concatenate(all_x) if len(all_x) else np.array([0.0])
    all_y = np.concatenate(all_y) if len(all_y) else np.array([0.0])

    xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
    ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # obstacles
    if centers is not None and centers.size and radii is not None and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), alpha=0.5, color="tab:red"))

    # tubes
    if lo is not None and up is not None:
        for k in range(0, lo.shape[0], max(1, int(tube_stride))):
            w = up[k, 0] - lo[k, 0]
            h = up[k, 1] - lo[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            rect = Rectangle((lo[k, 0], lo[k, 1]), w, h, alpha=tube_alpha)
            ax.add_patch(rect)
        ax.plot([], [], alpha=tube_alpha, label=f"Tube boxes (step {step_idx})")

    # plan
    if show_plan and plans_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, plans_xy.shape[0] - 1))
        ax.plot(
            plans_xy[step_idx, :, 0],
            plans_xy[step_idx, :, 1],
            linestyle="--",
            linewidth=2,
            label="Planned (open-loop)",
        )

    # rollouts (nan-padded -> line breaks automatically)
    for i in range(n_rollouts):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=rollout_alpha, color="tab:orange")
    ax.plot([], [], alpha=rollout_alpha, label=f"Rollouts (n={n_rollouts})")

    ax.set_title("Dubins: Rollouts + Robust Tube + Obstacle Centers")
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Dimensions
    n = 3      # [px, py, theta]
    nu = 1     # [omega]

    # Horizon and dt
    N = 110
    dt = 0.1

    # Weights: (x, y, theta, omega)
    W = jnp.array([25.0, 10.0, 0.01, 0.01], dtype=jnp.float64)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float64),
    )

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e5,
        max_iterations=1000,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=3,
        sls_primal_tol=1e-2,
        enable_fastsls=False
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=1000,
        warm_start=False
    )

    parameter = dt

    om_max = 4.0
    u_min = jnp.array([-om_max], dtype=jnp.float64)
    u_max = jnp.array([om_max], dtype=jnp.float64)

    constraints_u = make_control_box_constraints(u_min, u_max)

    x_max = jnp.array([15.0, 15.0, jnp.inf], dtype=jnp.float64)
    x_min = -x_max
    constraints_x = make_state_box_constraints(x_min, x_max)

    constraints_all = combine_constraints(constraints_x, constraints_u)

    obstacles = jnp.array([[0.0, 0.0, 0.3]], dtype=jnp.float64)
    n_obs = obstacles.shape[0]
    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    nc = 2 * nu + 2 * n + n_obs

    # disturbance model for the controller (E[t] = alpha_sim * I)
    E_mag = 0.025
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

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
        X_in=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64),
    )

    # -----------------------------
    # Initial condition / goal / reference
    # -----------------------------
    x0 = jnp.array([-0.75, -0.75, 0.0], dtype=jnp.float64)
    x_goal = jnp.array([1.0, 0.6, 0.0], dtype=jnp.float64)

    X_ref = jnp.tile(x_goal[None, :], (N + 1, 1))
    reference = X_ref
    T_steps = N

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(3, dtype=jnp.float64)

    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    # -----------------------------
    # Warmup / nominal solve
    # -----------------------------
    jax.debug.print("Warmup complete.")
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    u0.block_until_ready()
    jax.debug.print("Nominal trajectory done")

    # -----------------------------
    # Update configs for robust run
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e6,
        max_iterations=1000,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        warm_start=False,
    )

    sqp_cfg = SQPConfig(
        warm_start=False,
        max_sqp_iterations=50,
    )

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
        X_in=X_pred,
        U_in=U_pred,
    )

    # robust plan (single call in your script)
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )

    tube = get_trajectory_tubes(Phi_x)              # (N+1, n) presumably
    plan_xy = X_pred[:, :2]
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    # -----------------------------
    # Rollout simulations with early stopping
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)  # first k where we stop (exclusive)

    for i in range(N_ROLLOUTS):
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]  # w history, each (n,)
        x = x0

        for k in range(T_steps):
            # stop if within 0.1m of the goal (before applying step k)
            if bool(reached_goal_xy(x, x_goal, GOAL_TOL)):
                stop_steps[i] = k
                break

            print(f"sim iteration {k}")

            # u = U_pred[k] + sum_{j<=k} Phi_u[k,j] w_j
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback

            key, x, w = dubins_step_with_disturbance(key, x, u, E_sim, dt, i)

            # log deviation wrt nominal prediction
            disturbed[i, k, :2] = np.abs(np.asarray(X_pred[k + 1, :2] - x[:2]))
            disturbed[i, k, 2]  = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 2] - x[2])))

            disturbance_history.append(w)
            xs[i, k] = np.asarray(x)

    # -----------------------------
    # Plot rollouts + tube + obstacles
    # -----------------------------
    plot_rollouts_tubes_centers(
        xs=xs,                      # (N_ROLLOUTS, T_steps, 3) (nan-padded after stop)
        centers=np.asarray(centers),
        radii=np.asarray(radii),
        plans_xy=np.asarray(plans_xy),
        lowers_xy=np.asarray(lowers_xy),
        uppers_xy=np.asarray(uppers_xy),
        step_idx=0,
        tube_stride=1,
        filename="rollouts_tubes_centers.png",
        show_plan=False,
        tube_alpha=0.1,
        margin=0.2,
        rollout_alpha=0.5,
    )

    # -----------------------------
    # Deviation vs tube size plots (nan-safe)
    # -----------------------------
    dx_np_all  = disturbed[:, :, 0]
    dy_np_all  = disturbed[:, :, 1]
    dth_np_all = disturbed[:, :, 2]

    tube_x_np  = np.asarray(tube[1:, 0])
    tube_y_np  = np.asarray(tube[1:, 1])
    tube_th_np = np.asarray(tube[1:, 2])

    t = np.arange(dx_np_all.shape[1]) * dt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # ---- X direction ----
    ax1.plot(t, tube_x_np, label="tube size (x)", linewidth=4)
    for r, dx_np in enumerate(dx_np_all):
        m = np.isfinite(dx_np)
        ax1.plot(t[m], dx_np[m], label="|x - x_nominal|" if r == 0 else None)
    ax1.set_ylabel("meters")
    ax1.set_title("X-direction: Deviation vs Tube Size")
    ax1.grid(True)
    ax1.legend()

    # ---- Y direction ----
    ax2.plot(t, tube_y_np, label="tube size (y)", linewidth=4)
    for r, dy_np in enumerate(dy_np_all):
        m = np.isfinite(dy_np)
        ax2.plot(t[m], dy_np[m], label="|y - y_nominal|" if r == 0 else None)
    ax2.set_ylabel("meters")
    ax2.set_title("Y-direction: Deviation vs Tube Size")
    ax2.grid(True)
    ax2.legend()

    # ---- Theta direction ----
    ax3.plot(t, tube_th_np, label="tube size (theta)", linewidth=4)
    for r, dth_np in enumerate(dth_np_all):
        m = np.isfinite(dth_np)
        ax3.plot(t[m], dth_np[m], label="|wrap(theta - theta_nominal)|" if r == 0 else None)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("radians")
    ax3.set_title("Theta-direction: Deviation vs Tube Size")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("disturbance_vs_tube_size_xytheta_dubins.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------
    # Save rollouts + tubes to NPZ (includes stop_steps + goal info)
    # -----------------------------
    out_dir = os.path.join(os.getcwd(), "testing_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "sls_vs_deepreach.npz")

    plans_xy_np  = np.asarray(plans_xy)
    lowers_xy_np = np.asarray(lowers_xy)
    uppers_xy_np = np.asarray(uppers_xy)

    np.savez(
        npz_path,
        xs=np.asarray(xs),                      # (N_ROLLOUTS, T_steps, 3) (nan-padded after stop)
        disturbed=np.asarray(disturbed),        # (N_ROLLOUTS, T_steps, 3) (nan-padded after stop)
        stop_steps=np.asarray(stop_steps),      # (N_ROLLOUTS,)
        goal_tol=float(GOAL_TOL),
        x_goal=np.asarray(x_goal),

        plans_xy=plans_xy_np,
        lowers_xy=lowers_xy_np,
        uppers_xy=uppers_xy_np,
        centers=np.asarray(centers),
        radii=np.asarray(radii),
        obstacles=np.asarray(obstacles),

        dt=float(dt),
        N=int(N),
        T_steps=int(T_steps),
        V_CONST=float(V_CONST),
        E_mag=float(E_mag),
        alpha_sim=float(alpha_sim),
        num_random=int(NUM_RANDOM),
        num_adv=int(NUM_ADV),
        seed=0,
    )

    print(f"[Saved] {npz_path}")
    i = 0  # pick the rollout you want
    save_single_rollout_mp4(
        x_rollout=xs[i],                 # (T_steps,3) with NaNs after stop
        filename="disturbed_rollout_0.mp4",
        dt=dt,
        centers=np.asarray(centers),
        radii=np.asarray(radii),
        plan_xy=np.asarray(plans_xy[0]) if len(plans_xy) else None,
        lower_xy=np.asarray(lowers_xy[0]) if len(lowers_xy) else None,
        upper_xy=np.asarray(uppers_xy[0]) if len(uppers_xy) else None,
        tube_stride=1,
        show_plan=False,                 # you said “rollouts of the one disturbed trajectory”
        show_tubes=True,
    )

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

def save_single_rollout_mp4(
    x_rollout,                 # (T,3) with NaNs after stop OR (T,3) fully finite
    filename="disturbed_rollout.mp4",
    dt=0.1,
    fps=None,
    centers=None,              # (K,2) optional
    radii=None,                # (K,)  optional
    plan_xy=None,              # (N+1,2) optional
    lower_xy=None,             # (N+1,2) optional (tube lower)
    upper_xy=None,             # (N+1,2) optional (tube upper)
    tube_stride=2,
    margin=0.5,
    show_plan=True,
    show_tubes=True,
):
    """
    Saves an MP4 showing a *single* disturbed rollout over time.

    - Draws the rollout trajectory progressively.
    - Optionally draws obstacles, a fixed planned trajectory, and fixed tube rectangles
      (e.g., the step-0 tubes you computed).
    - Handles NaN-padded rollouts: animation stops at first NaN.

    Requires ffmpeg for MP4 (matplotlib uses FFMpegWriter).
    """
    x_rollout = np.asarray(x_rollout, dtype=float)
    if x_rollout.ndim != 2 or x_rollout.shape[1] != 3:
        raise ValueError(f"x_rollout must have shape (T,3). Got {x_rollout.shape}.")

    # Determine how many frames are valid (stop at first NaN in position)
    valid = np.isfinite(x_rollout[:, 0]) & np.isfinite(x_rollout[:, 1])
    T_valid = int(np.argmax(~valid)) if np.any(~valid) else x_rollout.shape[0]
    if T_valid <= 1:
        raise ValueError("Rollout has <=1 valid state; nothing to animate.")

    # Default FPS
    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))

    # Collect data for axis limits
    all_x = [x_rollout[:T_valid, 0]]
    all_y = [x_rollout[:T_valid, 1]]
    if plan_xy is not None:
        plan_xy = np.asarray(plan_xy, dtype=float)
        all_x.append(plan_xy[:, 0])
        all_y.append(plan_xy[:, 1])
    if show_tubes and lower_xy is not None and upper_xy is not None:
        lower_xy = np.asarray(lower_xy, dtype=float)
        upper_xy = np.asarray(upper_xy, dtype=float)
        all_x.append(lower_xy[:, 0]); all_x.append(upper_xy[:, 0])
        all_y.append(lower_xy[:, 1]); all_y.append(upper_xy[:, 1])
    if centers is not None and radii is not None:
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if centers.size:
            all_x.append(centers[:, 0])
            all_y.append(centers[:, 1])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
    ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)

    # Figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Obstacles
    if centers is not None and radii is not None and centers.size and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="tab:red", alpha=0.35))

    # Fixed plan
    if show_plan and plan_xy is not None:
        ax.plot(plan_xy[:, 0], plan_xy[:, 1], linestyle="--", linewidth=2, label="Plan")

    # Fixed tube boxes (e.g., step 0)
    tube_patches = []
    if show_tubes and (lower_xy is not None) and (upper_xy is not None):
        stride = max(1, int(tube_stride))
        for k in range(0, lower_xy.shape[0], stride):
            w = upper_xy[k, 0] - lower_xy[k, 0]
            h = upper_xy[k, 1] - lower_xy[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            tube_patches.append(Rectangle((lower_xy[k, 0], lower_xy[k, 1]), w, h, alpha=0.15))
        for p in tube_patches:
            ax.add_patch(p)
        if tube_patches:
            ax.plot([], [], alpha=0.15, label="Tube boxes")

    # Animated artists
    rollout_line, = ax.plot([], [], linewidth=2, label="Disturbed rollout")
    cur_pt = ax.scatter([], [], s=50, marker="o", label="Current state")
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.legend(loc="best", framealpha=0.9)

    def init():
        rollout_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        title.set_text("")
        return rollout_line, cur_pt, title

    def update(t):
        px = x_rollout[: t + 1, 0]
        py = x_rollout[: t + 1, 1]
        rollout_line.set_data(px, py)

        cur_pt.set_offsets(np.array([[x_rollout[t, 0], x_rollout[t, 1]]]))
        # title.set_text(f"t = {t} / {T_valid-1}  (time = {t*dt:.2f}s)")
        return rollout_line, cur_pt, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T_valid,
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if not animation.FFMpegWriter.isAvailable():
        plt.close(fig)
        raise RuntimeError("ffmpeg is not available; cannot save MP4. Install ffmpeg or save as .gif.")

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
