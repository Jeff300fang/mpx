#!/usr/bin/env python3
"""
dubins_car_mpc_experiment_multi_rollouts.py

End-to-end experiment: use mpx.utils.generic_mpc_wrapper.GenericMPCControllerWrapper
to control a Dubins car with box constraints on controls.

State:      x = [px, py, theta]
Control:    u = [omega]   (v is constant V_CONST in dynamics)

Assumptions (matches your mpc() usage):
  - dynamics(x, u, t, *, parameter=...) returns x_{t+1} (discrete-time)
  - cost(W, reference, x, u, t) returns scalar stage cost
  - constraints(x, u, t) returns g(x,u,t) with g <= 0 (inequality)
  - disturbance(X_prefix) returns E used by get_controller(..., E, eta)

This version:
  - Uses your stronger disturbance model with per-rollout index i.
  - Runs a set of rollouts (NUM_RANDOM random + deterministic adversarial cases).
  - Visualizes ALL rolled out trajectories together in one video.
  - Sets sprite alpha to 0.8.
"""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from typing import Callable

import time

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper
from mpx.utils.mpc_utils import combine_constraints
from mpx.utils.fast_sls_visual import get_trajectory_tubes
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig


# =============================================================================
# Style
# =============================================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


# =============================================================================
# Disturbance rollout settings (controls deterministic vs random mix)
# =============================================================================
NUM_RANDOM = 5          # number of purely random rollouts first
NUM_DETERMINISTIC = 26   # matches deterministic cases 5..30 below
NUM_ROLLOUTS = NUM_RANDOM + NUM_DETERMINISTIC


# =============================================================================
# Angle wrapping
# =============================================================================
def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles elementwise to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# =============================================================================
# Dubins car dynamics
# x = [px, py, theta], u = [omega]
# =============================================================================
V_CONST = 2.0  # constant forward speed


@jax.jit
def dubins_step(
    x: jnp.ndarray,
    u: jnp.ndarray,   # shape (1,) = [omega]
    dt: float,
) -> jnp.ndarray:
    px, py, th = x
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
    i: int,                  # rollout index
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
    start = i - NUM_RANDOM + 5
    if start == 5:
        w = jnp.array([0.0, 1.0, 0.0], dtype=x.dtype)
    if start == 6:
        w = jnp.array([0.0, -1.0, 0.0], dtype=x.dtype)
    if start == 7:
        w = jnp.array([1.0, 0.0, 0.0], dtype=x.dtype)
    if start == 8:
        w = jnp.array([-1.0, 0.0, 0.0], dtype=x.dtype)
    if start == 9:
        w = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)
    if start == 10:
        w = jnp.array([0.0, 0.0, -1.0], dtype=x.dtype)
    if start == 11:
        w = jnp.array([0.707, 0.707, 0.0], dtype=x.dtype)
    if start == 12:
        w = jnp.array([-0.707, 0.707, 0.0], dtype=x.dtype)
    if start == 13:
        w = jnp.array([0.707, -0.707, 0.0], dtype=x.dtype)
    if start == 14:
        w = jnp.array([-0.707, -0.707, 0.0], dtype=x.dtype)
    if start == 15:
        w = jnp.array([0.707, 0.0, 0.707], dtype=x.dtype)
    if start == 16:
        w = jnp.array([-0.707, 0.0, 0.707], dtype=x.dtype)
    if start == 17:
        w = jnp.array([0.707, 0.0, -0.707], dtype=x.dtype)
    if start == 18:
        w = jnp.array([-0.707, 0.0, -0.707], dtype=x.dtype)
    if start == 19:
        w = jnp.array([0.0, 0.707, 0.707], dtype=x.dtype)
    if start == 20:
        w = jnp.array([0.0, -0.707, 0.707], dtype=x.dtype)
    if start == 21:
        w = jnp.array([0.0, 0.707, -0.707], dtype=x.dtype)
    if start == 22:
        w = jnp.array([0.0, -0.707, -0.707], dtype=x.dtype)
    if start == 23:
        w = jnp.array([0.577, 0.577, 0.577], dtype=x.dtype)
    if start == 24:
        w = jnp.array([-0.577, 0.577, 0.577], dtype=x.dtype)
    if start == 25:
        w = jnp.array([0.577, -0.577, 0.577], dtype=x.dtype)
    if start == 26:
        w = jnp.array([0.577, 0.577, -0.577], dtype=x.dtype)
    if start == 27:
        w = jnp.array([-0.577, -0.577, 0.577], dtype=x.dtype)
    if start == 28:
        w = jnp.array([0.577, -0.577, -0.577], dtype=x.dtype)
    if start == 29:
        w = jnp.array([-0.577, 0.577, -0.577], dtype=x.dtype)
    if start == 30:
        w = jnp.array([-0.577, -0.577, -0.577], dtype=x.dtype)

    x_next = x_nom + E @ w
    return key, x_next, w


def dynamics(x, u, t, *, parameter):
    dt = parameter
    return dubins_step(x, u, dt)


# =============================================================================
# Cost and Constraints matching your mpc() expectations
# =============================================================================
def cost(W, reference, x, u, t):
    wx, wy, wtheta, womega = W
    xref = reference[t]

    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    dth = wrap_to_pi(x[2] - xref[2])

    om = u[0]

    return (
        wx * dx * dx
        + wy * dy * dy
        + wtheta * dth * dth
        + womega * om * om
    )


def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray
):
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
):
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


def make_zero_disturbance(n: int):
    """Returns E(t)=0 with shape (T,n,n)."""
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        return jnp.zeros((T, n, n), dtype=X_prefix.dtype)
    return disturbance


def make_constant_disturbance(
    n: int,
    alpha: float,
):
    """Returns constant E(t)=alpha*I with shape (T,n,n)."""
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)
        return jnp.broadcast_to(E0, (T, n, n))
    return disturbance


# =============================================================================
# Config
# =============================================================================
@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray


@partial(jax.jit, static_argnames=("N",))
def build_forward_reference(x0: jnp.ndarray, N: int, dt: float):
    om_ref = 0.0
    u = jnp.array([om_ref], dtype=x0.dtype)          # (1,)
    U_ref = jnp.broadcast_to(u, (N, 1))              # (N,1)

    def step(x, _):
        x_next = dubins_step(x, u, dt)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, xs=None, length=N)
    X_ref = jnp.concatenate([x0[None, :], xs], axis=0)
    return X_ref, U_ref


# =============================================================================
# Sprite helpers (imshow + Affine2D)
# =============================================================================
def load_sprite_rgba(path: str) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    elif img.shape[-1] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], dtype=img.dtype)])
    return img


def make_sprite_artist(
    ax: plt.Axes,
    img_rgba: np.ndarray,
    x: float, y: float, theta: float,
    length: float,
    width: float,
    alpha: float,
    zorder: int = 6,
    interpolation: str = "nearest",
):
    extent = (-length / 2.0, length / 2.0, -width / 2.0, width / 2.0)
    im = ax.imshow(
        img_rgba,
        extent=extent,
        origin="upper",
        interpolation=interpolation,
        alpha=alpha,
        zorder=zorder,
    )
    tf = Affine2D().rotate(theta).translate(x, y)
    im.set_transform(tf + ax.transData)
    return {"im": im, "tf": tf}


def update_sprite_artist(sprite, x: float, y: float, theta: float) -> None:
    tf: Affine2D = sprite["tf"]
    tf.clear()
    tf.rotate(theta)
    tf.translate(x, y)


def set_sprite_visible(sprite, visible: bool) -> None:
    sprite["im"].set_visible(visible)


# =============================================================================
# Replay (MULTI-ROLLOUT)
# =============================================================================
def save_replay(
    xs,                 # (T,3) or (R,T,3)
    centers,
    radii,
    plans_xy,
    lowers_xy,
    uppers_xy,
    filename: str = "replay_multi.mp4",
    dt: float = 0.1,
    fps: int | None = None,
    box_stride: int = 1,
    margin: float = 0.5,
    # --- sprite args ---
    sprite_path: str | None = None,
    sprite_rgba: np.ndarray | None = None,
    sprite_length: float = 0.18,
    sprite_width: float = 0.09,
    sprite_alpha: float = 0.80,   # requested
):
    """
    Per MPC step t:
      - executed trajectories up through x_{t+1} for ALL rollouts
      - planned trajectory at time t (plans_xy[t])
      - tube rectangles from lowers_xy[t], uppers_xy[t]
      - obstacle circles
      - CURRENT STATE rendered as sprites for ALL rollouts
    """
    xs = np.asarray(xs)
    if xs.ndim == 2:
        xs = xs[None, ...]  # (1,T,3)
    if xs.ndim != 3 or xs.shape[-1] != 3:
        raise ValueError(f"xs must be (T,3) or (R,T,3), got {xs.shape}")

    R, T, _ = xs.shape

    centers = np.asarray(centers)
    radii = np.asarray(radii)
    plans_xy = np.asarray(plans_xy)
    lowers_xy = np.asarray(lowers_xy)
    uppers_xy = np.asarray(uppers_xy)

    n_steps = T
    if n_steps == 0:
        raise ValueError("xs is empty; nothing to replay.")

    # sprite load
    if sprite_rgba is None:
        if sprite_path is None:
            raise ValueError("Provide sprite_path or sprite_rgba.")
        sprite_rgba = load_sprite_rgba(sprite_path)

    # fps / interval
    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))

    # axis limits (ignore NaNs)
    all_px = np.concatenate([
        xs[:, :, 0].ravel(),
        plans_xy[:, :, 0].ravel(),
        lowers_xy[:, :, 0].ravel(),
        uppers_xy[:, :, 0].ravel(),
        centers[:, 0].ravel() if centers.size else np.array([], dtype=float),
    ])
    all_py = np.concatenate([
        xs[:, :, 1].ravel(),
        plans_xy[:, :, 1].ravel(),
        lowers_xy[:, :, 1].ravel(),
        uppers_xy[:, :, 1].ravel(),
        centers[:, 1].ravel() if centers.size else np.array([], dtype=float),
    ])
    all_px = all_px[np.isfinite(all_px)]
    all_py = all_py[np.isfinite(all_py)]

    xmin, xmax = float(all_px.min() - margin), float(all_px.max() + margin)
    ymin, ymax = float(all_py.min() - margin), float(all_py.max() + margin)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # obstacles
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), color="red", alpha=0.35))

    # planned
    planned_line, = ax.plot([], [], lw=2, ls="--", alpha=0.9, label="Planned (open-loop)")
    end_pt = ax.scatter([], [], marker="x", s=60, label="End of plan")

    # executed (one line per rollout)
    executed_lines = []
    for _ in range(R):
        (ln,) = ax.plot([], [], lw=2, alpha=0.55, color="tab:orange")
        executed_lines.append(ln)

    # tube boxes
    tube_boxes = PatchCollection(
        [],
        alpha=0.20,
        match_original=False,
        label="Robust tubes (state uncertainty)",
    )
    ax.add_collection(tube_boxes)

    # sprites (one per rollout)
    cars = []
    for r in range(R):
        idx0 = np.where(np.isfinite(xs[r, :, 0]) & np.isfinite(xs[r, :, 1]) & np.isfinite(xs[r, :, 2]))[0]
        if idx0.size:
            t0 = int(idx0[0])
            x0, y0, th0 = float(xs[r, t0, 0]), float(xs[r, t0, 1]), float(xs[r, t0, 2])
        else:
            x0, y0, th0 = 0.0, 0.0, 0.0

        car = make_sprite_artist(
            ax,
            sprite_rgba,
            x0, y0, th0,
            length=sprite_length,
            width=sprite_width,
            alpha=sprite_alpha,
            zorder=6,
            interpolation="nearest",
        )
        set_sprite_visible(car, False)
        cars.append(car)

    ax.grid(True)
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        for ln in executed_lines:
            ln.set_data([], [])
        planned_line.set_data([], [])
        end_pt.set_offsets(np.zeros((0, 2)))
        tube_boxes.set_paths([])
        title.set_text("")
        for car in cars:
            set_sprite_visible(car, False)
        return (*executed_lines, planned_line, end_pt, tube_boxes, title, *[c["im"] for c in cars])

    def update(t: int):
        t_next = min(t + 1, T - 1)

        # executed trails + sprite pose per rollout
        for r in range(R):
            xr = xs[r, : t_next + 1, 0]
            yr = xs[r, : t_next + 1, 1]
            m = np.isfinite(xr) & np.isfinite(yr)
            executed_lines[r].set_data(xr[m], yr[m])

            if np.isfinite(xs[r, t_next, 0]) and np.isfinite(xs[r, t_next, 1]) and np.isfinite(xs[r, t_next, 2]):
                update_sprite_artist(cars[r], float(xs[r, t_next, 0]), float(xs[r, t_next, 1]), float(xs[r, t_next, 2]))
                set_sprite_visible(cars[r], True)
            else:
                set_sprite_visible(cars[r], False)

        # plan + tube at time t_plan
        t_plan = min(t, plans_xy.shape[0] - 1)
        pl_px = plans_xy[t_plan, :, 0]
        pl_py = plans_xy[t_plan, :, 1]
        planned_line.set_data(pl_px, pl_py)
        end_pt.set_offsets(np.array([[pl_px[-1], pl_py[-1]]]))

        lo_px = lowers_xy[t_plan, :, 0]
        lo_py = lowers_xy[t_plan, :, 1]
        up_px = uppers_xy[t_plan, :, 0]
        up_py = uppers_xy[t_plan, :, 1]

        rects = []
        stride = max(int(box_stride), 1)
        for k in range(0, lo_px.shape[0], stride):
            w = up_px[k] - lo_px[k]
            h = up_py[k] - lo_py[k]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            rects.append(Rectangle((lo_px[k], lo_py[k]), w, h))
        tube_boxes.set_paths(rects)

        # title.set_text(f"t = {t:03d}   rollouts = {R}")

        return (*executed_lines, planned_line, end_pt, tube_boxes, title, *[c["im"] for c in cars])

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
            ani.save(filename, writer=writer, dpi=400)
        else:
            raise RuntimeError("Requested .mp4 but ffmpeg is not available.")
    elif ext == "gif":
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
    else:
        raise ValueError(f"Unsupported extension .{ext}. Use .mp4 or .gif.")

    plt.close(fig)


# =============================================================================
# Main experiment
# =============================================================================
def main():
    # Dimensions
    n = 3      # [px, py, theta]
    nu = 1     # [omega]

    # Horizon and dt
    N = 190
    dt = 0.05

    # Weights: (x, y, theta, omega)
    W = jnp.array([5.0, 10.0, 0.01, 0.1])

    obstacles = jnp.array([
        [1.0, 0.4, 0.2], [2.0, -0.4, 0.2], [4.0, -0.2, 0.2], [0.8, -0.6, 0.2],
        [3.0, 0.7, 0.2], [2.0, 0.5, 0.2], [5.0, -0.3, 0.2], [3.8, 1.0, 0.2],
        [6.0, 0.0, 0.2], [9.0, 0.8, 0.2], [8.0, 1.1, 0.2], [6.2, 1.0, 0.2],
        [12.0, 0.4, 0.2], [11.0, 0.6, 0.2], [13.0, 1.0, 0.2], [15.0, 1.3, 0.2], [16.0, 1.1, 0.2],
        [8.0, -0.5, 0.2], [9.2, -1.2, 0.2], [1.3, -1.4, 0.2], [2.9, -1.0, 0.2], [4.1, -1.2, 0.2], [14.5, -0.2, 0.2],
        [13.0, -0.4, 0.2], [11.0, -1.0, 0.2], [16.0, -0.7, 0.2], [17.5, -0.7, 0.2], [18.0, 1.0, 0.2], [13.7, -1.0, 0.2], [7.0, -0.5, 0.2],
        [5.7, -1.1, 0.2], [10.0, 0.9, 0.2], [7.5, -1.1, 0.2],
        [5.0, 1.2, 0.2], [10.2, -0.9, 0.2], [11.4, 1.3, 0.2],
        [12.0, -0.9, 0.2], [15.0, -1.3, 0.2], [16.6, -0.6, 0.2],
        [17.2, 1.2, 0.2],
    ])

    x_goal = jnp.array([20.0, 0.0, 0.0])

    @dataclass
    class _Cfg:
        n: int
        nu: int
        N: int
        W: jnp.ndarray
        u_ref: jnp.ndarray

    cfg = _Cfg(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,)),
    )

    # First controller config
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=10,
        max_iterations=2000,
    )
    sls_cfg = SLSConfig(
        max_sls_iterations=3,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        warm_start=False,
    )
    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        line_search=True,
    )

    parameter = dt

    om_max = 20.0
    u_min = jnp.array([-om_max])
    u_max = jnp.array([ om_max])

    constraints_u = make_control_box_constraints(u_min, u_max)
    x_max = jnp.array([30.0, 30.0, jnp.inf], dtype=jnp.float64)
    x_min = -x_max
    constraints_x = make_state_box_constraints(x_min, x_max)
    constraints_all = combine_constraints(constraints_x, constraints_u)

    n_obs = obstacles.shape[0]
    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    nc = 2 * nu + 2 * n + n_obs

    # disturbance used inside controller (tube computation)
    E_mag = 0.075
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
        X_in=jnp.zeros((cfg.N + 1, cfg.n)),
        U_in=jnp.zeros((cfg.N, cfg.nu)),
    )

    # Initial condition
    x0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)

    # Reference: goal state repeated
    reference = jnp.tile(x_goal[None, :], (N + 1, 1))

    # disturbance simulation matrix
    E_sim = alpha_sim * jnp.eye(3, dtype=jnp.float64)

    # warmup / nominal plan
    print("[Info] Warmup/nominal solve...")
    start = time.perf_counter()
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(x0=x0, reference=reference, parameter=parameter)
    u0.block_until_ready()
    end = time.perf_counter()
    print("[Info] Nominal trajectory done.")

    # second controller config (your tuned settings)
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e6,
        max_iterations=800,
    )
    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        warm_start=False,
    )
    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        line_search=True,
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

    # re-run to get Phi_x/Phi_u etc for the plan we’ll use
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(x0=x0, reference=reference, parameter=parameter)
    u0.block_until_ready()

    # precompute one tube for visualization
    tube = get_trajectory_tubes(Phi_x)  # (N+1, n)
    plan_xy = X_pred[:, :2]
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    plans_xy = np.asarray([np.asarray(plan_xy)])   # (1, N+1, 2)
    lowers_xy = np.asarray([np.asarray(lower)])
    uppers_xy = np.asarray([np.asarray(upper)])

    # -----------------------------
    # MULTI-ROLLOUT closed-loop
    # -----------------------------
    print("[Info] Multi-rollout closed-loop with disturbance-feedback via Phi_u...")
    T_steps = N

    xs_rollouts = np.full((NUM_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)
    us_rollouts = np.full((NUM_ROLLOUTS, T_steps, 1), np.nan, dtype=np.float64)

    base_key = jax.random.PRNGKey(0)
    centers_np = np.asarray(centers)
    radii_np = np.asarray(radii)

    total_time = 0.0
    min_time = float("inf")

    for i in range(NUM_ROLLOUTS):
        print(f"[Rollout {i+1}/{NUM_ROLLOUTS}]")
        key_i = jax.random.fold_in(base_key, i)

        x = x0
        disturbance_history = [jnp.zeros((3,), dtype=jnp.float64)]  # w_0 = 0

        for k in range(T_steps):
            # (keep your original timing behavior; update if you want real per-iter timing)
            dt_run = end - start
            total_time += dt_run
            min_time = min(min_time, dt_run)

            # disturbance-feedback (causal) using Phi_u[k,j] and past w_j
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]
            u_k = U_pred[k] + disturbance_feedback

            key_i, x_next, w = dubins_step_with_disturbance(key_i, x, u_k, E_sim, dt, i)

            xs_rollouts[i, k, :] = np.asarray(x_next)
            us_rollouts[i, k, :] = np.asarray(u_k)

            # crash check (same as your snippet: against obstacle 0)
            dx = float(x_next[0]) - float(centers_np[0, 0])
            dy = float(x_next[1]) - float(centers_np[0, 1])
            dist0 = (dx * dx + dy * dy) ** 0.5
            if dist0 < float(radii_np[0]):
                print(f"  crashed at step k={k}, dist0={dist0:.4f} < r0={float(radii_np[0]):.4f}")
                break

            disturbance_history.append(w)
            x = x_next

    np.savez(
        "dubins_mpc_rollout_multi.npz",
        obstacles=np.asarray(obstacles),
        X_rollouts=xs_rollouts,      # (R,T,3) NaN padded
        U_rollouts=us_rollouts,      # (R,T,1) NaN padded
        X_pred=np.asarray(X_pred),
        U_pred=np.asarray(U_pred),
        Phi_x=np.asarray(Phi_x),
        Phi_u=np.asarray(Phi_u),
        plans_xy=plans_xy,
        lowers_xy=lowers_xy,
        uppers_xy=uppers_xy,
        dt=np.asarray(dt),
        NUM_RANDOM=np.asarray(NUM_RANDOM),
    )
    print("[Saved] Multi-rollout NPZ → dubins_mpc_rollout_multi.npz")

    print("Average Time (ms):", total_time / max(1, np.isfinite(xs_rollouts[:, :, 0]).sum()) * 1000)
    print("Min time (ms):", min_time * 1000)

    # --- SPRITE PATH (edit this) ---
    sprite_path = "car_removed.png"  # <- put your sprite file here

    save_replay(
        xs=xs_rollouts,       # (R,T,3)
        centers=centers,
        radii=radii,
        plans_xy=plans_xy,
        lowers_xy=lowers_xy,
        uppers_xy=uppers_xy,
        filename="replay_multi.mp4",
        dt=dt,
        fps=int(round(1.0 / dt)),
        box_stride=1,
        margin=0.5,
        sprite_path=sprite_path,
        sprite_length=0.4,
        sprite_width=0.25,
        sprite_alpha=0.80,    # requested
    )


if __name__ == "__main__":
    main()
