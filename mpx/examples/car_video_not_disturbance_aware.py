"""
dubins_car_mpc_experiment.py

End-to-end experiment: use mpx.utils.generic_mpc_wrapper.GenericMPCControllerWrapper
to control a Dubins car with box constraints on controls.

MODIFIED:
- Detects obstacle collision during rollout simulation.
- If collision occurs, stops the rollout immediately.
- In the MP4 for a single rollout: draws an 'X' at the crash location and
  keeps the video length fixed to T_steps (so alignment matches non-crash cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import os

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

config.update("jax_enable_x64", True)

# -----------------------------
# Styling / plotting defaults
# -----------------------------
mpl.rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,
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

NUM_RANDOM = 1
NUM_ADV = 0

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
    where w is sampled from a unit-ball-ish distribution.

    Returns (key_next, x_next, w).
    """
    px, py, th = x
    om = u[0]

    # Nominal step
    px_next = px + dt * V_CONST * jnp.cos(th)
    py_next = py + dt * V_CONST * jnp.sin(th)
    th_next = wrap_to_pi(th + dt * om)
    x_nom = jnp.array([px_next, py_next, th_next], dtype=x.dtype)

    # Disturbance sampling
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (x.shape[0],), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(x.shape[0], dtype=x.dtype)
    a = jnp.asarray(1.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)
    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = r * z

    # Optional deterministic disturbance (currently forced)
    # start = i - NUM_RANDOM + 5
    w = jnp.array([0.0, 1.0, 0.0], dtype=x.dtype)

    x_next = x_nom + E @ w
    return key, x_next, w

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dt = parameter
    return dubins_step(x, u, dt)

# -----------------------------
# Cost and Constraints
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
    return wx * (dx * dx) + wy * (dy * dy) + wtheta * theta_cost + womega * (om * om)

def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints

def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)

    return constraints

def make_constant_disturbance(n: int, alpha: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns E with shape (T, n, n), E[t] = alpha * I.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)
        return jnp.broadcast_to(E0, (T, n, n))
    return disturbance

# -----------------------------
# Collision detection
# -----------------------------
def collides_with_obstacles_xy(
    x: jnp.ndarray,               # (3,)
    obstacles: jnp.ndarray,       # (K,3) [cx,cy,r]
) -> jnp.bool_:
    if obstacles.size == 0:
        return jnp.asarray(False)
    p = x[:2]
    c = obstacles[:, :2]
    r = obstacles[:, 2]
    d2 = jnp.sum((c - p[None, :])**2, axis=1)
    return jnp.any(d2 <= (r * r))

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
    crash_xy: np.ndarray | None = None,      # (n_rollouts,2) or None
    crash_mask: np.ndarray | None = None,    # (n_rollouts,) bool or None
):
    xs = np.asarray(xs)

    # Normalize xs to (n_rollouts, T, 3)
    if xs.ndim == 2 and xs.shape[1] == 3:
        xs = xs[None, :, :]
    elif xs.ndim != 3 or xs.shape[2] != 3:
        raise ValueError(f"xs must have shape (T,3) or (n_rollouts,T,3). Got {xs.shape}.")

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

    # axis limits
    all_x = [xs[:, :, 0].ravel()]
    all_y = [xs[:, :, 1].ravel()]

    if plans_xy is not None:
        all_x.append(plans_xy[:, :, 0].ravel())
        all_y.append(plans_xy[:, :, 1].ravel())
    if lo is not None and up is not None:
        all_x.append(lo[:, 0].ravel()); all_x.append(up[:, 0].ravel())
        all_y.append(lo[:, 1].ravel()); all_y.append(up[:, 1].ravel())
    if centers is not None and centers.size:
        all_x.append(centers[:, 0].ravel())
        all_y.append(centers[:, 1].ravel())
    if crash_xy is not None:
        crash_xy = np.asarray(crash_xy)
        all_x.append(crash_xy[:, 0].ravel())
        all_y.append(crash_xy[:, 1].ravel())

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
        step_idx2 = int(step_idx if step_idx is not None else 0)
        step_idx2 = max(0, min(step_idx2, plans_xy.shape[0] - 1))
        ax.plot(plans_xy[step_idx2, :, 0], plans_xy[step_idx2, :, 1],
                linestyle="--", linewidth=2, label="Planned (open-loop)")

    # rollouts
    for i in range(n_rollouts):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=rollout_alpha, color="tab:orange")
    ax.plot([], [], alpha=rollout_alpha, label=f"Rollouts (n={n_rollouts})")

    # crash markers
    if crash_xy is not None and crash_mask is not None:
        crash_xy = np.asarray(crash_xy)
        crash_mask = np.asarray(crash_mask).astype(bool)
        if crash_xy.shape[0] == n_rollouts:
            ax.scatter(crash_xy[crash_mask, 0], crash_xy[crash_mask, 1],
                       marker="x", s=120, linewidths=3, label="Crash (collision)")

    ax.set_title("Dubins: Rollouts + Robust Tube + Obstacles")
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def save_single_rollout_mp4(
    x_rollout,                 # (T,3) with NaNs after stop OR fully finite
    filename="disturbed_rollout.mp4",
    dt=0.1,
    fps=None,
    total_frames=None,         # if set, force video to this many frames
    centers=None,              # (K,2) optional
    radii=None,                # (K,)  optional
    plan_xy=None,              # (N+1,2) optional
    lower_xy=None,             # (N+1,2) optional (tube lower)
    upper_xy=None,             # (N+1,2) optional (tube upper)
    tube_stride=2,
    margin=0.5,
    show_plan=True,
    show_tubes=True,
    crash_idx=None,            # int frame index where crash occurs, or None
    crash_xy=None,             # (2,) crash location for X, or None
):
    """
    Saves an MP4 showing a *single* disturbed rollout over time.

    MODIFIED:
      - If crash_idx/crash_xy are provided, draws an 'X' at crash_xy from crash_idx onward.
      - If total_frames is provided, the animation always has that many frames.
        After termination (NaN/stop), it "holds" the last valid state and keeps showing
        the same picture until the end so timing matches non-crash runs.

    Requires ffmpeg for MP4.
    """
    x_rollout = np.asarray(x_rollout, dtype=float)
    if x_rollout.ndim != 2 or x_rollout.shape[1] != 3:
        raise ValueError(f"x_rollout must have shape (T,3). Got {x_rollout.shape}.")

    # Find last valid frame (stop at first NaN in position)
    valid = np.isfinite(x_rollout[:, 0]) & np.isfinite(x_rollout[:, 1])
    first_invalid = int(np.argmax(~valid)) if np.any(~valid) else x_rollout.shape[0]
    last_valid_idx = max(0, first_invalid - 1)
    if last_valid_idx < 1:
        raise ValueError("Rollout has <=1 valid state; nothing to animate.")

    # Frames
    if total_frames is None:
        total_frames = x_rollout.shape[0]
    total_frames = int(total_frames)
    total_frames = max(total_frames, last_valid_idx + 1)

    # FPS
    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))

    # Axis limits
    all_x = [x_rollout[: last_valid_idx + 1, 0]]
    all_y = [x_rollout[: last_valid_idx + 1, 1]]
    if plan_xy is not None:
        plan_xy = np.asarray(plan_xy, dtype=float)
        all_x.append(plan_xy[:, 0]); all_y.append(plan_xy[:, 1])
    if show_tubes and lower_xy is not None and upper_xy is not None:
        lower_xy = np.asarray(lower_xy, dtype=float)
        upper_xy = np.asarray(upper_xy, dtype=float)
        all_x.append(lower_xy[:, 0]); all_x.append(upper_xy[:, 0])
        all_y.append(lower_xy[:, 1]); all_y.append(upper_xy[:, 1])
    if centers is not None and radii is not None:
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if centers.size:
            all_x.append(centers[:, 0]); all_y.append(centers[:, 1])
    if crash_xy is not None:
        crash_xy = np.asarray(crash_xy, dtype=float).reshape(2)
        all_x.append(crash_xy[0:1]); all_y.append(crash_xy[1:2])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
    ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)

    # Figure
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    # Obstacles
    if centers is not None and radii is not None and centers.size and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), alpha=0.35, color="tab:red"))

    # Fixed plan
    if show_plan and plan_xy is not None:
        ax.plot(plan_xy[:, 0], plan_xy[:, 1], linestyle="--", linewidth=2, label="Plan")

    # Fixed tube boxes
    if show_tubes and (lower_xy is not None) and (upper_xy is not None):
        stride = max(1, int(tube_stride))
        for k in range(0, lower_xy.shape[0], stride):
            w = upper_xy[k, 0] - lower_xy[k, 0]
            h = upper_xy[k, 1] - lower_xy[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            ax.add_patch(Rectangle((lower_xy[k, 0], lower_xy[k, 1]), w, h, alpha=0.15))
        ax.plot([], [], alpha=0.15, label="Tube boxes")

    # Animated artists
    rollout_line, = ax.plot([], [], linewidth=2, label="Disturbed rollout")
    cur_pt = ax.scatter([], [], s=50, marker="o", label="Current state")

    # Crash marker (hidden initially)
    crash_artist = ax.scatter([], [], marker="x", s=160, linewidths=3, label="Crash")  # shown conditionally

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="best", framealpha=0.9)

    # Normalize crash info
    if crash_idx is not None:
        crash_idx = int(crash_idx)
    if crash_xy is not None:
        crash_xy = np.asarray(crash_xy, dtype=float).reshape(2)

    def init():
        rollout_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        crash_artist.set_offsets(np.zeros((0, 2)))
        title.set_text("")
        return rollout_line, cur_pt, crash_artist, title

    def update(t):
        # Hold last valid frame after termination, for fixed-length video
        t_eff = min(int(t), last_valid_idx)

        px = x_rollout[: t_eff + 1, 0]
        py = x_rollout[: t_eff + 1, 1]
        rollout_line.set_data(px, py)

        cur_pt.set_offsets(np.array([[x_rollout[t_eff, 0], x_rollout[t_eff, 1]]]))

        crashed_now = (crash_idx is not None) and (crash_xy is not None) and (t >= crash_idx)
        if crashed_now:
            crash_artist.set_offsets(np.array([[crash_xy[0], crash_xy[1]]]))
            # title.set_text(f"CRASH at frame {crash_idx} | t={t}/{total_frames-1} (time={t*dt:.2f}s)")
        else:
            crash_artist.set_offsets(np.zeros((0, 2)))
            # title.set_text(f"t = {t} / {total_frames-1}  (time = {t*dt:.2f}s)")
        return rollout_line, cur_pt, crash_artist, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
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

    # Obstacles: [cx, cy, r]
    obstacles = jnp.array([[0.0, 0.0, 0.3]], dtype=jnp.float64)
    n_obs = obstacles.shape[0]
    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]

    # NOTE: you had nc = 2*nu + 2*n + n_obs (assumes one obstacle constraint per obstacle)
    nc = 2 * nu + 2 * n + n_obs

    # disturbance model for controller (E[t] = alpha_sim * I)
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

    # Initial condition / goal / reference
    x0 = jnp.array([-0.75, -0.75, 0.0], dtype=jnp.float64)
    x_goal = jnp.array([1.0, 0.6, 0.0], dtype=jnp.float64)

    X_ref = jnp.tile(x_goal[None, :], (N + 1, 1))
    reference = X_ref
    T_steps = N  # rollout length (frames alignment target)

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(3, dtype=jnp.float64)

    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    # Warmup / nominal solve
    jax.debug.print("Warmup complete.")
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    u0.block_until_ready()
    jax.debug.print("Nominal trajectory done")

    # Update configs for robust run
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e6,
        max_iterations=1000,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
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

    # robust plan (single call)
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )

    tube = get_trajectory_tubes(Phi_x)              # (N+1, n)
    plan_xy = X_pred[:, :2]
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    # Rollout simulations with early stopping + COLLISION STOP
    xs = np.full((N_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)

    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)      # exclusive
    crash_steps = np.full((N_ROLLOUTS,), -1, dtype=np.int32)          # frame index where crash happens
    crash_xy = np.full((N_ROLLOUTS, 2), np.nan, dtype=np.float64)     # crash position

    for i in range(N_ROLLOUTS):
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]  # w history
        x = x0

        for k in range(T_steps):
            # stop if within tol of goal (before applying step k)
            if bool(reached_goal_xy(x, x_goal, GOAL_TOL)):
                stop_steps[i] = k
                break

            # u = U_pred[k] + sum_{j<=k} Phi_u[k,j] w_j
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]
            u = U_pred[k] + disturbance_feedback

            key, x_next, w = dubins_step_with_disturbance(key, x, u, E_sim, dt, i)

            # log deviation wrt nominal prediction (at k+1)
            disturbed[i, k, :2] = np.abs(np.asarray(X_pred[k + 1, :2] - x_next[:2]))
            disturbed[i, k, 2]  = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 2] - x_next[2])))

            # advance
            x = x_next
            disturbance_history.append(w)
            xs[i, k] = np.asarray(x)

            # COLLISION CHECK (after state update)
            if bool(collides_with_obstacles_xy(x, obstacles)):
                crash_steps[i] = k
                crash_xy[i] = np.asarray(x[:2])
                stop_steps[i] = k + 1  # exclusive
                break

    crash_mask = (crash_steps >= 0)

    # Plot rollouts + tube + obstacles (+ crash X's)
    plot_rollouts_tubes_centers(
        xs=xs,
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
        crash_xy=crash_xy,
        crash_mask=crash_mask,
    )

    # Deviation vs tube size plots (nan-safe)
    dx_np_all  = disturbed[:, :, 0]
    dy_np_all  = disturbed[:, :, 1]
    dth_np_all = disturbed[:, :, 2]

    tube_x_np  = np.asarray(tube[1:, 0])
    tube_y_np  = np.asarray(tube[1:, 1])
    tube_th_np = np.asarray(tube[1:, 2])

    t_arr = np.arange(dx_np_all.shape[1]) * dt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax1.plot(t_arr, tube_x_np, label="tube size (x)", linewidth=4)
    for r, dx_np in enumerate(dx_np_all):
        m = np.isfinite(dx_np)
        ax1.plot(t_arr[m], dx_np[m], label="|x - x_nominal|" if r == 0 else None)
    ax1.set_ylabel("meters")
    ax1.set_title("X-direction: Deviation vs Tube Size")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t_arr, tube_y_np, label="tube size (y)", linewidth=4)
    for r, dy_np in enumerate(dy_np_all):
        m = np.isfinite(dy_np)
        ax2.plot(t_arr[m], dy_np[m], label="|y - y_nominal|" if r == 0 else None)
    ax2.set_ylabel("meters")
    ax2.set_title("Y-direction: Deviation vs Tube Size")
    ax2.grid(True)
    ax2.legend()

    ax3.plot(t_arr, tube_th_np, label="tube size (theta)", linewidth=4)
    for r, dth_np in enumerate(dth_np_all):
        m = np.isfinite(dth_np)
        ax3.plot(t_arr[m], dth_np[m], label="|wrap(theta - theta_nominal)|" if r == 0 else None)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("radians")
    ax3.set_title("Theta-direction: Deviation vs Tube Size")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("disturbance_vs_tube_size_xytheta_dubins.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save rollouts + tubes to NPZ (includes stop_steps + crash info)
    out_dir = os.path.join(os.getcwd(), "testing_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "sls_vs_deepreach.npz")

    plans_xy_np  = np.asarray(plans_xy)
    lowers_xy_np = np.asarray(lowers_xy)
    uppers_xy_np = np.asarray(uppers_xy)

    np.savez(
        npz_path,
        xs=np.asarray(xs),
        disturbed=np.asarray(disturbed),
        stop_steps=np.asarray(stop_steps),
        crash_steps=np.asarray(crash_steps),
        crash_xy=np.asarray(crash_xy),
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

    # Save a single rollout MP4 with fixed length (T_steps frames)
    i = 0
    ci = int(crash_steps[i]) if crash_steps[i] >= 0 else None
    cxy = crash_xy[i] if crash_steps[i] >= 0 else None

    save_single_rollout_mp4(
        x_rollout=xs[i],                 # (T_steps,3) with NaNs after stop
        filename="disturbed_rollout_0.mp4",
        dt=dt,
        fps=None,
        total_frames=T_steps,             # <-- forces consistent video length
        centers=np.asarray(centers),
        radii=np.asarray(radii),
        plan_xy=np.asarray(plans_xy[0]) if len(plans_xy) else None,
        lower_xy=np.asarray(lowers_xy[0]) if len(lowers_xy) else None,
        upper_xy=np.asarray(uppers_xy[0]) if len(uppers_xy) else None,
        tube_stride=1,
        show_plan=False,
        show_tubes=True,
        crash_idx=ci,
        crash_xy=cxy,
    )

if __name__ == "__main__":
    main()
