#!/usr/bin/env python3
"""
render_h1_rollout_via_model_dynamics.py

Re-simulate a saved H1 rollout NPZ by stepping the SAME dynamics function used in MPC,
then generate:

1) A 2D tube replay animation (executed trajectory + nominal + tube boxes + obstacles)
2) (Optional) A "6 plots per row" tube-vs-diff visualization for each selected state dimension

THIS VERSION:
  - Removes the previous w-estimation/inference logic entirely.
  - If Phi_u is present and USE_SLS_FEEDBACK=True, it samples an exogenous disturbance w_t
    (uniform in a unit L2 ball) ONLY in the first 3 dimensions; all remaining dims are zero.
  - Uses that sampled w_t for BOTH:
      (a) the SLS feedback law u_t = u_nom + sum_j Phi_u[t,j] w_j
      (b) state disturbance injection: x_{t+1} = f(x_t,u_t,...) + E @ w_t
    where E is a diagonal-like mapping (nx x n_w) with nonzeros on the first E_FIRST_K diagonals.

MODIFIED (minimal changes):
  - Optionally overwrite the contact sequence parameter[t,:4] during rollout by inferring contact
    from current foot-point heights (geom_xpos z) with hysteresis.
  - This is done only in the replay/rollout script (does NOT change your MPC code).

No argparse: configure via variables in the "USER CONFIG" block below.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from functools import partial

import mpx.config.config_h1 as config
from mpx.utils.fast_sls_visual import get_trajectory_tubes


# ============================================================
# USER CONFIG (edit these instead of CLI arguments)
# ============================================================

# ---- IO ----
NPZ_PATH = os.path.join("mpc_data", "h1_mpc_rollout.npz")
OUT_ANIM = "h1_replay.mp4"              # .mp4 or .gif
TUBE_GRID = True
TUBE_GRID_OUT = "tube_vs_diff.png"      # .png or .pdf

# ---- Horizon truncation ----
# NUM = number of iterations/controls to keep (U[:NUM]) and states (X[:NUM+1]).
NUM = 30

# ---- Replay plot config ----
IDX_PX = 0
IDX_PY = 1
FPS = 0                 # 0 => infer from dt
BOX_STRIDE = 1          # draw every k-th box along nominal
MARGIN = 0.75
SHOW_OBSTACLES = True   # if NPZ has 'obstacles'

# ---- Nominal path source for tube boxes/nominal line ----
# "stored" uses X from NPZ, "sim" uses X_sim
NOMINAL_SOURCE = "stored"  # "stored" or "sim"

# ---- Tube grid config ----
TUBE_GRID_QPOSQVEL_ONLY = False
TUBE_GRID_MAX_DIMS = 0     # 0 => all selected; else cap to first K dims
TUBE_GRID_NCOLS = 6

# ---- SLS feedback rollout ----
USE_SLS_FEEDBACK = True    # requires Phi_u in NPZ
E_SCALE = 0.1
E_FIRST_K = 3

# Disturbance sampling:
# sample w_t only in the first 3 dims (uniform L2 ball), zeros elsewhere.
W_SEED = 0
W_RADIUS = 1.0  # unit ball if 1.0

# ---- Control selection from U ----
# True: apply only first n_joints entries (tau prefix)
USE_TAU_PREFIX = True

# ---- Contact inference from current foot heights (NEW) ----
# If True, during rollout we overwrite parameter[t,:4] using current geom_xpos z with hysteresis.
INFER_CONTACT_FROM_HEIGHT = True

# Hysteresis thresholds (meters):
#   - turn ON contact when z < Z_ON
#   - turn OFF contact when z > Z_OFF
# (Z_OFF should be > Z_ON to avoid chatter)
Z_ON = 0.015
Z_OFF = 0.030

# Tie the two points per foot together (recommended for your H1 4-point setup)
TIE_POINTS_PER_FOOT = True

# If True, initialize hysteresis state from stored parameter[0,:4]; else start from zeros.
INIT_CONTACT_FROM_LOGGED = True

# ============================================================


# -----------------------------
# Quaternion utilities (kept for compatibility / debugging)
# -----------------------------
def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q
    return q / n


def align_quat_sign(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    return -q if float(np.dot(q, q_ref)) < 0.0 else q


# -----------------------------
# X layout handling (kept for potential future use)
# -----------------------------
@dataclass(frozen=True)
class LayoutSpec:
    mode: str = "prefix"
    offset: int = 0


def extract_qpos_qvel(
    x_row: np.ndarray,
    model: mujoco.MjModel,
    n_joints: int,
    layout: LayoutSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    x_row = np.asarray(x_row)

    if layout.mode == "model_dims_prefix":
        nq = int(model.nq)
        nv = int(model.nv)
        start = 0
    else:
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

        if layout.mode == "prefix":
            start = 0
        elif layout.mode == "offset":
            start = int(layout.offset)
        else:
            raise ValueError(f"Unknown layout.mode={layout.mode}")

    need = start + nq + nv
    if x_row.shape[0] < need:
        raise ValueError(
            f"State row too small for requested layout: need at least {need} entries "
            f"(start={start}, nq={nq}, nv={nv}), got {x_row.shape[0]}."
        )

    qpos = np.array(x_row[start: start + nq], dtype=np.float64, copy=True)
    qvel = np.array(x_row[start + nq: start + nq + nv], dtype=np.float64, copy=True)
    return qpos, qvel


# -----------------------------
# Contact inference helper (NEW)
# -----------------------------
def infer_contact_from_height_hysteresis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_geom_ids: list[int],
    qpos: np.ndarray,
    qvel: np.ndarray,
    c_prev: np.ndarray,
    *,
    z_on: float,
    z_off: float,
    tie_points_per_foot: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer 4 contact bits from current geom heights with hysteresis.

    Returns:
      c_new: (4,) float64 in {0,1}
      z: (4,) float64 foot-point heights
    """
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    z = np.zeros((4,), dtype=np.float64)
    for i, gid in enumerate(contact_geom_ids):
        z[i] = float(data.geom_xpos[gid, 2])

    c = np.array(c_prev, dtype=np.float64, copy=True)

    # hysteresis update
    c[z < float(z_on)] = 1.0
    c[z > float(z_off)] = 0.0

    if tie_points_per_foot:
        # Your verified mapping: (0,1) left foot, (2,3) right foot
        c[1] = c[0]
        c[3] = c[2]

    return c, z


# -----------------------------
# Dynamics instantiation (matches MPC wrapper style)
# -----------------------------
def build_h1_step_dynamics():
    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)

    n_joints = int(getattr(config, "n_joints", 0))
    dt = float(getattr(config, "dt", 0.02))

    contact_id = []
    for name in getattr(config, "contact_frame", []):
        contact_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name))

    body_id = []
    for name in getattr(config, "body_name", []):
        body_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name))

    if any(int(i) < 0 for i in contact_id):
        raise RuntimeError(
            f"Failed to resolve some contact_frame geoms: {getattr(config,'contact_frame',None)}"
        )
    if any(int(i) < 0 for i in body_id):
        raise RuntimeError(
            f"Failed to resolve some body_name bodies: {getattr(config,'body_name',None)}"
        )

    dynamics = partial(
        config.dynamics,
        model, mjx_model,
        contact_id, body_id,
        n_joints, dt
    )

    @jax.jit
    def step_dynamics(x, u, t, parameter):
        return dynamics(x, u, t, parameter=parameter)

    return model, mjx_model, contact_id, n_joints, dt, step_dynamics


# -----------------------------
# Nominal rollout via model dynamics (MODIFIED: optional contact inference)
# -----------------------------
def rollout_with_model_dynamics(
    X0: np.ndarray,              # (nx,)
    U: np.ndarray,               # (N, nu)
    parameter: np.ndarray,       # (>=N+1, p_dim) or indexable by t
    model: mujoco.MjModel,       # (NEW) for height-based contact inference
    contact_id: list[int],       # (NEW) geom ids (len=4)
    n_joints: int,
    step_dynamics,               # jitted: (x,u,t,parameter)->x_next
    *,
    use_tau_prefix: bool = True,
    infer_contact_from_height: bool = False,
    z_on: float = 0.015,
    z_off: float = 0.030,
    tie_points_per_foot: bool = True,
    init_contact_from_logged: bool = True,
) -> np.ndarray:
    N = int(U.shape[0])
    nx = int(X0.shape[0])

    x = jnp.asarray(X0, dtype=jnp.float64)
    Uj = jnp.asarray(U, dtype=jnp.float64)
    Pj = jnp.asarray(parameter, dtype=jnp.float64)

    # CPU MuJoCo data for contact inference (if enabled)
    mj_data = mujoco.MjData(model) if infer_contact_from_height else None

    # Hysteresis state
    if infer_contact_from_height:
        if init_contact_from_logged and parameter.shape[0] > 0 and parameter.shape[1] >= 4:
            c_prev = np.array(parameter[0, :4], dtype=np.float64)
        else:
            c_prev = np.zeros((4,), dtype=np.float64)

    xs = np.zeros((N + 1, nx), dtype=np.float64)
    xs[0] = np.asarray(x, dtype=np.float64)

    for t in range(N):
        u = Uj[t, :n_joints] if use_tau_prefix else Uj[t]

        if infer_contact_from_height:
            # Infer contact from current state x (CPU forward kinematics)
            x_np = np.asarray(x, dtype=np.float64)
            qpos = x_np[: n_joints + 7]
            qvel = x_np[n_joints + 7 : 2 * n_joints + 13]

            c_new, _z = infer_contact_from_height_hysteresis(
                model=model,
                data=mj_data,
                contact_geom_ids=contact_id,
                qpos=qpos,
                qvel=qvel,
                c_prev=c_prev,
                z_on=float(z_on),
                z_off=float(z_off),
                tie_points_per_foot=bool(tie_points_per_foot),
            )
            c_prev = c_new

            # Overwrite parameter[t,:4] on-device
            Pj = Pj.at[t, :4].set(jnp.asarray(c_new, dtype=jnp.float64))

        x = step_dynamics(x, u, t, Pj)
        xs[t + 1] = np.asarray(x, dtype=np.float64)

    return xs


# -----------------------------
# SLS (Phi_u) feedback rollout utilities (UPDATED: sampled w, no inference)
# MODIFIED: optional contact inference same as above
# -----------------------------
def make_E_diag_rect(nx: int, n_w: int, scale: float = 0.05, first_k: int = 3) -> np.ndarray:
    """
    Build E as (nx, n_w) with diagonal entries on the first min(first_k, nx, n_w).
    """
    E = np.diag(np.zeros(nx))
    E[0:3] = 0.025
    E[26:29] = 0.2
    return E


def sample_unit_ball_first3_into_nw(
    rng: np.random.Generator,
    n_w: int,
    radius: float = 1.0,
) -> np.ndarray:
    """
    w in R^{n_w} where:
      - w[0:3] ~ Uniform L2 ball of radius 'radius'
      - w[3:] = 0
    """
    if n_w < 3:
        raise ValueError(f"n_w must be >= 3, got {n_w}")

    v = rng.normal(size=(3,))
    n = np.linalg.norm(v)
    if n < 1e-12:
        v = np.array([1.0, 0.0, 0.0])
        n = 1.0
    v = v / n

    r = float(radius) * (rng.random() ** (1.0 / 3.0))
    w3 = r * v

    w = np.zeros((n_w,), dtype=np.float64)
    w[:3] = w3
    w[0] = 0.05
    w[1] = 0.05
    w[2] = 0.00
    return w


def rollout_with_model_dynamics_sls_feedback(
    X0: np.ndarray,               # (nx,) initial state
    U: np.ndarray,                # (N, nu) nominal controls (tau prefix)
    Phi_u: np.ndarray,            # indexable Phi_u[i, j] -> (m, n_w)
    parameter: np.ndarray,        # (>=N+1, p_dim)
    model: mujoco.MjModel,        # (NEW) for height-based contact inference
    contact_id: list[int],        # (NEW) geom ids (len=4)
    n_joints: int,
    step_dynamics,                # jitted
    *,
    use_tau_prefix: bool = True,
    E_scale: float = 0.05,
    E_first_k: int = 3,
    w_seed: int = 0,
    w_radius: float = 1.0,
    infer_contact_from_height: bool = False,
    z_on: float = 0.015,
    z_off: float = 0.030,
    tie_points_per_foot: bool = True,
    init_contact_from_logged: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executed rollout:
      u_i = u_nom_i + sum_{j=0..i} Phi_u[i,j] @ w_j
      x_{i+1} = f(x_i, u_i, ...) + E @ w_i

    w_i is sampled exogenously each step:
      - uniform L2 ball in first 3 dims
      - zeros in remaining dims

    If infer_contact_from_height=True, overwrites parameter[i,:4] from current foot heights.
    """
    U = np.asarray(U, dtype=np.float64)
    Phi_u = np.asarray(Phi_u, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    X0 = np.asarray(X0, dtype=np.float64)

    N = int(U.shape[0])
    nx = int(X0.shape[0])

    m = int(n_joints) if use_tau_prefix else int(U.shape[1])

    if Phi_u.ndim < 4:
        raise ValueError(f"Expected Phi_u to have >=4 dims, got shape={Phi_u.shape}")

    n_w = nx

    E = make_E_diag_rect(nx=nx, n_w=n_w, scale=float(E_scale), first_k=int(E_first_k))
    rng = np.random.default_rng(int(w_seed))

    X_sim = np.zeros((N + 1, nx), dtype=np.float64)
    U_applied = np.zeros((N, m), dtype=np.float64)
    W = np.zeros((N + 1, n_w), dtype=np.float64)  # W[0]=0
    w_hist = np.zeros((N + 1, n_w), dtype=np.float64)

    x = jnp.asarray(X0, dtype=jnp.float64)
    Pj = jnp.asarray(parameter, dtype=jnp.float64)

    mj_data = mujoco.MjData(model) if infer_contact_from_height else None

    if infer_contact_from_height:
        if init_contact_from_logged and parameter.shape[0] > 0 and parameter.shape[1] >= 4:
            c_prev = np.array(parameter[0, :4], dtype=np.float64)
        else:
            c_prev = np.zeros((4,), dtype=np.float64)

    X_sim[0] = X0.copy()

    for i in range(N):
        u_nom_i = U[i, :m] if use_tau_prefix else U[i].copy()

        if infer_contact_from_height:
            x_np = np.asarray(x, dtype=np.float64)
            qpos = x_np[: n_joints + 7]
            qvel = x_np[n_joints + 7 : 2 * n_joints + 13]

            c_new, _z = infer_contact_from_height_hysteresis(
                model=model,
                data=mj_data,
                contact_geom_ids=contact_id,
                qpos=qpos,
                qvel=qvel,
                c_prev=c_prev,
                z_on=float(z_on),
                z_off=float(z_off),
                tie_points_per_foot=bool(tie_points_per_foot),
            )
            c_prev = c_new
            Pj = Pj.at[i, :4].set(jnp.asarray(c_new, dtype=jnp.float64))

        # SLS feedback term from history
        disturbance_feedback = np.zeros(U.shape[1], dtype=np.float64)
        for j in range(i + 1):
            disturbance_feedback += Phi_u[i, j] @ w_hist[j]
        u0 = u_nom_i + disturbance_feedback

        # Step nominal model under applied control
        u_jax = jnp.asarray(u0, dtype=jnp.float64)
        x_next_nom = step_dynamics(x, u_jax, i, Pj)
        x_next_nom_np = np.asarray(x_next_nom, dtype=np.float64)

        w_i = sample_unit_ball_first3_into_nw(rng, n_w=n_w, radius=float(w_radius))

        W[i + 1] = w_i
        w_hist[i + 1] = w_i

        x_next = x_next_nom_np + E @ w_i * 0.05

        U_applied[i] = u0
        X_sim[i + 1] = x_next
        x = jnp.asarray(x_next, dtype=jnp.float64)

    return X_sim, U_applied, W


# -----------------------------
# State names (optional)
# -----------------------------
def build_state_name_list(nx: int, n_joints: int) -> list[str]:
    nq = 7 + n_joints
    nv = 6 + n_joints
    names: list[str] = []
    names += ["base_x", "base_y", "base_z"]
    names += ["base_qw", "base_qx", "base_qy", "base_qz"]
    names += [f"joint_pos_{j}" for j in range(n_joints)]
    names += ["base_vx", "base_vy", "base_vz"]
    names += ["base_wx", "base_wy", "base_wz"]
    names += [f"joint_vel_{j}" for j in range(n_joints)]
    for k in range(nq + nv, nx):
        names.append(f"state_{k}")
    return names[:nx]


# -----------------------------
# Tube replay animation (MP4/GIF)
# -----------------------------
def save_tube_replay(
    executed: np.ndarray,            # (T_exec, nx)
    nominal: np.ndarray,             # (T_nom, nx)
    tube_sizes: np.ndarray,          # (T_tube, nx)
    *,
    idx_px: int,
    idx_py: int,
    obstacles: Optional[np.ndarray],
    filename: str,
    dt: float,
    fps: Optional[int],
    box_stride: int,
    margin: float,
    title: str = "H1 Replay: Executed + Nominal + Tube Boxes",
):
    executed = np.asarray(executed)
    nominal = np.asarray(nominal)
    tube_sizes = np.asarray(tube_sizes)

    T = min(executed.shape[0], nominal.shape[0], tube_sizes.shape[0])
    if T <= 1:
        raise ValueError(f"Not enough timesteps to animate (T={T}).")

    ex_xy = executed[:T, [idx_px, idx_py]]
    nom_xy = nominal[:T, [idx_px, idx_py]]
    tube_xy = tube_sizes[:T, [idx_px, idx_py]]  # half-widths in x/y

    lowers = nom_xy - tube_xy
    uppers = nom_xy + tube_xy

    if fps is None:
        fps = max(1, int(round(1.0 / float(dt))))
    interval_ms = int(round(1000.0 / fps))

    all_px = [ex_xy[:, 0], nom_xy[:, 0], lowers[:, 0], uppers[:, 0]]
    all_py = [ex_xy[:, 1], nom_xy[:, 1], lowers[:, 1], uppers[:, 1]]
    if obstacles is not None and obstacles.size:
        all_px.append(obstacles[:, 0])
        all_py.append(obstacles[:, 1])

    all_px = np.concatenate(all_px)
    all_py = np.concatenate(all_py)
    xmin, xmax = float(all_px.min() - margin), float(all_px.max() + margin)
    ymin, ymax = float(all_py.min() - margin), float(all_py.max() + margin)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    if obstacles is not None and obstacles.size:
        for k in range(obstacles.shape[0]):
            cx, cy, r = float(obstacles[k, 0]), float(obstacles[k, 1]), float(obstacles[k, 2])
            ax.add_patch(
                plt.Circle(
                    (cx, cy),
                    r,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=1.0,
                )
            )

    executed_line, = ax.plot([], [], lw=2, alpha=0.85, label="Executed (sim)")
    nominal_line,  = ax.plot([], [], lw=2, ls="--", alpha=0.90, label="Nominal", color="tab:blue")
    cur_pt = ax.scatter([], [], marker="o", s=55, label="Current", alpha=0.0)
    end_pt = ax.scatter([], [], marker="x", s=65, label="Nominal end", color="tab:blue")

    tube_patches: list[Rectangle] = []
    frame_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.legend(loc="upper left", bbox_to_anchor=(0., 0.28), framealpha=0.9)

    def init():
        executed_line.set_data([], [])
        nominal_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        end_pt.set_offsets(np.zeros((0, 2)))
        frame_text.set_text("")
        return (executed_line, nominal_line, cur_pt, end_pt, frame_text)

    def update(t: int):
        nonlocal tube_patches
        t = int(t)
        t = max(0, min(t, T - 1))

        executed_line.set_data(ex_xy[: t + 1, 0], ex_xy[: t + 1, 1])
        cur_pt.set_offsets(np.array([[ex_xy[t, 0], ex_xy[t, 1]]]))

        nominal_line.set_data(nom_xy[:, 0], nom_xy[:, 1])
        end_pt.set_offsets(np.array([[nom_xy[-1, 0], nom_xy[-1, 1]]]))

        for p in tube_patches:
            p.remove()
        tube_patches = []

        stride = max(int(box_stride), 1)
        for k in range(0, T, stride):
            w = uppers[k, 0] - lowers[k, 0]
            h = uppers[k, 1] - lowers[k, 1]
            if not np.isfinite(w) or not np.isfinite(h):
                continue
            if w < 0.0 or h < 0.0:
                continue
            rect = Rectangle((lowers[k, 0], lowers[k, 1]), w, h, alpha=0.18)
            ax.add_patch(rect)
            tube_patches.append(rect)

        frame_text.set_text(f"Step {t}/{T-1}")
        return (executed_line, nominal_line, cur_pt, end_pt, frame_text, *tube_patches)

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init, blit=False, interval=interval_ms
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


# -----------------------------
# 6-per-row tube-vs-diff grid plot
# -----------------------------
def save_tube_vs_diff_grid(
    *,
    tube_sizes: np.ndarray,     # (T, nx)
    diff: np.ndarray,           # (T, nx)
    dt: float,
    idx: np.ndarray,            # indices plotted
    state_names: Optional[list[str]],
    ncols: int,
    out_path: str,
    suptitle: str,
):
    tube_sizes = np.asarray(tube_sizes)
    diff = np.asarray(diff)
    idx = np.asarray(idx, dtype=int)

    if tube_sizes.ndim != 2 or diff.ndim != 2:
        raise ValueError(f"tube_sizes and diff must be 2D. got {tube_sizes.shape=} {diff.shape=}")

    T = min(tube_sizes.shape[0], diff.shape[0])
    if T <= 1:
        raise ValueError(f"Not enough timesteps for grid plot (T={T}).")

    tube_plot = tube_sizes[:T, idx]
    diff_plot = diff[:T, idx]

    n_idx = idx.shape[0]
    nrows = int(math.ceil(n_idx / ncols))
    t = np.arange(T) * float(dt)

    fig_w = 18
    fig_h = 3.0 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_2d(axes)

    for j in range(n_idx):
        r = j // ncols
        c = j % ncols
        ax = axes[r, c]

        ax.plot(t, tube_plot[:, j], linewidth=1.2, label="tube")
        ax.plot(t, diff_plot[:, j], linewidth=1.2, label="|x_sim - x_stored|")

        title = f"state_{idx[j]}"
        if state_names is not None and idx[j] < len(state_names):
            title = state_names[idx[j]]
        ax.set_title(title, fontsize=9)

        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    for j in range(n_idx, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(suptitle, y=0.995)

    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def save_standalone_legend(out_path: str = "h1_legend.pdf", *,
                           figsize=(6.5, 1.4),
                           ncol=3,
                           fontsize=14,
                           frameon=True,
                           dpi=300):
    planned  = Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=2.5,
                      label="Planned trajectory")
    obstacle = Line2D([0], [0], color="red", linestyle="--", linewidth=2.5,
                      label="Inflated obstacle boundary")
    tube     = Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.18,
                     label="Tubes")

    handles = [planned, obstacle, tube]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.legend(handles=handles,
              loc="center",
              ncol=ncol,
              frameon=frameon,
              fontsize=fontsize,
              handlelength=3.2,
              handleheight=1.4,
              columnspacing=1.8,
              handletextpad=0.8,
              borderpad=0.8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    save_standalone_legend()
    jax.config.update("jax_enable_x64", True)

    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PATH}")

    data_npz = np.load(NPZ_PATH)
    need = {"X", "U", "parameter"}
    missing = [k for k in need if k not in data_npz.files]
    if missing:
        raise RuntimeError(f"NPZ missing required arrays {missing}. Found: {list(data_npz.files)}")

    X = np.asarray(data_npz["X"])
    U = np.asarray(data_npz["U"])
    parameter = np.asarray(data_npz["parameter"])

    if "Phi_x" not in data_npz.files:
        raise RuntimeError(
            "Tube visualization requires 'Phi_x' in the NPZ.\n"
            "Save Phi_x during rollout generation."
        )
    Phi_x = data_npz["Phi_x"]

    Phi_u = None
    if "Phi_u" in data_npz.files:
        Phi_u = np.asarray(data_npz["Phi_u"], dtype=np.float64)

    obstacles = None
    if SHOW_OBSTACLES and ("obstacles" in data_npz.files):
        obstacles = np.asarray(data_npz["obstacles"], dtype=np.float64)

    # Build dynamics (also gives n_joints, dt)
    model, mjx_model, contact_id, n_joints, dt, step_dynamics = build_h1_step_dynamics()

    print(f"Loaded NPZ: {NPZ_PATH}")
    print(f"Arrays: {list(data_npz.files)}")
    print(f"Stored X shape: {X.shape}, U shape: {U.shape}, parameter shape: {parameter.shape}")
    if Phi_u is not None:
        print(f"Phi_u shape: {Phi_u.shape}")
    if obstacles is not None:
        print(f"obstacles shape: {obstacles.shape}")
    print(f"Using config.model_path: {config.model_path}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}")
    print(f"n_joints={n_joints}, dt={dt}")
    print(f"INFER_CONTACT_FROM_HEIGHT={INFER_CONTACT_FROM_HEIGHT} (Z_ON={Z_ON}, Z_OFF={Z_OFF})")

    # -----------------------------
    # Horizon truncation via NUM
    # -----------------------------
    if int(NUM) > 0:
        H = int(NUM)
        H = max(1, min(H, int(U.shape[0]), int(X.shape[0]) - 1))
        X = X[: H + 1]
        U = U[: H]
        parameter = parameter[: H + 1]

        if hasattr(Phi_x, "shape") and Phi_x.shape[0] >= H:
            Phi_x = Phi_x[:H]

        if Phi_u is not None:
            Phi_u = Phi_u[:H]
            if Phi_u.ndim >= 2:
                Phi_u = Phi_u[:, : min(Phi_u.shape[1], H + 1)]

        print(f"[NUM] Truncated to H={H}: X{X.shape}, U{U.shape}, parameter{parameter.shape}, "
              f"Phi_x{getattr(Phi_x,'shape',None)}, Phi_u{None if Phi_u is None else Phi_u.shape}")

    # -----------------------------
    # Roll out executed trajectory
    # -----------------------------
    if USE_SLS_FEEDBACK and (Phi_u is not None):
        print("Rolling out with model dynamics + Phi_u feedback + SAMPLED disturbance (first 3 dims)...")
        X_sim, U_applied, W = rollout_with_model_dynamics_sls_feedback(
            X0=X[0],
            U=U,
            Phi_u=Phi_u,
            parameter=parameter,
            model=model,
            contact_id=contact_id,
            n_joints=n_joints,
            step_dynamics=step_dynamics,
            use_tau_prefix=bool(USE_TAU_PREFIX),
            E_scale=float(E_SCALE),
            E_first_k=int(E_FIRST_K),
            w_seed=int(W_SEED),
            w_radius=float(W_RADIUS),
            infer_contact_from_height=bool(INFER_CONTACT_FROM_HEIGHT),
            z_on=float(Z_ON),
            z_off=float(Z_OFF),
            tie_points_per_foot=bool(TIE_POINTS_PER_FOOT),
            init_contact_from_logged=bool(INIT_CONTACT_FROM_LOGGED),
        )
        print(f"SLS rollout complete: X_sim{X_sim.shape}, U_applied{U_applied.shape}, W{W.shape}")
    else:
        if USE_SLS_FEEDBACK and (Phi_u is None):
            print("NOTE: USE_SLS_FEEDBACK=True but NPZ has no Phi_u. Falling back to nominal rollout.")
        print("Rolling out with model dynamics (nominal controls only)...")
        X_sim = rollout_with_model_dynamics(
            X0=X[0],
            U=U,
            parameter=parameter,
            model=model,
            contact_id=contact_id,
            n_joints=n_joints,
            step_dynamics=step_dynamics,
            use_tau_prefix=bool(USE_TAU_PREFIX),
            infer_contact_from_height=bool(INFER_CONTACT_FROM_HEIGHT),
            z_on=float(Z_ON),
            z_off=float(Z_OFF),
            tie_points_per_foot=bool(TIE_POINTS_PER_FOOT),
            init_contact_from_logged=bool(INIT_CONTACT_FROM_LOGGED),
        )
        print(f"Nominal rollout complete: X_sim{X_sim.shape}")

    # -----------------------------
    # Tube sizes from Phi_x
    # -----------------------------
    print("Computing tube sizes from Phi_x...")
    tube_sizes = np.asarray(get_trajectory_tubes(Phi_x), dtype=np.float64)
    print(f"tube_sizes shape: {tube_sizes.shape}")

    # Choose nominal source for the plotted tube boxes/nominal line
    if str(NOMINAL_SOURCE).lower() == "sim":
        nominal = X_sim
    else:
        nominal = X

    # FPS
    fps = None if int(FPS) <= 0 else int(FPS)

    np.savez(
        "data.npz",
        X=X,
        tube_sizes=tube_sizes,
        X_rollout=X_sim,
    )

    # Save animation
    os.makedirs(os.path.dirname(OUT_ANIM) or ".", exist_ok=True)
    print(f"Saving tube replay to: {OUT_ANIM}")
    save_tube_replay(
        executed=X_sim,
        nominal=nominal,
        tube_sizes=tube_sizes,
        idx_px=int(IDX_PX),
        idx_py=int(IDX_PY),
        obstacles=obstacles,
        filename=str(OUT_ANIM),
        dt=float(dt),
        fps=fps,
        box_stride=int(BOX_STRIDE),
        margin=float(MARGIN),
        title="H1 Replay: Executed + Nominal + Tube Boxes",
    )
    print(f"Saved replay animation to: {OUT_ANIM}")

    # Optional: 6-per-row tube-vs-diff grid
    if bool(TUBE_GRID):
        T = min(X_sim.shape[0], X.shape[0], tube_sizes.shape[0], 30)
        if T <= 2:
            raise RuntimeError(f"Not enough timesteps for tube grid (T={T}).")

        diff_full = np.abs(X_sim[1:T] - X[1:T])  # (T-1, nx)
        tube_full = tube_sizes[1:T]              # (T-1, nx)

        nx = int(X.shape[1])
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

        if bool(TUBE_GRID_QPOSQVEL_ONLY):
            idx = np.arange(min(nq + nv, nx))
        else:
            idx = np.arange(nx)

        if int(TUBE_GRID_MAX_DIMS) > 0:
            idx = idx[: int(TUBE_GRID_MAX_DIMS)]

        state_names = build_state_name_list(nx=nx, n_joints=int(n_joints))

        os.makedirs(os.path.dirname(TUBE_GRID_OUT) or ".", exist_ok=True)
        print(f"Saving tube grid plot to: {TUBE_GRID_OUT} (dims={len(idx)})")
        save_tube_vs_diff_grid(
            tube_sizes=tube_full,
            diff=diff_full,
            dt=float(dt),
            idx=idx,
            state_names=state_names,
            ncols=int(TUBE_GRID_NCOLS),
            out_path=str(TUBE_GRID_OUT),
            suptitle="Tube size vs model deviation (per state dimension)",
        )
        print(f"Saved tube grid plot to: {TUBE_GRID_OUT}")

    print("Done.")


if __name__ == "__main__":
    main()
