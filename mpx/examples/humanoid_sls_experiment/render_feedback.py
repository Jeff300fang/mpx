"""
render_h1_rollout_via_model_dynamics.py

Re-simulate a saved H1 rollout NPZ by stepping the SAME dynamics function used in MPC,
then generate:

1) A 2D tube replay animation (executed trajectory + nominal + tube boxes + obstacles)
2) (Optional) A "6 plots per row" tube-vs-diff visualization for each selected state dimension

This version supports rolling out the executed trajectory using SLS disturbance-feedback
via Phi_u (if present in the NPZ):

    u_i = u_nom_i + sum_{j=0..i} Phi_u[i, j] @ w_hist[j]

and infers w_{i+1} from mismatch between stored X[i+1] ("actual") and model prediction
x_next_nom using:

    E w = (x_next_act - x_next_nom)

with E = diag([0.05, 0.05, 0, 0, ...]) by default.

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
# Example: NUM = 15 -> first 15 iterations of the horizon.
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
E_SCALE = 0.05
E_FIRST_K = 3

# ---- Control selection from U ----
# True: apply only first n_joints entries (tau prefix)
USE_TAU_PREFIX = True

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

    return model, n_joints, dt, step_dynamics


# -----------------------------
# Nominal rollout via model dynamics
# -----------------------------
def rollout_with_model_dynamics(
    X0: np.ndarray,              # (nx,)
    U: np.ndarray,               # (N, nu)
    parameter: np.ndarray,       # (>=N+1, p_dim) or indexable by t
    n_joints: int,
    step_dynamics,               # jitted: (x,u,t,parameter)->x_next
    *,
    use_tau_prefix: bool = True,
) -> np.ndarray:
    N = int(U.shape[0])
    nx = int(X0.shape[0])

    x = jnp.asarray(X0, dtype=jnp.float64)
    Uj = jnp.asarray(U, dtype=jnp.float64)
    Pj = jnp.asarray(parameter, dtype=jnp.float64)

    xs = np.zeros((N + 1, nx), dtype=np.float64)
    xs[0] = np.asarray(x, dtype=np.float64)

    for t in range(N):
        u = Uj[t, :n_joints] if use_tau_prefix else Uj[t]
        x = step_dynamics(x, u, t, Pj)
        xs[t + 1] = np.asarray(x, dtype=np.float64)

    return xs


# -----------------------------
# SLS (Phi_u) feedback rollout utilities
# -----------------------------
def make_E_diag(nx: int, scale: float = 0.05, first_k: int = 2) -> np.ndarray:
    E = np.zeros((nx, nx), dtype=np.float64)
    k = int(min(first_k, nx))
    for i in range(k):
        E[i, i] = float(scale)
    return E


def infer_disturbance(E_i: np.ndarray, x_next_act: np.ndarray, x_next_nom: np.ndarray) -> np.ndarray:
    rhs = np.asarray(x_next_act, dtype=np.float64) - np.asarray(x_next_nom, dtype=np.float64)
    w_i, *_ = np.linalg.lstsq(E_i, rhs, rcond=None)
    return w_i


def sls_control_from_history_general(
    u_nom_i: np.ndarray,      # (m,)
    Phi_u_i: np.ndarray,      # (i+1, m, n_w)
    w_hist: np.ndarray,       # (i+1, n_w)
) -> np.ndarray:
    u = u_nom_i.copy()
    for j in range(Phi_u_i.shape[0]):
        u += Phi_u_i[j] @ w_hist[j]
    return u


def rollout_with_model_dynamics_sls_feedback(
    X: np.ndarray,                # (N+1, nx) stored rollout used as "actual" for disturbance inference
    U: np.ndarray,                # (N, nu) nominal controls (tau prefix)
    Phi_u: np.ndarray,            # indexable Phi_u[i, j] -> (m, n_w)
    parameter: np.ndarray,        # (>=N+1, p_dim)
    n_joints: int,
    step_dynamics,                # jitted
    *,
    use_tau_prefix: bool = True,
    E_scale: float = 0.05,
    E_first_k: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    Phi_u = np.asarray(Phi_u, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)

    N = int(U.shape[0])
    nx = int(X.shape[1])

    m = int(n_joints) if use_tau_prefix else int(U.shape[1])

    if Phi_u.ndim < 4:
        raise ValueError(f"Expected Phi_u to have >=4 dims, got shape={Phi_u.shape}")

    n_w = int(Phi_u.shape[-1])
    E = make_E_diag(nx=nx, scale=float(E_scale), first_k=int(E_first_k))

    X_sim = np.zeros((N + 1, nx), dtype=np.float64)
    U_applied = np.zeros((N, m), dtype=np.float64)
    W = np.zeros((N + 1, n_w), dtype=np.float64)  # W[0]=0
    w_hist = np.zeros((N + 1, n_w), dtype=np.float64)

    x = jnp.asarray(X[0], dtype=jnp.float64)
    Pj = jnp.asarray(parameter, dtype=jnp.float64)
    X_sim[0] = X[0].copy()

    for i in range(N):
        u_nom_i = U[i, :m] if use_tau_prefix else U[i].copy()

        # Phi_u slice: (i+1, m, n_w) for j=0..i
        Phi_u_i = Phi_u[i, : i + 1]

        # Applied control
        u_i = sls_control_from_history_general(u_nom_i, Phi_u_i, w_hist[: i + 1])

        # Step model under applied control
        u_jax = jnp.asarray(u_i, dtype=jnp.float64)
        x_next_nom = step_dynamics(x, u_jax, i, Pj)
        x_next_nom_np = np.asarray(x_next_nom, dtype=np.float64)

        # Infer disturbance from mismatch to stored rollout
        x_next_act = X[i + 1]
        w_full = infer_disturbance(E, x_next_act, x_next_nom_np)  # (nx,)

        # Map to n_w if needed
        if w_full.shape[0] == n_w:
            w_i = np.asarray(w_full, dtype=np.float64)
        else:
            w_i = np.asarray(w_full[:n_w], dtype=np.float64)

        U_applied[i] = u_i
        W[i + 1] = w_i
        w_hist[i + 1] = w_i

        X_sim[i + 1] = x_next_nom_np
        x = x_next_nom

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
            ax.add_patch(plt.Circle((cx, cy), r, alpha=0.30))

    executed_line, = ax.plot([], [], lw=2, alpha=0.85, label="Executed (sim)")
    nominal_line,  = ax.plot([], [], lw=2, ls="--", alpha=0.90, label="Nominal")
    cur_pt = ax.scatter([], [], marker="o", s=55, label="Current")
    end_pt = ax.scatter([], [], marker="x", s=65, label="Nominal end")

    tube_patches: list[Rectangle] = []
    frame_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.28), framealpha=0.9)

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


# -----------------------------
# Main
# -----------------------------
def main():
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
    Phi_x = data_npz["Phi_x"]  # can be memmap-like

    Phi_u = None
    if "Phi_u" in data_npz.files:
        Phi_u = np.asarray(data_npz["Phi_u"], dtype=np.float64)

    obstacles = None
    if SHOW_OBSTACLES and ("obstacles" in data_npz.files):
        obstacles = np.asarray(data_npz["obstacles"], dtype=np.float64)

    # Build dynamics (also gives n_joints, dt)
    model, n_joints, dt, step_dynamics = build_h1_step_dynamics()

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

    # -----------------------------
    # Horizon truncation via NUM
    # -----------------------------
    if int(NUM) > 0:
        H = int(NUM)
        H = max(1, min(H, int(U.shape[0]), int(X.shape[0]) - 1))
        X = X[: H + 1]
        U = U[: H]
        parameter = parameter[: H + 1]

        # Phi_x is required for tubes; keep first H steps along time axis if possible
        if hasattr(Phi_x, "shape") and Phi_x.shape[0] >= H:
            Phi_x = Phi_x[:H]

        # Phi_u optional; keep first H along i-axis, and clamp history axis to <= H+1
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
        print("Rolling out with model dynamics + Phi_u disturbance feedback...")
        X_sim, U_applied, W = rollout_with_model_dynamics_sls_feedback(
            X=X,
            U=U,
            Phi_u=Phi_u,
            parameter=parameter,
            n_joints=n_joints,
            step_dynamics=step_dynamics,
            use_tau_prefix=bool(USE_TAU_PREFIX),
            E_scale=float(E_SCALE),
            E_first_k=int(E_FIRST_K),
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
            n_joints=n_joints,
            step_dynamics=step_dynamics,
            use_tau_prefix=bool(USE_TAU_PREFIX),
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
        T = min(X_sim.shape[0], X.shape[0], tube_sizes.shape[0])
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
