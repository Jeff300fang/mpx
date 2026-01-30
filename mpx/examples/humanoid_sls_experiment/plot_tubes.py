"""
render_h1_rollout_via_model_dynamics.py

Re-simulate a saved H1 rollout NPZ by stepping the SAME dynamics function used in MPC,
then generate:

1) A 2D tube replay animation (executed trajectory + nominal + tube boxes + obstacles)
2) (Optional) A "6 plots per row" tube-vs-diff visualization for each selected state dimension

Workflow:
  - Load X, U, parameter (and optionally Phi_x, obstacles) from NPZ
  - Build step_dynamics = jit(partial(config.dynamics, model, mjx_model, ...))
  - Rollout X_sim from X[0] using step_dynamics
  - Tube sizes from Phi_x using get_trajectory_tubes(Phi_x)
  - Save:
      - replay animation (.mp4 or .gif)
      - optional tube grid plot (.png or .pdf)

Assumptions:
  - config_h1 provides: model_path, n_joints, dt, contact_frame, body_name, dynamics(...)
  - NPZ contains:
      X: (N+1, nx)
      U: (N, nu)
      parameter: (>=N+1, p_dim) or indexable by t used in dynamics
      Phi_x: (required for tubes)
    Optional:
      obstacles: (n_obs, 3) as [cx, cy, r] for overlay

Run:
  python render_h1_rollout_via_model_dynamics.py --npz mpc_data/h1_mpc_rollout.npz --out h1_replay.mp4
  python render_h1_rollout_via_model_dynamics.py --npz ... --out ... --tube-grid --tube-grid-out tube_vs_diff.png
"""

from __future__ import annotations

import os
import math
import argparse
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
    """
    Defines how to extract [qpos, qvel] from one state x row.

    mode:
      - "prefix": x[:(nq+nv)] is [qpos,qvel] with nq=7+n_joints, nv=6+n_joints
      - "offset": x[offset:offset+(nq+nv)] is [qpos,qvel]
      - "model_dims_prefix": use model.nq/model.nv from prefix (x[:model.nq+model.nv])
    """
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

    qpos = np.array(x_row[start : start + nq], dtype=np.float64, copy=True)
    qvel = np.array(x_row[start + nq : start + nq + nv], dtype=np.float64, copy=True)
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
# Rollout via model dynamics
# -----------------------------
def rollout_with_model_dynamics(
    X0: np.ndarray,              # (nx,)
    U: np.ndarray,               # (N, nu)
    parameter: np.ndarray,       # (>=N+1, p_dim) or indexable by t
    n_joints: int,
    step_dynamics,               # jitted: (x,u,t,parameter)->x_next
    *,
    use_tau_prefix: bool = True, # if True, u = U[t,:n_joints]
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
# State names (optional, but helpful for the tube grid)
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

    # fps / interval
    if fps is None:
        fps = max(1, int(round(1.0 / float(dt))))
    interval_ms = int(round(1000.0 / fps))

    # Axis limits
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

    # Obstacles
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

        # Clear previous boxes
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
    idx: np.ndarray,            # indices plotted (len = n_idx)
    state_names: Optional[list[str]],
    ncols: int = 6,
    out_path: str = "tube_vs_diff.png",
    suptitle: str = "Tube size vs model deviation (per state dimension)",
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

    # turn off unused axes
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
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Re-simulate H1 rollout with config.dynamics(...) then save tube replay mp4/gif + optional tube grid."
    )
    p.add_argument(
        "--npz",
        type=str,
        default=os.path.join("mpc_data", "h1_mpc_rollout.npz"),
        help="Path to rollout NPZ (must contain X, U, parameter; Phi_x required for tubes).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="h1_replay.mp4",
        help="Output animation filename (.mp4 or .gif).",
    )
    p.add_argument("--fps", type=int, default=0, help="FPS (0 -> infer from dt).")
    p.add_argument("--box-stride", type=int, default=1, help="Draw every k-th tube rectangle along nominal.")
    p.add_argument("--margin", type=float, default=0.75, help="Plot margin around data extents.")
    p.add_argument("--idx-px", type=int, default=0, help="State index for planar x.")
    p.add_argument("--idx-py", type=int, default=1, help="State index for planar y.")
    p.add_argument(
        "--nominal",
        type=str,
        default="stored",
        choices=["stored", "sim"],
        help="Nominal path source: 'stored' uses X from NPZ, 'sim' uses X_sim.",
    )
    p.add_argument("--no-obstacles", action="store_true", help="Disable obstacle overlay even if NPZ has 'obstacles'.")

    # ---- tube grid options ----
    p.add_argument("--tube-grid", action="store_true",
                   help="Also save a 6-per-row tube-vs-diff grid plot (png/pdf).")
    p.add_argument("--tube-grid-out", type=str, default="tube_vs_diff.png",
                   help="Output path for the tube grid plot (.png or .pdf).")
    p.add_argument("--tube-grid-qposqvel-only", action="store_true",
                   help="Plot only first (nq+nv) dims (qpos+qvel).")
    p.add_argument("--tube-grid-max-dims", type=int, default=0,
                   help="If >0, plot only the first this many selected dims.")

    return p.parse_args()


def main():
    # Keep your typical x64 setting for dynamics fidelity
    jax.config.update("jax_enable_x64", True)

    args = parse_args()
    npz_path = str(args.npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")

    data_npz = np.load(npz_path)
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

    obstacles = None
    if (not bool(args.no_obstacles)) and ("obstacles" in data_npz.files):
        obstacles = np.asarray(data_npz["obstacles"], dtype=np.float64)

    # Build dynamics
    model, n_joints, dt, step_dynamics = build_h1_step_dynamics()

    print(f"Loaded NPZ: {npz_path}")
    print(f"Arrays: {list(data_npz.files)}")
    print(f"Stored X shape: {X.shape}, U shape: {U.shape}, parameter shape: {parameter.shape}")
    print(f"Using config.model_path: {config.model_path}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}")
    print(f"n_joints={n_joints}, dt={dt}")

    # Rollout X_sim
    print("Rolling out with model dynamics...")
    X_sim = rollout_with_model_dynamics(
        X0=X[0],
        U=U,
        parameter=parameter,
        n_joints=n_joints,
        step_dynamics=step_dynamics,
        use_tau_prefix=True,
    )
    print(f"Sim rollout complete: X_sim shape={X_sim.shape}")

    # Tube sizes
    print("Computing tube sizes from Phi_x...")
    tubes = get_trajectory_tubes(Phi_x)
    tube_sizes = np.asarray(tubes, dtype=np.float64)
    print(f"tube_sizes shape: {tube_sizes.shape}")

    # Choose nominal source
    nominal = X if str(args.nominal) == "stored" else X_sim

    # FPS
    fps = None if int(args.fps) <= 0 else int(args.fps)

    # Save animation
    out_anim = str(args.out)
    os.makedirs(os.path.dirname(out_anim) or ".", exist_ok=True)
    print(f"Saving tube replay to: {out_anim}")
    save_tube_replay(
        executed=X_sim,
        nominal=nominal,
        tube_sizes=tube_sizes,
        idx_px=int(args.idx_px),
        idx_py=int(args.idx_py),
        obstacles=obstacles,
        filename=out_anim,
        dt=float(dt),
        fps=fps,
        box_stride=int(args.box_stride),
        margin=float(args.margin),
        title="H1 Replay: Executed + Nominal + Tube Boxes",
    )
    print(f"Saved replay animation to: {out_anim}")

    # Optional: 6-per-row tube-vs-diff grid
    if bool(args.tube_grid):
        # Align like your Go2 code: compare x[t] at the same time index, but
        # using 1..T-1 to mirror "next-step" style (avoids the initial condition trivially matching).
        T = min(X_sim.shape[0], X.shape[0], tube_sizes.shape[0])
        if T <= 2:
            raise RuntimeError(f"Not enough timesteps for tube grid (T={T}).")

        diff_full = np.abs(X_sim[1:T] - X[1:T])  # (T-1, nx)
        tube_full = tube_sizes[1:T]              # (T-1, nx)

        nx = int(X.shape[1])
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

        if bool(args.tube_grid_qposqvel_only):
            idx = np.arange(min(nq + nv, nx))
        else:
            idx = np.arange(nx)

        if int(args.tube_grid_max_dims) > 0:
            idx = idx[: int(args.tube_grid_max_dims)]

        state_names = build_state_name_list(nx=nx, n_joints=int(n_joints))

        out_grid = str(args.tube_grid_out)
        os.makedirs(os.path.dirname(out_grid) or ".", exist_ok=True)

        print(f"Saving tube grid plot to: {out_grid}  (dims={len(idx)})")
        save_tube_vs_diff_grid(
            tube_sizes=tube_full,
            diff=diff_full,
            dt=float(dt),
            idx=idx,
            state_names=state_names,
            ncols=6,
            out_path=out_grid,
            suptitle="Tube size vs model deviation (per state dimension)",
        )
        print(f"Saved tube grid plot to: {out_grid}")

    print("Done.")


if __name__ == "__main__":
    main()
