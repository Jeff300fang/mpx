"""
replay_go2_from_npz.py

Replay a saved MPC rollout (X, U, ...) in MuJoCo/QuadrupedEnv and compute one-step
prediction error by FORCE-SETTING the simulator to X[i] each step.

This script matches your state vector structure:
  x = [qpos, qvel, p_legs, grf]
where:
  qpos = [p(3), quat(4), q_joints(n_joints)]               -> nq = 7 + n_joints
  qvel = [base_twist(6), dq_joints(n_joints)]             -> nv = 6 + n_joints
  p_legs = foot positions (world frame)                   -> 3*n_contact
  grf = ground reaction forces per foot (world frame)     -> 3*n_contact

Notes:
- It compares errors on:
  (a) base pose (first 7)
  (b) qpos+qvel block (first nq+nv)
  (c) optionally full state (includes feet/grf), but feet/grf in X may not be
      dynamically consistent unless your model enforces it.

- Video writing can trigger os.fork warnings under JAX multithreading. You can:
  - disable video via SAVE_VIDEO = False
  - or keep it if it works on your system.

Usage:
  python replay_go2_from_npz.py
"""

from __future__ import annotations

import os
import numpy as np
import mujoco
import imageio
import os
import math
import matplotlib.pyplot as plt

from gym_quadruped.quadruped_env import QuadrupedEnv
import mpx.config.config_go2 as config
from mpx.utils.fast_sls_visual import get_trajectory_tubes


# -----------------------------
# Configuration
# -----------------------------
NPZ_PATH = os.path.join("mpc_data", "go2_mpc_rollout.npz")
SAVE_VIDEO = True
VIDEO_PATH = "go2_replay_forced_state.mp4"
FPS = 30

# If you want to strictly compare only qpos+qvel (recommended first):
COMPARE_ONLY_QPOS_QVEL = True

# If you want to compute GRFs from MuJoCo contacts and include them in x_actual:
# (If your X stores GRF as zeros, you probably want False for fair comparison.)
COMPUTE_GRF = False


# -----------------------------
# GRF computation (per-foot, world frame)
# -----------------------------
def compute_per_foot_grf_world(env: QuadrupedEnv,
                               foot_geom_ids: list[int],
                               n_contact: int = 4,
                               ground_geom_ids: set[int] | None = None) -> np.ndarray:
    """
    Sum MuJoCo contact forces per foot geom, returning (3*n_contact,) WORLD forces:
      [Fx, Fy, Fz] for each foot, ordered as foot_geom_ids.

    Uses mujoco.mj_contactForce (force in contact frame) and converts to world frame
    using contact.frame.

    ground_geom_ids: optionally filter to only ground contacts. If None, sums all
    contacts involving the foot.
    """
    m = env.mjModel
    d = env.mjData

    foot_index = {gid: i for i, gid in enumerate(foot_geom_ids)}
    grf = np.zeros((n_contact, 3), dtype=np.float64)

    for ci in range(d.ncon):
        con = d.contact[ci]
        g1 = con.geom1
        g2 = con.geom2

        foot_gid = None
        other_gid = None
        if g1 in foot_index:
            foot_gid = g1
            other_gid = g2
        elif g2 in foot_index:
            foot_gid = g2
            other_gid = g1
        else:
            continue

        if ground_geom_ids is not None and other_gid not in ground_geom_ids:
            continue

        cf = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(m, d, ci, cf)

        # Contact frame axes in world coordinates (3x3 flattened)
        R = np.array(con.frame, dtype=np.float64).reshape(3, 3)

        # Force in world frame
        f_world = R @ cf[:3]
        grf[foot_index[foot_gid]] += f_world

    return grf.reshape(-1)


# -----------------------------
# State mapping: X <-> MuJoCo
# -----------------------------
def set_env_state_from_X(env: QuadrupedEnv, X_i: np.ndarray, n_joints: int):
    """
    Force MuJoCo to the planned state X_i.

    X ordering:
      [qpos(7+n_joints), qvel(6+n_joints), foot_pos(3*n_contact), grf(3*n_contact)]
    """
    m = env.mjModel
    d = env.mjData
    nq = m.nq
    nv = m.nv

    # Expected for floating base + joints
    assert nq == 7 + n_joints, f"Unexpected nq: got {nq}, expected {7+n_joints}"
    assert nv == 6 + n_joints, f"Unexpected nv: got {nv}, expected {6+n_joints}"

    qpos = np.asarray(X_i[:nq], dtype=np.float64)
    qvel = np.asarray(X_i[nq:nq+nv], dtype=np.float64)

    d.qpos[:] = qpos
    d.qvel[:] = qvel

    # Ensure derived quantities consistent
    mujoco.mj_forward(m, d)


def build_X_from_env(env: QuadrupedEnv,
                     foot_geom_ids: list[int],
                     n_contact: int,
                     grf_as_state: bool,
                     compute_grf: bool) -> np.ndarray:
    """
    Build x_actual in the same ordering as X:
      [qpos, qvel, foot_op(world), grf(world)]
    """
    m = env.mjModel
    d = env.mjData

    qpos = np.asarray(d.qpos, dtype=np.float64).copy()
    qvel = np.asarray(d.qvel, dtype=np.float64).copy()

    mujoco.mj_kinematics(m, d)
    foot_op = np.array([d.geom_xpos[g] for g in foot_geom_ids], dtype=np.float64).reshape(-1)

    if grf_as_state:
        if compute_grf:
            grf = compute_per_foot_grf_world(env, foot_geom_ids, n_contact=n_contact)
        else:
            grf = np.zeros(3 * n_contact, dtype=np.float64)
    else:
        grf = np.zeros(0, dtype=np.float64)

    return np.concatenate([qpos, qvel, foot_op, grf], axis=0)


# -----------------------------
# Main
# -----------------------------
def main():
    assert os.path.exists(NPZ_PATH), f"Missing file: {NPZ_PATH}"

    data = np.load(NPZ_PATH)
    X = data["X"]  # (N+1, nx)
    U = data["U"]  # (N, nu)
    print(X[:, :3])
    N = U.shape[0]
    nx = X.shape[1]
    Phi_x = data["Phi_x"]

    print(f"Loaded {NPZ_PATH}")
    print(f"X shape: {X.shape}, U shape: {U.shape}")

    # Create env (match your setup)
    sim_frequency = 100.0
    env = QuadrupedEnv(
        robot="go2",
        scene="flat",
        sim_dt=1.0 / sim_frequency,
        ref_base_lin_vel=0.0,
        ground_friction_coeff=0.7,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )
    env.reset(random=False)

    # Resolve foot geom ids in your ordering (must match config.p_legs0 ordering!)
    # You used: ['FL','FR','RL','RR'] earlier. Keep consistent.
    foot_names = ["FL", "FR", "RL", "RR"]
    foot_geom_ids = [
        mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_GEOM, nm) for nm in foot_names
    ]
    if any(g < 0 for g in foot_geom_ids):
        raise RuntimeError(f"Could not resolve foot geom IDs for names {foot_names}. Check your model naming.")

    # Model sizes
    m = env.mjModel
    nq, nv = m.nq, m.nv
    n_joints = int(config.n_joints)
    n_contact = int(getattr(config, "n_contact", 4))
    grf_as_state = bool(getattr(config, "grf_as_state", False))

    print(f"MuJoCo nq={nq}, nv={nv} | config.n_joints={n_joints} n_contact={n_contact} grf_as_state={grf_as_state}")
    print(f"Expected X dim >= nq+nv+3*n_contact (+3*n_contact if grf_as_state): {nq+nv+3*n_contact + (3*n_contact if grf_as_state else 0)}")
    print(f"Actual X dim: {nx}")

    # Choose indices to compare
    if COMPARE_ONLY_QPOS_QVEL:
        idx = np.arange(nq + nv)
        print(f"Comparing {len(idx)} indices (qpos+qvel).")
    else:
        idx = np.arange(nx)
        print(f"Comparing {len(idx)} indices (full state).")

    # Video and renderer
    writer = imageio.get_writer(VIDEO_PATH, fps=FPS) if SAVE_VIDEO else None

    fb_w = int(getattr(m.vis.global_, "offwidth", 640))
    fb_h = int(getattr(m.vis.global_, "offheight", 480))
    W = min(640, fb_w)
    H = min(480, fb_h)
    renderer = mujoco.Renderer(m, height=H, width=W)
    diff = np.zeros((config.N, config.n - 24))
    # Error buffers
    abs_err = np.zeros((N, len(idx)), dtype=np.float64)
    base_pose_err = np.zeros(N, dtype=np.float64)

    # Forced-state one-step test
    for i in range(N - 1):
        # 1) Force simulator to X[i]
        set_env_state_from_X(env, X[i], n_joints=n_joints)

        # 2) Apply planned control for one step
        tau = np.asarray(U[i, :n_joints], dtype=np.float64)
        env.step(action=tau)

        # 3) Build x_actual in same ordering
        x_actual = build_X_from_env(
            env,
            foot_geom_ids=foot_geom_ids,
            n_contact=n_contact,
            grf_as_state=grf_as_state,
            compute_grf=COMPUTE_GRF,
        )
        x_pred = np.asarray(X[i + 1], dtype=np.float64)

        # 4) Errors
        abs_err[i] = np.abs(x_actual[idx] - x_pred[idx])
        base_pose_err[i] = np.linalg.norm(x_actual[:7] - x_pred[:7])

        if (i % 10) == 0 or i == N - 1:
            print(f"Step {i:3d}/{N}: base_pose_err={base_pose_err[i]:.6f}")

        # 5) Render / record
        env.render()
        renderer.update_scene(env.mjData)
        frame = renderer.render()
        if writer is not None:
            writer.append_data(frame)

    if writer is not None:
        writer.close()
        print(f"Saved video to: {VIDEO_PATH}")

    # Summaries
    print("\nError summary:")
    print(f"  mean |x_actual - x_pred| over compared indices: {abs_err.mean():.6e}")
    print(f"  max  |x_actual - x_pred| over compared indices: {abs_err.max():.6e}")
    print(f"  mean base pose (first 7) L2 error: {base_pose_err.mean():.6e}")
    print(f"  max  base pose (first 7) L2 error: {base_pose_err.max():.6e}")

    # Save errors
    out_err = "replay_errors.npz"
    np.savez(out_err, abs_err=abs_err, base_pose_err=base_pose_err, idx=idx)
    print(f"Saved errors to: {out_err}")

    diff = abs_err
    tubes = get_trajectory_tubes(Phi_x)
    tube_sizes = tubes[1:]                       # (N, nx)
    tube_sizes = np.asarray(tube_sizes)           # (N, nx) if you used tubes[1:]
    N, nx = diff.shape

    t = np.arange(N) * config.dt

    # Layout
    ncols = 6
    nrows = math.ceil(nx / ncols)

    fig_w = 18
    fig_h = 3.0 * nrows

    outdir = "tube_vs_diff"
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, f"tube_vs_diff_6perrow_N{N}_nx{nx}.png")

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_2d(axes)

    for j in range(nx):
        r = j // ncols
        c = j % ncols
        ax = axes[r, c]

        ax.plot(t, tube_sizes[:, j], linewidth=1.2, label="tube")
        ax.plot(t, diff[:, j],       linewidth=1.2, label="|x_actual - x_pred|")

        ax.set_title(f"x[{j}]", fontsize=9)
        ax.grid(True)

        # Optional: log scale helps when magnitudes vary a lot
        # ax.set_yscale("log")

    # Turn off unused axes
    for j in range(nx, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    # Only label bottom row to reduce clutter
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    # Single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle("Tube size vs actual deviation (per state dimension)", y=0.995)
    plt.tight_layout()

    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
