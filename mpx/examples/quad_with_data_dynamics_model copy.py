"""
replay_go2_from_npz_model_dynamics.py

Replay / one-step error evaluation for a saved MPC rollout (X, U, ...) by stepping
DIRECTLY with the SAME dynamics function you used inside MPC:

    dynamics = partial(config.dynamics, model, mjx_model, contact_id, body_id, n_joints, dt)
    x_next   = dynamics(x, u, t, parameter=parameter)

This script instantiates the dynamics exactly like your MPCControllerWrapper / BatchedMPCControllerWrapper
does (MuJoCo model + mjx_model + contact/body ids from config), then performs:

    for i:
        x_model_next = dynamics(X[i], U[i,:n_joints], i, parameter=parameter)
        compare with X[i+1]

Notes / assumptions:
- Your config module provides:
    - model_path, n_joints, dt
    - contact_frame (geom names) and body_name (body names)
    - dynamics callable with signature:
          config.dynamics(model, mjx_model, contact_id, body_id, n_joints, dt, x, u, t, parameter)
      (i.e., accepts keyword parameter=...)
- NPZ contains:
    - X: (N+1, nx)
    - U: (N, nu)
    - parameter: (N+1, p_dim)  (or at least enough for your dynamics; required here)
    - (optional) Phi_x for tube plotting

If you do not have "parameter" saved in the NPZ, you should save it during rollout generation.
"""

from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

from typing import Callable, Optional, Tuple
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from functools import partial

import mujoco
from mujoco import mjx

# Your go2 config (must match what produced the rollout)
import mpx.config.config_go2 as config

from mpx.utils.fast_sls_visual import get_trajectory_tubes


def sample_uniform_l2_ball_np(rng: np.random.Generator, dim: int) -> np.ndarray:
    """
    Sample w ~ Uniform({w: ||w||_2 <= 1}) in R^dim.
    Uses: w = r * z/||z||, z ~ N(0,I), r ~ U(0,1)^(1/dim)
    """
    z = rng.normal(size=(dim,))
    z_norm = np.linalg.norm(z) + 1e-12
    z = z / z_norm
    r = rng.uniform(0.0, 1.0) ** (1.0 / float(dim))
    return r * z


def step_dynamics_with_disturbance(
    rng: np.random.Generator,
    step_dynamics: Callable,          # step_dynamics(x,u,t,parameter) -> x_nom_next
    x: np.ndarray,                    # (nx,)
    u: np.ndarray,                    # (nu,)
    t: int,
    parameter: np.ndarray,            # whatever your dynamics expects (often (N+1,pdim))
    E: np.ndarray,                    # (nx, nw) (often nx x nx)
    mode: str = "sample_ball",        # "sample_ball" | "given_w" | "fixed_w"
    w_given: Optional[np.ndarray] = None,
    w_fixed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates: x_{t+1} = f(x_t,u_t,t,parameter) + E @ w_t

    Returns: (x_next, w_t)
    """
    x_nom = step_dynamics(x, u, t, parameter)

    nw = E.shape[1]

    if mode == "sample_ball":
        w = sample_uniform_l2_ball_np(rng, nw)
    elif mode == "given_w":
        if w_given is None:
            raise ValueError("mode='given_w' requires w_given.")
        w = np.asarray(w_given, dtype=float).reshape(nw,)
    elif mode == "fixed_w":
        if w_fixed is None:
            raise ValueError("mode='fixed_w' requires w_fixed.")
        w = np.asarray(w_fixed, dtype=float).reshape(nw,)
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    x_next = np.asarray(x_nom) + E @ w
    return x_next, w

# -----------------------------
# Configuration
# -----------------------------
NPZ_PATH = os.path.join("mpc_data", "go2_mpc_rollout.npz")

COMPARE_ONLY_QPOS_QVEL = False
MAKE_TUBE_PLOT = True

# If True, attempt to use MuJoCo joint names for joint_pos/joint_vel labels.
# If unavailable or if indexing is ambiguous, it falls back to joint_pos_{j}, joint_vel_{j}.
USE_MUJOCO_JOINT_NAMES = True


def _safe_mujoco_joint_names(model: mujoco.MjModel, n_joints: int):
    """
    Try to obtain a list of joint names from the MuJoCo model.

    Important:
    - MuJoCo joints include freejoint(s) etc. Depending on your model, joint ordering
      may include non-actuated joints. We do a best-effort extraction.
    - If names are missing or insufficient, return None to trigger fallback naming.
    """
    try:
        # model.njnt includes all joints. The first joint might be the free joint, etc.
        # We attempt to collect all joint names and then take the *last* n_joints as a heuristic,
        # because many legged models define freejoint first, then actuated joints.
        all_names = []
        for j in range(model.njnt):
            # mujoco>=3 has model.joint(j).name
            nm = model.joint(j).name
            all_names.append(nm if nm is not None else "")

        # If too few, bail.
        if len(all_names) < n_joints:
            return None

        # Heuristic: use the last n_joints. If those are empty, bail.
        cand = all_names[-n_joints:]
        if any((c is None) or (str(c).strip() == "") for c in cand):
            return None

        return [str(c) for c in cand]
    except Exception:
        return None


def build_state_name_list(
    nx: int,
    n_joints: int,
    model: mujoco.MjModel | None = None,
    use_mujoco_joint_names: bool = True,
):
    """
    Build a canonical list of length nx mapping state index -> human-readable name.

    Assumes the first (nq+nv) entries follow MuJoCo floating-base convention:
      qpos: base_pos(3), base_quat(4), joint_pos(n_joints)
      qvel: base_lin_vel(3), base_ang_vel(3), joint_vel(n_joints)

    Any remaining states beyond nq+nv are named "state_{k}".
    """
    names = []

    # Determine (nq, nv) from conventions used in your script.
    nq = 7 + n_joints
    nv = 6 + n_joints
    n_base = 3 + 4  # base position + quaternion
    assert nq == n_base + n_joints

    joint_names = None
    if use_mujoco_joint_names and model is not None:
        joint_names = _safe_mujoco_joint_names(model, n_joints)

    if joint_names is None:
        joint_pos_names = [f"joint_pos_{j}" for j in range(n_joints)]
        joint_vel_names = [f"joint_vel_{j}" for j in range(n_joints)]
    else:
        # Use the MuJoCo names but keep qpos/qvel suffixes explicit for clarity
        joint_pos_names = [f"{jn}_qpos" for jn in joint_names]
        joint_vel_names = [f"{jn}_qvel" for jn in joint_names]

    # --- qpos ---
    names += ["base_x", "base_y", "base_z"]
    names += ["base_qw", "base_qx", "base_qy", "base_qz"]
    names += joint_pos_names

    # --- qvel ---
    names += ["base_vx", "base_vy", "base_vz"]
    names += ["base_wx", "base_wy", "base_wz"]
    names += joint_vel_names

    # --- any remaining state entries ---
    if nx > (nq + nv):
        for k in range(nq + nv, nx):
            names.append(f"state_{k}")

    # If nx is smaller than expected, truncate; if larger, we already padded.
    return names[:nx]


def main():
    assert os.path.exists(NPZ_PATH), f"Missing file: {NPZ_PATH}"

    data = np.load(NPZ_PATH)
    print("NPZ arrays:", data.files)

    X = data["X"]  # (N+1, nx)
    U = data["U"]  # (N, nu)
    Phi_x = data["Phi_x"] if "Phi_x" in data.files else None
    Phi_u = data["Phi_u"]
    if "parameter" not in data.files:
        raise RuntimeError(
            "NPZ is missing required array 'parameter'.\n"
            "Your dynamics uses parameter[t,...] (e.g., contact schedule, liftoff, etc.).\n"
            "Save 'parameter' into the rollout NPZ when generating X/U."
        )
    parameter_np = data["parameter"]  # (N+1, p_dim)

    N = U.shape[0]
    nx = X.shape[1]

    # -----------------------------
    # Instantiate MuJoCo + MJX like your MPC wrappers
    # -----------------------------
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)

    n_joints = int(config.n_joints)
    dt = float(config.dt)

    # contact_id: geoms
    contact_id = []
    for name in config.contact_frame:
        contact_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name))

    # body_id: bodies
    body_id = []
    for name in config.body_name:
        body_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name))

    if any(int(i) < 0 for i in contact_id):
        raise RuntimeError(f"Failed to resolve some contact_frame geoms: {config.contact_frame}")
    if any(int(i) < 0 for i in body_id):
        raise RuntimeError(f"Failed to resolve some body_name bodies: {config.body_name}")

    # Instantiate the exact dynamics wrapper used in MPC
    dynamics = partial(
        config.dynamics,
        model, mjx_model,
        contact_id, body_id,
        n_joints, dt
    )

    # JIT a single step (parameter passed by keyword, like your MPC)
    @partial(jax.jit, static_argnums=())
    def step_dynamics(x, u, t, parameter):
        return dynamics(x, u, t, parameter=parameter)

    # -----------------------------
    # Comparison indices
    # -----------------------------
    nq = 7 + n_joints
    nv = 6 + n_joints
    if COMPARE_ONLY_QPOS_QVEL:
        idx = np.arange(min(nq + nv, nx))
        print(f"Comparing qpos+qvel only: {len(idx)} dims.")
    else:
        idx = np.arange(nx)
        print(f"Comparing full state: {len(idx)} dims.")

    # -----------------------------
    # Build state names (for plot labels)
    # -----------------------------
    state_names = build_state_name_list(
        nx=nx,
        n_joints=n_joints,
        model=model if USE_MUJOCO_JOINT_NAMES else None,
        use_mujoco_joint_names=USE_MUJOCO_JOINT_NAMES,
    )
    state_names_idx = [state_names[i] for i in idx]

    print(f"Loaded: {NPZ_PATH}")
    print(f"X shape: {X.shape}, U shape: {U.shape}, parameter shape: {parameter_np.shape}")
    print(f"n_joints={n_joints}, dt={dt}")
    print(f"Resolved contact_id={list(map(int, contact_id))}")
    print(f"Resolved body_id   ={list(map(int, body_id))}")

    # -----------------------------
    # Convert to JAX arrays
    # -----------------------------
    X_j = jnp.asarray(X, dtype=jnp.float64)
    U_j = jnp.asarray(U, dtype=jnp.float64)
    parameter_j = jnp.asarray(parameter_np, dtype=jnp.float64)

    # Take the first n_joints columns as torques (same as your wrapper uses tau = U[0, :n_joints])
    U_tau = U_j[:, :n_joints]

    # -----------------------------
    # One-step roll-forward errors
    # -----------------------------
    abs_err = np.zeros((N, len(idx)), dtype=np.float64)
    disturbance_history = np.zeros((N, len(idx)), dtype=np.float64)
    base_pose_err = np.zeros((N,), dtype=np.float64)
    x_i = X_j[0]
    diag = np.zeros(nx)
    diag[:2] = 0.004
    E = np.diag(diag)
    rng = np.random.default_rng(0)
    for i in range(N):
        u_i = U_tau[i]
        # x_i = X[i]
        disturbance_feedback_sum = np.zeros((u_i.shape[0],))
        for j in range(i):
            disturbance_feedback_sum += Phi_u[i, j + 1] @ disturbance_history[j]
        # u0 = u_i + disturbance_feedback_sum
        u0 = u_i
        # x_model_next, w_i = step_dynamics_with_disturbance(
        #     rng, step_dynamics, x_i, u0, i, parameter_np, E,
        #     mode="sample_ball",
        # )
        x_model_next = step_dynamics(x_i, u0, i, parameter_np)

        x_pred = np.asarray(X[i + 1], dtype=np.float64)
        x_act = np.asarray(x_model_next, dtype=np.float64)

        abs_err[i] = np.abs(x_act[idx] - x_pred[idx])
        # disturbance_history[i] = w_i
        base_pose_err[i] = np.linalg.norm(x_act[:7] - x_pred[:7])

        print(f"Step {i:3d}/{N}: base_pose_err={base_pose_err[i]:.6e}")

    # -----------------------------
    # Summaries
    # -----------------------------
    print("\nError summary (model-step vs stored X[i+1]):")
    print(f"  mean |x_model_next - X[i+1]| over compared indices: {abs_err.mean():.6e}")
    print(f"  max  |x_model_next - X[i+1]| over compared indices: {abs_err.max():.6e}")
    print(f"  mean base pose (first 7) L2 error: {base_pose_err.mean():.6e}")
    print(f"  max  base pose (first 7) L2 error: {base_pose_err.max():.6e}")

    # Save errors
    out_err = "replay_errors_modelstep.npz"
    np.savez(out_err, abs_err=abs_err, base_pose_err=base_pose_err, idx=idx)
    print(f"Saved errors to: {out_err}")

    # -----------------------------
    # Optional tube-vs-diff plot (uses Phi_x)
    # -----------------------------
    if MAKE_TUBE_PLOT and (Phi_x is not None):
        diff = abs_err  # (N, len(idx))
        tubes = get_trajectory_tubes(Phi_x)
        tube_sizes = np.asarray(tubes[1:])  # (N, nx_full) typically

        # If comparing only a subset, slice tube sizes to the same indices
        tube_sizes = tube_sizes[:, idx]

        Np, nxp = diff.shape
        t = np.arange(Np) * float(dt)

        ncols = 6
        nrows = math.ceil(nxp / ncols)

        fig_w = 18
        fig_h = 3.0 * nrows

        outdir = "tube_vs_diff"
        os.makedirs(outdir, exist_ok=True)
        save_path = os.path.join(outdir, f"tube_vs_diff_modelstep_6perrow_N{Np}_nx{nxp}.png")

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
        axes = np.atleast_2d(axes)

        for j in range(nxp):
            r = j // ncols
            c = j % ncols
            ax = axes[r, c]
            ax.plot(t, tube_sizes[:, j], linewidth=1.2, label="tube")
            ax.plot(t, diff[:, j], linewidth=1.2, label="|x_model - x_pred|")

            # Use semantic state name here
            ax.set_title(state_names_idx[j], fontsize=9)

            ax.grid(True)

        for j in range(nxp, nrows * ncols):
            r = j // ncols
            c = j % ncols
            axes[r, c].axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("Time [s]")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("Tube size vs model deviation (per compared state dimension)", y=0.995)
        plt.tight_layout()
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved tube-vs-diff plot to: {save_path}")
    else:
        if MAKE_TUBE_PLOT and Phi_x is None:
            print("Skipping tube plot: Phi_x not found in NPZ.")

    print("Done.")


if __name__ == "__main__":
    main()
