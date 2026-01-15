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

from jax import config
config.update("jax_enable_x64", True)

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


# -----------------------------
# Configuration
# -----------------------------
NPZ_PATH = os.path.join("mpc_data", "go2_mpc_rollout.npz")

COMPARE_ONLY_QPOS_QVEL = True
MAKE_TUBE_PLOT = True


def main():
    assert os.path.exists(NPZ_PATH), f"Missing file: {NPZ_PATH}"

    data = np.load(NPZ_PATH)
    print("NPZ arrays:", data.files)

    X = data["X"]  # (N+1, nx)
    U = data["U"]  # (N, nu)
    Phi_x = data["Phi_x"] if "Phi_x" in data.files else None

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
        idx = np.arange(nq + nv)
        print(f"Comparing qpos+qvel only: {len(idx)} dims.")
    else:
        idx = np.arange(nx)
        print(f"Comparing full state: {len(idx)} dims.")

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
    base_pose_err = np.zeros((N,), dtype=np.float64)

    for i in range(N):
        x_i = X_j[i]
        u_i = U_tau[i]

        x_model_next = step_dynamics(x_i, u_i, i, parameter_j)

        def align_quat_sign(q_model, q_ref):
            # both shape (4,)
            return jnp.where(jnp.dot(q_model, q_ref) < 0.0, -q_model, q_model)

        # inside loop after x_model_next computed:
        # q_m = x_model_next[3:7]
        # q_r = X_j[i+1, 3:7]
        # q_m = align_quat_sign(q_m, q_r)
        # x_model_next = x_model_next.at[3:7].set(q_m)

        x_pred = np.asarray(X[i + 1], dtype=np.float64)
        x_act = np.asarray(x_model_next, dtype=np.float64)

        abs_err[i] = np.abs(x_act[idx] - x_pred[idx])
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
            ax.set_title(f"idx[{j}] (state[{int(idx[j])}])", fontsize=9)
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
