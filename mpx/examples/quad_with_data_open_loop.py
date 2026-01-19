"""
openloop_go2_rollout_diagnostics.py

Open-loop rollout using the SAME dynamics used in MPC, with detailed diagnostics.

Rollout:
  x_hat[0] = X[0]
  x_hat[i+1] = dynamics(x_hat[i], U[i,:n_joints], i, parameter=parameter)

Compares to nominal logged X[i+1] and reports:
  - base pose error (always on first 7 dims)
  - compare-scope RMS error (either qpos+qvel only, or full state)
  - quaternion norm drift + sign mismatch rate
  - contact switches from parameter
  - GRF spikes (if GRF is in state)
  - NaN/Inf detection and first failure step
  - Top offending indices at spike steps (within compare scope)

Assumptions:
  - NPZ has X, U, parameter (and optionally Phi_x)
  - config_go2 matches the rollout generation, and provides:
      model_path, n_joints, dt, contact_frame, body_name, dynamics,
      (optional) n_contact, grf_as_state
"""

from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from functools import partial

import mujoco
from mujoco import mjx

import mpx.config.config_go2 as config


# -----------------------------
# User config
# -----------------------------
NPZ_PATH = os.path.join("mpc_data", "go2_mpc_rollout.npz")
OUTDIR = "openloop_diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

# Comparison scope:
#   True  -> compare ONLY qpos+qvel (recommended for diagnosing rollout drift)
#   False -> compare full state (includes feet/grf tails if present)
COMPARE_ONLY_QPOS_QVEL = True

# Diagnostics thresholds (tune if needed)
BASE_POSE_ERR_SPIKE = 0.10          # spike threshold on ||err[:7]||
COMPARE_RMS_SPIKE = 1e-2            # spike threshold on RMS over compare scope
QUAT_NORM_TOL = 1e-2                # deviation from 1.0
GRF_NORM_SPIKE = 5e3                # only used for printing/plotting if GRF present
PRINT_TOPK = 8                      # top-k state dims at a spike step

# Plotting
MAKE_PLOTS = True
SAVE_PLOTS = True


# -----------------------------
# Helpers
# -----------------------------
def normalize_quat(q: jnp.ndarray) -> jnp.ndarray:
    return q / (jnp.linalg.norm(q) + 1e-12)

def align_quat_sign(q_model: jnp.ndarray, q_ref: jnp.ndarray) -> jnp.ndarray:
    # q and -q represent same rotation; align sign to minimize component error
    return jnp.where(jnp.dot(q_model, q_ref) < 0.0, -q_model, q_model)

def safe_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)

def block_rms(err_vec: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err_vec)))) if err_vec.size > 0 else 0.0

def find_state_layout(n_joints: int, nx: int):
    """
    Canonical WB layout (matches your wrappers):
      qpos: [0 : 7+n_joints]
      qvel: [7+n_joints : 13+2*n_joints]
    Optional tails (commonly):
      feet world: 12 (4*3)
      grf: 12 (4*3)
    """
    nq = 7 + n_joints
    nv = 6 + n_joints
    qpos_sl = slice(0, nq)
    qvel_sl = slice(nq, nq + nv)

    # partitions
    base_pos_sl = slice(0, 3)
    quat_sl = slice(3, 7)
    joint_q_sl = slice(7, nq)

    base_vel_sl = slice(nq, nq + 6)
    joint_dq_sl = slice(nq + 6, nq + nv)

    tail_start = nq + nv
    remaining = nx - tail_start

    feet_sl = None
    grf_sl = None
    if remaining >= 24:
        feet_sl = slice(tail_start, tail_start + 12)
        grf_sl = slice(tail_start + 12, tail_start + 24)

    return dict(
        nq=nq,
        nv=nv,
        qpos_sl=qpos_sl,
        qvel_sl=qvel_sl,
        base_pos_sl=base_pos_sl,
        quat_sl=quat_sl,
        joint_q_sl=joint_q_sl,
        base_vel_sl=base_vel_sl,
        joint_dq_sl=joint_dq_sl,
        feet_sl=feet_sl,
        grf_sl=grf_sl,
    )

def make_compare_mask(layout: dict, nx: int, compare_only_qpos_qvel: bool) -> np.ndarray:
    """Boolean mask over state dims to include in error comparisons."""
    if not compare_only_qpos_qvel:
        return np.ones(nx, dtype=bool)
    mask = np.zeros(nx, dtype=bool)
    mask[layout["qpos_sl"]] = True
    mask[layout["qvel_sl"]] = True
    return mask

def savefig(path: str):
    if SAVE_PLOTS:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {path}")
        plt.close()
    else:
        plt.show()


def main():
    assert os.path.exists(NPZ_PATH), f"Missing file: {NPZ_PATH}"
    data = np.load(NPZ_PATH)
    print("NPZ arrays:", data.files)

    X = data["X"]  # (N+1, nx)
    U = data["U"]  # (N, nu)
    if "parameter" not in data.files:
        raise RuntimeError("NPZ must contain 'parameter' for open-loop dynamics rollout.")
    parameter_np = data["parameter"]

    N = U.shape[0]
    nx = X.shape[1]
    n_joints = int(config.n_joints)
    dt = float(config.dt)

    # Parameter length sanity
    if parameter_np.shape[0] == N:
        param_len = "N"
    elif parameter_np.shape[0] == N + 1:
        param_len = "N+1"
    else:
        raise RuntimeError(f"parameter has incompatible length: {parameter_np.shape[0]} vs N={N}")

    # Build dynamics exactly like wrapper
    muj_model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(muj_model)

    contact_id = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.contact_frame]
    body_id = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in config.body_name]

    if any(int(i) < 0 for i in contact_id):
        raise RuntimeError(f"Failed to resolve some contact_frame geoms: {config.contact_frame}")
    if any(int(i) < 0 for i in body_id):
        raise RuntimeError(f"Failed to resolve some body_name bodies: {config.body_name}")

    dynamics = partial(config.dynamics, muj_model, mjx_model, contact_id, body_id, n_joints, dt)

    # JIT step (pass full parameter array; dynamics uses parameter[t,...] internally)
    @jax.jit
    def step_dynamics(x, u, t, parameter_all):
        x_next = dynamics(x, u, t, parameter=parameter_all)
        # quaternion hygiene (rollout stability; does not change represented rotation)
        x_next = x_next.at[3:7].set(normalize_quat(x_next[3:7]))
        return x_next

    layout = find_state_layout(n_joints, nx)
    compare_mask = make_compare_mask(layout, nx, COMPARE_ONLY_QPOS_QVEL)
    compare_idx = np.where(compare_mask)[0]

    # Contact schedule from parameter (first 4 dims)
    contact_from_param = parameter_np[:N, :4].copy() if parameter_np.shape[1] >= 4 else None

    print("\n--- Setup ---")
    print(f"Loaded: {NPZ_PATH}")
    print(f"X: {X.shape}, U: {U.shape}, parameter: {parameter_np.shape} ({param_len})")
    print(f"n_joints={n_joints}, dt={dt}, nx={nx}")
    print(f"contact_id={list(map(int, contact_id))}")
    print(f"body_id   ={list(map(int, body_id))}")
    print(f"Assumed layout: nq={layout['nq']}, nv={layout['nv']}, tail={nx-(layout['nq']+layout['nv'])}")
    print(f"Compare scope: {'qpos+qvel' if COMPARE_ONLY_QPOS_QVEL else 'full state'} "
          f"({compare_idx.size}/{nx} dims)")

    if COMPARE_ONLY_QPOS_QVEL and (layout["grf_sl"] is not None):
        tail0 = layout["nq"] + layout["nv"]
        print(f"Note: GRF/feet tail detected (dims [{tail0}:{nx}]) but will be ignored in comparisons.")

    # Convert to JAX
    X_j = jnp.asarray(X, dtype=jnp.float64)
    U_j = jnp.asarray(U, dtype=jnp.float64)
    parameter_j = jnp.asarray(parameter_np, dtype=jnp.float64)
    U_tau = U_j[:, :n_joints]

    # Logs
    xhat = np.zeros((N + 1, nx), dtype=np.float64)
    xhat[0] = safe_np(X_j[0])

    base_pose_err = np.zeros(N, dtype=np.float64)
    compare_rms_err = np.zeros(N, dtype=np.float64)

    quat_norm = np.zeros(N + 1, dtype=np.float64)
    quat_sign_flip = np.zeros(N + 1, dtype=np.int32)
    nan_or_inf = np.zeros(N + 1, dtype=np.int32)

    # Block RMS errors within qpos/qvel (always meaningful)
    rms_base_pos = np.zeros(N, dtype=np.float64)
    rms_quat = np.zeros(N, dtype=np.float64)
    rms_joint_q = np.zeros(N, dtype=np.float64)
    rms_base_vel = np.zeros(N, dtype=np.float64)
    rms_joint_dq = np.zeros(N, dtype=np.float64)

    # Optional tails (only computed/compared if full-state compare)
    rms_feet = np.zeros(N, dtype=np.float64) if (not COMPARE_ONLY_QPOS_QVEL and layout["feet_sl"] is not None) else None
    rms_grf = np.zeros(N, dtype=np.float64) if (not COMPARE_ONLY_QPOS_QVEL and layout["grf_sl"] is not None) else None
    grf_norm = np.zeros(N + 1, dtype=np.float64) if (layout["grf_sl"] is not None) else None

    first_bad = None

    # Rollout
    for i in range(N):
        x_i = jnp.asarray(xhat[i], dtype=jnp.float64)
        u_i = U_tau[i]

        x_next = step_dynamics(x_i, u_i, i, parameter_j)
        x_next_np = safe_np(x_next)
        xhat[i + 1] = x_next_np

        # validity
        if not np.all(np.isfinite(x_next_np)):
            nan_or_inf[i + 1] = 1
            if first_bad is None:
                first_bad = i

        # quat diagnostics
        qhat = x_next_np[3:7]
        quat_norm[i + 1] = np.linalg.norm(qhat)

        qnom = safe_np(X_j[i + 1, 3:7])
        quat_sign_flip[i + 1] = 1 if (np.dot(qhat, qnom) < 0.0) else 0

        # error vs nominal (sign-align quat for error computation only)
        qhat_aligned = safe_np(align_quat_sign(jnp.asarray(qhat), jnp.asarray(qnom)))
        x_next_for_err = x_next_np.copy()
        x_next_for_err[3:7] = qhat_aligned

        err_full = x_next_for_err - safe_np(X_j[i + 1])

        # base pose error (always)
        base_pose_err[i] = float(np.linalg.norm(err_full[:7]))

        # compare-scope RMS
        compare_rms_err[i] = block_rms(err_full[compare_idx])

        # block RMS (qpos/qvel blocks)
        rms_base_pos[i] = block_rms(err_full[layout["base_pos_sl"]])
        rms_quat[i]     = block_rms(err_full[layout["quat_sl"]])
        rms_joint_q[i]  = block_rms(err_full[layout["joint_q_sl"]])
        rms_base_vel[i] = block_rms(err_full[layout["base_vel_sl"]])
        rms_joint_dq[i] = block_rms(err_full[layout["joint_dq_sl"]])

        # tail diagnostics
        if grf_norm is not None:
            grf_vec = x_next_np[layout["grf_sl"]]
            grf_norm[i + 1] = float(np.linalg.norm(grf_vec))

        if rms_feet is not None:
            rms_feet[i] = block_rms(err_full[layout["feet_sl"]])
        if rms_grf is not None:
            rms_grf[i] = block_rms(err_full[layout["grf_sl"]])

        # Spike report trigger
        spike = (
            (base_pose_err[i] > BASE_POSE_ERR_SPIKE) or
            (compare_rms_err[i] > COMPARE_RMS_SPIKE) or
            (nan_or_inf[i + 1] == 1) or
            ((grf_norm is not None) and (grf_norm[i + 1] > GRF_NORM_SPIKE) and (not COMPARE_ONLY_QPOS_QVEL))
        )

        if spike:
            # Masked top-k within compare scope
            err = err_full.copy()
            err[~compare_mask] = 0.0
            abs_err = np.abs(err)
            topk = np.argsort(-abs_err)[:PRINT_TOPK]

            print("\n--- Spike/Failure Report ---")
            print(f"step i={i}")
            print(f"base_pose_err={base_pose_err[i]:.3e}")
            print(f"compare_rms_err={compare_rms_err[i]:.3e} (scope={'qpos+qvel' if COMPARE_ONLY_QPOS_QVEL else 'full'})")
            print(f"quat_norm={quat_norm[i+1]:.6f} (dev={abs(quat_norm[i+1]-1.0):.3e}) sign_flip={quat_sign_flip[i+1]}")
            if contact_from_param is not None:
                print(f"contact(parameter[{i},:4])={contact_from_param[i]}")
            if grf_norm is not None:
                print(f"grf_norm={grf_norm[i+1]:.3e}")
            if nan_or_inf[i + 1] == 1:
                print("NaN/Inf detected in x_next.")

            print("top |err| dims (within compare scope):")
            for k in topk:
                if not compare_mask[k]:
                    continue
                print(f"  idx {k:3d}: err={err_full[k]:+.3e}  (xhat={x_next_for_err[k]:+.3e}, xnom={safe_np(X_j[i+1,k]):+.3e})")

            if first_bad is None:
                first_bad = i

    # Summary
    print("\n--- Summary ---")
    print(f"mean base_pose_err: {base_pose_err.mean():.3e}")
    print(f"max  base_pose_err: {base_pose_err.max():.3e}")
    print(f"mean compare_rms_err: {compare_rms_err.mean():.3e}")
    print(f"max  compare_rms_err: {compare_rms_err.max():.3e}")
    print(f"quat_norm mean: {quat_norm[1:].mean():.6f}, max dev: {np.max(np.abs(quat_norm[1:]-1.0)):.3e}")
    print(f"quat sign flips fraction: {quat_sign_flip[1:].mean():.3f}")
    if grf_norm is not None:
        print(f"grf_norm mean: {grf_norm[1:].mean():.3e}, max: {grf_norm[1:].max():.3e}")
    if first_bad is not None:
        print(f"first spike/failure step (heuristic): {first_bad}")
    else:
        print("no spikes above thresholds detected.")

    # Save diagnostics NPZ
    out_npz = os.path.join(OUTDIR, "openloop_rollout_diagnostics.npz")
    np.savez(
        out_npz,
        xhat=xhat,
        base_pose_err=base_pose_err,
        compare_rms_err=compare_rms_err,
        quat_norm=quat_norm,
        quat_sign_flip=quat_sign_flip,
        nan_or_inf=nan_or_inf,
        rms_base_pos=rms_base_pos,
        rms_quat=rms_quat,
        rms_joint_q=rms_joint_q,
        rms_base_vel=rms_base_vel,
        rms_joint_dq=rms_joint_dq,
        contact_from_param=contact_from_param if contact_from_param is not None else np.zeros((0,)),
        grf_norm=grf_norm if grf_norm is not None else np.zeros((0,)),
    )
    print(f"Saved diagnostics NPZ: {out_npz}")

    # Plots
    if MAKE_PLOTS:
        t = np.arange(N) * dt

        plt.figure()
        plt.plot(t, base_pose_err, label="||err[:7]||")
        plt.plot(t, compare_rms_err, label="compare RMS")
        plt.xlabel("Time [s]")
        plt.ylabel("Error")
        plt.title("Open-loop rollout error vs nominal")
        plt.grid(True)
        plt.legend()
        savefig(os.path.join(OUTDIR, "base_pose_and_compare_rms.png"))

        plt.figure()
        plt.plot(t, rms_base_pos, label="base pos rms")
        plt.plot(t, rms_quat, label="quat rms")
        plt.plot(t, rms_joint_q, label="joint q rms")
        plt.plot(t, rms_base_vel, label="base vel rms")
        plt.plot(t, rms_joint_dq, label="joint dq rms")
        if (not COMPARE_ONLY_QPOS_QVEL) and (rms_feet is not None):
            plt.plot(t, rms_feet, label="feet rms")
        if (not COMPARE_ONLY_QPOS_QVEL) and (rms_grf is not None):
            plt.plot(t, rms_grf, label="grf rms")
        plt.xlabel("Time [s]")
        plt.ylabel("RMS error")
        plt.title("Block-wise RMS errors (open-loop vs nominal)")
        plt.grid(True)
        plt.legend()
        savefig(os.path.join(OUTDIR, "block_rms_errors.png"))

        plt.figure()
        plt.plot(np.arange(N + 1) * dt, quat_norm, label="||quat||")
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Time [s]")
        plt.ylabel("Quaternion norm")
        plt.title("Quaternion norm drift during open-loop rollout")
        plt.grid(True)
        plt.legend()
        savefig(os.path.join(OUTDIR, "quat_norm.png"))

        if contact_from_param is not None:
            plt.figure()
            for k in range(contact_from_param.shape[1]):
                plt.plot(t, contact_from_param[:, k], label=f"contact[{k}]")
            plt.xlabel("Time [s]")
            plt.ylabel("Contact flag")
            plt.title("Contact schedule from parameter[:, :4]")
            plt.grid(True)
            plt.legend()
            savefig(os.path.join(OUTDIR, "contact_flags.png"))

        if grf_norm is not None:
            plt.figure()
            plt.plot(np.arange(N + 1) * dt, grf_norm, label="||grf||")
            plt.xlabel("Time [s]")
            plt.ylabel("GRF norm")
            plt.title("GRF norm during open-loop rollout")
            plt.grid(True)
            plt.legend()
            savefig(os.path.join(OUTDIR, "grf_norm.png"))

    print("Done.")


if __name__ == "__main__":
    main()
