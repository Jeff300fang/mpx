"""
render_replay_go2_modelstep.py

Interactive MuJoCo viewer that renders a *model-stepped replay* of a saved rollout.

Instead of teleporting to stored X[i], this script:
  1) Loads X, U, parameter from NPZ
  2) Instantiates the *exact same* MJX dynamics wrapper used by MPC (config.dynamics)
  3) Rolls forward:
        x_{i+1} = dynamics(x_i, u_i, i, parameter=parameter)
  4) Teleports MuJoCo's MjData to the predicted qpos/qvel each frame and renders.

This gives you a visual check of "what the model thinks happens" under the recorded controls.

Assumptions:
  - X: (N+1, nx) with [qpos, qvel, ...]
  - U: (N, nu) where the first n_joints columns are joint torques
  - parameter: (N+1, p_dim) passed to dynamics as keyword argument parameter=...
  - config provides: model_path, n_joints, dt, contact_frame, body_name, dynamics

Run:
  python render_replay_go2_modelstep.py --npz mpc_data/go2_mpc_rollout.npz

Options:
  --render stored      : render stored X trajectory (teleport)
  --render modelstep   : render model-stepped trajectory (default)
  --no-realtime        : render as fast as possible
  --stride K           : render every Kth step
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np

from functools import partial

import jax
import jax.numpy as jnp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import mujoco
import mujoco.viewer
from mujoco import mjx

import mpx.config.config_go2 as config


# -----------------------------
# Quaternion utilities
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
# State extraction / teleport
# -----------------------------
def extract_qpos_qvel_from_x(x: np.ndarray, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses the same convention as your replay script:
      nq = 7 + n_joints
      nv = 6 + n_joints
      x[:nq] -> qpos, x[nq:nq+nv] -> qvel
    """
    nq = 7 + int(n_joints)
    nv = 6 + int(n_joints)
    if x.shape[0] < nq + nv:
        raise ValueError(f"x is too small for nq+nv: need {nq+nv}, got {x.shape[0]}")
    qpos = np.array(x[:nq], dtype=np.float64, copy=True)
    qvel = np.array(x[nq:nq+nv], dtype=np.float64, copy=True)
    return qpos, qvel


def teleport_mujoco(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray):
    """
    Teleport MuJoCo to given qpos/qvel and run mj_forward for consistent rendering.
    """
    # Normalize base quaternion in qpos[3:7] (pos(3) then quat(4))
    if qpos.shape[0] >= 7:
        qpos = qpos.copy()
        qpos[3:7] = normalize_quat(qpos[3:7])

    nq_write = min(qpos.shape[0], data.qpos.shape[0])
    nv_write = min(qvel.shape[0], data.qvel.shape[0])

    data.qpos[:nq_write] = qpos[:nq_write]
    data.qvel[:nv_write] = qvel[:nv_write]

    mujoco.mj_forward(model, data)


# -----------------------------
# MJX dynamics instantiation (matches MPC wrapper)
# -----------------------------
def build_mjx_step():
    """
    Builds a jitted single-step function:
        step(x, u, t, parameter_all) -> x_next
    where parameter_all is the whole parameter array so step can index parameter_all[t] etc.
    (Your config.dynamics appears to accept parameter=parameter_all.)
    """
    # JAX compilation cache settings (same as your replay script)
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

    dynamics = partial(
        config.dynamics,
        model, mjx_model,
        contact_id, body_id,
        n_joints, dt
    )

    @partial(jax.jit, static_argnums=())
    def step(x, u, t, parameter_all):
        # Important: parameter_all is passed as keyword like in your MPC
        return dynamics(x, u, t, parameter=parameter_all)

    return step, model, n_joints, dt


# -----------------------------
# Viewer loop
# -----------------------------
def run_viewer(npz_path: str, render_mode: str, realtime: bool, stride: int, quat_smoothing: bool):
    assert os.path.exists(npz_path), f"Missing NPZ: {npz_path}"

    npz = np.load(npz_path)
    if "X" not in npz.files or "U" not in npz.files:
        raise RuntimeError(f"NPZ must contain X and U. Found: {npz.files}")
    if "parameter" not in npz.files:
        raise RuntimeError("NPZ missing required array 'parameter' needed by dynamics.")

    X = npz["X"]              # (N+1, nx)
    U = npz["U"]              # (N, nu)
    parameter_np = npz["parameter"]  # (N+1, p_dim) or similar

    N = U.shape[0]
    nx = X.shape[1]

    # Build dynamics step + load mujoco model for rendering
    step_mjx, mj_model, n_joints, dt = build_mjx_step()

    # Separate MuJoCo MjData used only for rendering
    data = mujoco.MjData(mj_model)

    # Convert arrays to JAX
    X_j = jnp.asarray(X, dtype=jnp.float64)
    U_j = jnp.asarray(U, dtype=jnp.float64)
    parameter_j = jnp.asarray(parameter_np, dtype=jnp.float64)

    # Torques (first n_joints)
    U_tau = U_j[:, :n_joints]

    print(f"Loaded NPZ: {npz_path}")
    print(f"X shape={X.shape}, U shape={U.shape}, parameter shape={parameter_np.shape}")
    print(f"Render mode: {render_mode}")
    print(f"n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print("Close the viewer window to exit.")

    prev_quat = None
    N = 150
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        i = 0

        # x_i for model-stepped rollout
        x_i = X_j[0]

        while viewer.is_running():
            if render_mode == "stored":
                # Teleport to stored state
                x_vis = np.asarray(X[i], dtype=np.float64)
            elif render_mode == "modelstep":
                # Teleport to current predicted state x_i
                x_vis = np.asarray(x_i, dtype=np.float64)
            else:
                raise ValueError(f"Unknown render_mode: {render_mode}")

            # Extract qpos/qvel and teleport MuJoCo
            qpos, qvel = extract_qpos_qvel_from_x(x_vis, n_joints=n_joints)

            # Optional quaternion sign smoothing for visual continuity
            if quat_smoothing and qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if prev_quat is not None:
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            teleport_mujoco(mj_model, data, qpos, qvel)
            viewer.sync()

            # Advance time index and (if modelstep) roll forward dynamics
            i_next = i + int(stride)
            if i_next >= (N + 1):
                # restart
                i = 0
                prev_quat = None
                x_i = X_j[0]
                continue

            # If modelstep, roll x forward stride steps so visualization matches "frame skipping"
            if render_mode == "modelstep":
                # We need to advance x_i from current i to i_next by applying controls.
                # Controls exist for steps 0..N-1 (U_tau index).
                # If stride>1, apply multiple steps.
                for k in range(i, min(i_next, N)):
                    x_i = step_mjx(x_i, U_tau[k], k, parameter_j)

            i = i_next

            if realtime:
                time.sleep(float(dt) * float(stride))


def parse_args():
    p = argparse.ArgumentParser(description="Render stored or model-stepped replay in an interactive MuJoCo viewer.")
    p.add_argument("--npz", type=str, default=os.path.join("mpc_data", "go2_mpc_rollout.npz"),
                   help="Path to rollout npz (must contain X, U, parameter).")
    p.add_argument("--render", type=str, choices=["modelstep", "stored"], default="modelstep",
                   help="Render 'modelstep' (MJX dynamics rollout) or 'stored' (teleport to X).")
    p.add_argument("--stride", type=int, default=1, help="Render every Kth step.")
    p.add_argument("--realtime", action="store_true", help="Sleep dt each frame (scaled by stride).")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="Render as fast as possible.")
    p.set_defaults(realtime=True)
    p.add_argument("--no-quat-smoothing", dest="quat_smoothing", action="store_false",
                   help="Disable quaternion sign smoothing.")
    p.set_defaults(quat_smoothing=True)
    return p.parse_args()


def main():
    args = parse_args()
    run_viewer(
        npz_path=args.npz,
        render_mode=args.render,
        realtime=bool(args.realtime),
        stride=int(args.stride),
        quat_smoothing=bool(args.quat_smoothing),
    )


if __name__ == "__main__":
    main()
