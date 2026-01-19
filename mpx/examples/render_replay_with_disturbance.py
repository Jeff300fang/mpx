"""
render_replay_go2_modelstep_with_disturbance.py

Interactive MuJoCo viewer that renders either:
  - stored rollout states X (teleport), OR
  - model-stepped rollout using the exact MJX dynamics wrapper used by MPC,
    optionally with additive disturbance: x_{k+1} = f(x_k,u_k) + E w_k.

Disturbance is generated in NumPy (np.random.Generator) and then added to the
JAX state (x) each step.

Run:
  python render_replay_go2_modelstep_with_disturbance.py --npz mpc_data/go2_mpc_rollout.npz

Modes:
  --render stored      : render stored X trajectory (teleport)
  --render modelstep   : render model-stepped trajectory (default)

Disturbance:
  --disturb off|sample_ball|fixed
  --disturb-seed 0
  --disturb-ediag 0.004
  --disturb-fixed-index 1
  --disturb-fixed-value -1.0

Notes:
- This assumes your state packs qpos then qvel at least:
    nq = 7 + n_joints
    nv = 6 + n_joints
    x[:nq] -> qpos, x[nq:nq+nv] -> qvel
- By default, E = ediag * I(nx), i.e., disturbance affects all state dims.
  If you want to disturb only parts of state (recommended), modify `build_E(...)`.
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
    Convention:
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
    where parameter_all is the whole parameter array, passed as keyword `parameter=...`,
    matching your MPC usage.
    """
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)

    n_joints = int(config.n_joints)
    dt = float(config.dt)

    contact_id = []
    for name in config.contact_frame:
        contact_id.append(mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name))

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
        return dynamics(x, u, t, parameter=parameter_all)

    return step, model, n_joints, dt


# -----------------------------
# Disturbance helpers (NumPy RNG)
# -----------------------------
def sample_uniform_l2_ball_np(rng: np.random.Generator, dim: int) -> np.ndarray:
    """
    Sample w ~ Uniform({w: ||w||_2 <= 1}) in R^dim via:
      z ~ N(0,I),  dir = z/||z||,  r ~ U(0,1)^(1/dim),  w = r*dir
    """
    z = rng.normal(size=(dim,))
    z /= (np.linalg.norm(z) + 1e-12)
    r = rng.uniform(0.0, 1.0) ** (1.0 / float(dim))
    return r * z


def build_E(nx: int, ediag: float) -> np.ndarray:
    """
    Default: E = ediag * I(nx).

    If you want disturbance only in a sub-block (recommended), change this, e.g.:
      - only qvel: E[nq:nq+nv, nq:nq+nv] = ediag * I(nv)
      - only joint velocities, etc.
    """
    return float(ediag) * np.eye(nx, dtype=np.float64)


def apply_disturbance_after_step(
    rng: np.random.Generator,
    x_j: jnp.ndarray,          # (nx,)
    E_j: jnp.ndarray,          # (nx, nw)
    disturb_mode: str,         # "off" | "sample_ball" | "fixed"
    w_fixed_np: np.ndarray | None,
) -> tuple[jnp.ndarray, np.ndarray]:
    """
    Returns (x_j_plus, w_np) where w_np is the disturbance drawn/used in NumPy.
    """
    if disturb_mode == "off":
        w_np = np.zeros((int(E_j.shape[1]),), dtype=np.float64)
        return x_j, w_np

    nw = int(E_j.shape[1])

    if disturb_mode == "sample_ball":
        w_np = sample_uniform_l2_ball_np(rng, nw)
    elif disturb_mode == "fixed":
        if w_fixed_np is None:
            raise ValueError("disturb_mode='fixed' requires w_fixed_np.")
        w_np = np.asarray(w_fixed_np, dtype=np.float64).reshape(nw,)
    else:
        raise ValueError(f"Unknown disturb_mode={disturb_mode!r}")

    x_j = x_j + E_j @ jnp.asarray(w_np, dtype=x_j.dtype)
    return x_j, w_np


# -----------------------------
# Viewer loop
# -----------------------------
def run_viewer(
    npz_path: str,
    render_mode: str,
    realtime: bool,
    stride: int,
    quat_smoothing: bool,
    disturb_mode: str,
    disturb_seed: int,
    disturb_ediag: float,
    disturb_fixed_index: int,
    disturb_fixed_value: float,
):
    assert os.path.exists(npz_path), f"Missing NPZ: {npz_path}"

    npz = np.load(npz_path)
    if "X" not in npz.files or "U" not in npz.files:
        raise RuntimeError(f"NPZ must contain X and U. Found: {npz.files}")
    if "parameter" not in npz.files:
        raise RuntimeError("NPZ missing required array 'parameter' needed by dynamics.")

    X = npz["X"]                    # (N+1, nx)
    U = npz["U"]                    # (N, nu)
    parameter_np = npz["parameter"] # (N+1, p_dim) or similar

    N = U.shape[0]
    nx = X.shape[1]

    # Build dynamics step + load mujoco model for rendering
    step_mjx, mj_model, n_joints, dt = build_mjx_step()
    data = mujoco.MjData(mj_model)

    # Convert arrays to JAX
    X_j = jnp.asarray(X, dtype=jnp.float64)
    U_j = jnp.asarray(U, dtype=jnp.float64)
    parameter_j = jnp.asarray(parameter_np, dtype=jnp.float64)

    # Torques (first n_joints)
    U_tau = U_j[:, :n_joints]

    # Disturbance setup
    rng = np.random.default_rng(int(disturb_seed))
    E_np = build_E(nx=nx, ediag=float(disturb_ediag))
    E_j = jnp.asarray(E_np, dtype=jnp.float64)

    w_fixed_np = None
    if disturb_mode == "fixed":
        w_fixed_np = np.zeros((E_np.shape[1],), dtype=np.float64)
        idx = int(disturb_fixed_index)
        w_fixed_np[idx] = float(disturb_fixed_value)

    print(f"Loaded NPZ: {npz_path}")
    print(f"X shape={X.shape}, U shape={U.shape}, parameter shape={parameter_np.shape}")
    print(f"Render mode: {render_mode}")
    print(f"n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"Disturbance: mode={disturb_mode}, seed={disturb_seed}, E=({E_np.shape[0]}x{E_np.shape[1]}) diag={disturb_ediag}")
    if disturb_mode == "fixed":
        print(f"Fixed w: index={disturb_fixed_index}, value={disturb_fixed_value}")
    print("Close the viewer window to exit.")

    prev_quat = None

    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        i = 0
        x_i = X_j[0]  # model-stepped state

        while viewer.is_running():
            # Choose visualization state
            if render_mode == "stored":
                x_vis = np.asarray(X[i], dtype=np.float64)
            elif render_mode == "modelstep":
                x_vis = np.asarray(x_i, dtype=np.float64)
            else:
                raise ValueError(f"Unknown render_mode: {render_mode!r}")

            # Teleport MuJoCo to qpos/qvel extracted from x_vis
            qpos, qvel = extract_qpos_qvel_from_x(x_vis, n_joints=n_joints)

            if quat_smoothing and qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if prev_quat is not None:
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            teleport_mujoco(mj_model, data, qpos, qvel)
            viewer.sync()

            # Advance frame index
            i_next = i + int(stride)
            if i_next >= (N + 1):
                # restart
                i = 0
                prev_quat = None
                x_i = X_j[0]
                continue

            # If modelstep, roll forward dynamics (and disturbance) for stride steps
            if render_mode == "modelstep":
                for k in range(i, min(i_next, N)):
                    # nominal MJX step
                    x_i = step_mjx(x_i, U_tau[k], k, parameter_j)

                    # additive disturbance AFTER nominal step
                    x_i, _w_k = apply_disturbance_after_step(
                        rng=rng,
                        x_j=x_i,
                        E_j=E_j,
                        disturb_mode=disturb_mode,
                        w_fixed_np=w_fixed_np,
                    )

            i = i_next

            if realtime:
                time.sleep(float(dt) * float(stride))


def parse_args():
    p = argparse.ArgumentParser(description="Render stored or model-stepped replay in an interactive MuJoCo viewer (with optional disturbance).")
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

    # Disturbance args
    p.add_argument("--disturb", type=str, choices=["off", "sample_ball", "fixed"], default="off",
                   help="Additive disturbance mode applied after each dynamics step.")
    p.add_argument("--disturb-seed", type=int, default=0, help="RNG seed for NumPy disturbance.")
    p.add_argument("--disturb-ediag", type=float, default=0.004, help="E = ediag * I(nx) by default.")
    p.add_argument("--disturb-fixed-index", type=int, default=1, help="Index for fixed disturbance vector w.")
    p.add_argument("--disturb-fixed-value", type=float, default=-1.0, help="Value at that index for fixed w.")
    return p.parse_args()


def main():
    args = parse_args()
    run_viewer(
        npz_path=args.npz,
        render_mode=args.render,
        realtime=bool(args.realtime),
        stride=int(args.stride),
        quat_smoothing=bool(args.quat_smoothing),
        disturb_mode=str(args.disturb),
        disturb_seed=int(args.disturb_seed),
        disturb_ediag=float(args.disturb_ediag),
        disturb_fixed_index=int(args.disturb_fixed_index),
        disturb_fixed_value=float(args.disturb_fixed_value),
    )


if __name__ == "__main__":
    main()
