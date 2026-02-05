#!/usr/bin/env python3
"""
render_h1_rollout_via_model_dynamics.py

Render a saved H1 rollout NPZ by *re-simulating* it using the SAME dynamics
function used inside MPC, then visualizing the resulting states in an
interactive MuJoCo viewer.

UPDATED BEHAVIOR (per your request):
  - NO disturbance inference from mismatch. The old "infer w" path is removed.
  - If --use-sls-feedback and Phi_u present:
        u_i = u_nom_i + sum_{j=0..i} Phi_u[i,j] @ w_hist[j]
    where w_j is SAMPLED (uniform L2 ball) in the FIRST 3 dims only; rest = 0.
  - And the state is advanced with a disturbed step:
        x_{i+1} = f(x_i, u_i, i, parameter) + E @ w_i
    with E = diag([E_scale]*E_first_k, 0, 0, ...) (nx x nx).
  - If not using SLS feedback, it performs nominal rollout (no disturbance).

Notes:
  - We keep argparse as in your file.
  - The disturbance is injected additively *after* the dynamics step.
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

import mujoco
import mujoco.viewer
from mujoco import mjx

import jax
import jax.numpy as jnp
from functools import partial


import mpx.config.config_h1 as config


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
# X layout handling
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

    qpos = np.array(x_row[start: start + nq], dtype=np.float64, copy=True)
    qvel = np.array(x_row[start + nq: start + nq + nv], dtype=np.float64, copy=True)
    return qpos, qvel


# -----------------------------
# Optional overlay: obstacles + visual-only ground plane
# -----------------------------
def clear_user_geoms(viewer: mujoco.viewer.Handle) -> None:
    viewer.user_scn.ngeom = 0


def add_cylinder_pillar(
    viewer: mujoco.viewer.Handle,
    pos_xyz: np.ndarray,
    radius: float,
    height: float,
    rgba=(1.0, 0.2, 0.2, 0.6),
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([float(radius), float(height) / 2.0, 0.0], dtype=np.float64),
        pos=np.array(pos_xyz, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def add_ground_plane(
    viewer: mujoco.viewer.Handle,
    *,
    z: float = 0.0,
    size_xy: float = 50.0,
    rgba=(0.7, 0.7, 0.7, 1.0),
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_PLANE,
        size=np.array([size_xy, size_xy, 0.0], dtype=np.float64),
        pos=np.array([0.0, 0.0, float(z)], dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


# -----------------------------
# Dynamics instantiation (matches MPC wrapper style)
# -----------------------------
def build_h1_step_dynamics():
    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)

    n_joints = int(getattr(config, "n_joints", 0))
    dt = float(getattr(config, "dt", 0.02))

    # Resolve ids (geoms for contacts, bodies for feet/etc.)
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
# Disturbance + SLS helpers
# -----------------------------
def make_E_diag(nx: int, scale: float = 0.05, first_k: int = 2) -> np.ndarray:
    """
    E is (nx, nx) diagonal, with 'scale' on first_k diagonals, zeros elsewhere.
    """
    E = np.zeros((nx, nx), dtype=np.float64)
    k = int(min(first_k, nx))
    for i in range(k):
        E[i, i] = float(scale)
    return E


def sample_unit_ball_first3(rng: np.random.Generator, n_w: int, radius: float = 1.0) -> np.ndarray:
    """
    w in R^{n_w}:
      - w[0:3] ~ Uniform L2 ball radius 'radius'
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
    w[0] = 0.6
    w[1] = 0.7
    w[2] = 0
    return w


def sls_control_from_history_general(
    u_nom_i: np.ndarray,      # (m,)
    Phi_u_i: np.ndarray,      # (i+1, m, n_w)
    w_hist: np.ndarray,       # (i+1, n_w)
) -> np.ndarray:
    """
    u = u_nom_i + sum_{j=0..i} Phi_u_i[j] @ w_hist[j]
    """
    u = u_nom_i.copy()
    for j in range(Phi_u_i.shape[0]):
        u += Phi_u_i[j] @ w_hist[j]
    return u


def step_dynamics_with_disturbance(
    step_dynamics,            # jitted: (x,u,t,parameter)->x_nom_next (jax array)
    x: np.ndarray,            # (nx,)
    u: np.ndarray,            # (m,)
    t: int,
    parameter,                # array-like
    E: np.ndarray,            # (nx, nx) or (nx, n_w)
    w_t: np.ndarray,          # (n_w,) (or (nx,) if E is square and you pass nx)
) -> np.ndarray:
    """
    x_next = f(x,u,t,parameter) + E @ w_t
    """
    x_nom = step_dynamics(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(u, dtype=jnp.float64),
        int(t),
        jnp.asarray(parameter, dtype=jnp.float64),
    )
    x_nom = np.asarray(x_nom, dtype=np.float64)
    return x_nom + np.asarray(E, dtype=np.float64) @ np.asarray(w_t, dtype=np.float64) * 0.05


# -----------------------------
# Rollout via model dynamics (nominal only)
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
# Rollout via model dynamics + SLS Phi_u feedback + disturbed steps
# -----------------------------
def rollout_with_model_dynamics_sls_feedback_disturbed(
    X0: np.ndarray,                # (nx,)
    U: np.ndarray,                 # (N, nu) nominal controls (tau prefix)
    Phi_u: np.ndarray,             # Phi_u[i,j] -> (m,n_w)
    parameter: np.ndarray,         # (>=N+1, p_dim)
    n_joints: int,
    step_dynamics,                 # jitted
    *,
    use_tau_prefix: bool = True,
    E_scale: float = 0.05,
    E_first_k: int = 2,
    w_radius: float = 1.0,
    w_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_sim: (N+1, nx) simulated states
      U_applied: (N, m) applied controls (nominal + feedback)
      W: (N+1, n_w) sampled disturbances, with W[0]=0
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
    n_w = int(Phi_u.shape[-1])

    # State disturbance mapping (nx x nx); only first_k diagonals nonzero.
    # We'll still pass w_t as length nx by embedding n_w into nx if needed.
    E = make_E_diag(nx=nx, scale=float(E_scale), first_k=int(E_first_k))  # (nx, nx)

    rng = np.random.default_rng(int(w_seed))

    X_sim = np.zeros((N + 1, nx), dtype=np.float64)
    U_applied = np.zeros((N, m), dtype=np.float64)
    W = np.zeros((N + 1, n_w), dtype=np.float64)  # W[0]=0
    N = 30
    w_hist = np.zeros((N + 1, n_w), dtype=np.float64)

    x = X0.copy()
    X_sim[0] = x

    for i in range(N):
        u_nom_i = U[i, :m] if use_tau_prefix else U[i].copy()

        # Sample w_i (first 3 dims only)
        w_i = sample_unit_ball_first3(rng, n_w=n_w, radius=float(w_radius))
        W[i + 1] = w_i
        w_hist[i + 1] = w_i

        # Control uses history w_0..w_i:
        # Phi_u[i,j] multiplies w_j, and w_j is stored in w_hist[j+1].
        Phi_u_i = Phi_u[i, : i + 1]      # (i+1, m, n_w)
        w_used = w_hist[1 : i + 2]       # (i+1, n_w) corresponds to w_0..w_i
        u_i = sls_control_from_history_general(u_nom_i, Phi_u_i, w_used)

        # Disturbed state step uses current w_i.
        # E is (nx,nx) so embed w_i into an nx vector (first n_w entries).
        w_embed = np.zeros((nx,), dtype=np.float64)
        w_embed[: min(n_w, nx)] = w_i[: min(n_w, nx)]
        x_next = step_dynamics_with_disturbance(step_dynamics, x, u_i, i, parameter, E, w_embed)

        U_applied[i] = u_i
        X_sim[i + 1] = x_next
        x = x_next

    return X_sim, U_applied, W


# -----------------------------
# Viewer: render x_sim[t] each frame
# -----------------------------
def run_viewer_model_rollout(
    npz_path: str,
    *,
    realtime: bool,
    stride: int,
    start_index: int,
    quat_sign_smoothing: bool,
    layout: LayoutSpec,
    show_obstacles: bool,
    obstacle_height: float,
    show_ground: bool,
    use_sls_feedback: bool,
    E_scale: float,
    E_first_k: int,
) -> None:
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

    # Build dynamics + MuJoCo model (model is used for rendering too)
    model, n_joints, dt, step_dynamics = build_h1_step_dynamics()

    # Optional arrays
    Phi_u = None
    if "Phi_u" in data_npz.files:
        Phi_u = np.asarray(data_npz["Phi_u"], dtype=np.float64)

    obs = None
    if "obstacles" in data_npz.files:
        obs = np.asarray(data_npz["obstacles"], dtype=np.float64)

    print(f"Loaded NPZ: {npz_path}")
    print(f"Arrays: {list(data_npz.files)}")
    print(f"Stored X shape: {X.shape}, U shape: {U.shape}, parameter shape: {parameter.shape}")
    if Phi_u is not None:
        print(f"Phi_u shape: {Phi_u.shape}")
    if obs is not None:
        print(f"obstacles shape: {obs.shape}")
    print(f"Using config.model_path: {config.model_path}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}")
    print(f"n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"layout={layout}")

    # Re-simulate
    if use_sls_feedback and (Phi_u is not None):
        print("Rolling out with model dynamics + Phi_u feedback + DISTURBED STEPS...")
        # Use the stored initial state, but DO NOT infer w; we sample it inside.
        X_sim, U_applied, W = rollout_with_model_dynamics_sls_feedback_disturbed(
            X0=X[0],
            U=U,
            Phi_u=Phi_u,
            parameter=parameter,
            n_joints=n_joints,
            step_dynamics=step_dynamics,
            use_tau_prefix=True,
            E_scale=float(E_scale),
            E_first_k=int(E_first_k),
            w_radius=1.0,
            w_seed=0,
        )
        print(f"Sim rollout complete: X_sim shape={X_sim.shape}, U_applied shape={U_applied.shape}, W shape={W.shape}")
    else:
        if use_sls_feedback and (Phi_u is None):
            print("NOTE: --use-sls-feedback set, but NPZ has no Phi_u. Falling back to nominal rollout.")
        print("Rolling out with model dynamics (nominal controls only)...")
        X_sim = rollout_with_model_dynamics(
            X0=X[0],
            U=U,
            parameter=parameter,
            n_joints=n_joints,
            step_dynamics=step_dynamics,
            use_tau_prefix=True,
        )
        print(f"Sim rollout complete: X_sim shape={X_sim.shape}")

    # Rendering uses its own MjData
    render_data = mujoco.MjData(model)

    prev_quat: Optional[np.ndarray] = None
    T = int(X_sim.shape[0])
    i = max(0, min(int(start_index), T - 1))

    print("Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, render_data) as viewer:
        while viewer.is_running():
            clear_user_geoms(viewer)

            if show_ground:
                add_ground_plane(viewer, z=0.0, size_xy=50.0, rgba=(0.7, 0.7, 0.7, 1.0))

            if show_obstacles and (obs is not None):
                for k in range(obs.shape[0]):
                    add_cylinder_pillar(
                        viewer,
                        pos_xyz=np.array([obs[k, 0], obs[k, 1], 0.0]),
                        radius=float(obs[k, 2] - 0.3),
                        height=float(obstacle_height),
                    )

            qpos, qvel = extract_qpos_qvel(X_sim[i], model=model, n_joints=n_joints, layout=layout)

            if qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if quat_sign_smoothing and (prev_quat is not None):
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            nq_write = min(qpos.shape[0], render_data.qpos.shape[0])
            nv_write = min(qvel.shape[0], render_data.qvel.shape[0])
            render_data.qpos[:nq_write] = qpos[:nq_write]
            render_data.qvel[:nv_write] = qvel[:nv_write]

            mujoco.mj_forward(model, render_data)
            viewer.sync()

            i_next = i + int(stride)
            if i_next >= T:
                i_next = 0
                prev_quat = None
            i = i_next

            if realtime:
                time.sleep(float(dt) * float(stride))


def parse_args():
    p = argparse.ArgumentParser(
        description="Render H1 rollout by re-simulating with config.dynamics(...) then viewing."
    )
    p.add_argument(
        "--npz",
        type=str,
        default=os.path.join("mpc_data", "h1_mpc_rollout.npz"),
        help="Path to rollout NPZ (must contain X, U, parameter).",
    )

    p.add_argument("--realtime", action="store_true", help="Sleep dt each frame (scaled by stride).")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="Render as fast as possible.")
    p.set_defaults(realtime=True)

    p.add_argument("--start", type=int, default=0, help="Start index into simulated rollout.")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (1=every state).")

    p.add_argument(
        "--layout",
        type=str,
        default="prefix",
        choices=["prefix", "offset", "model_dims_prefix"],
        help="How to read qpos/qvel out of each simulated x[t].",
    )
    p.add_argument("--offset", type=int, default=0, help="Offset into x[t] if --layout=offset.")

    p.add_argument(
        "--no-quat-smoothing",
        dest="quat_smoothing",
        action="store_false",
        help="Disable quaternion sign smoothing (may cause visual popping).",
    )
    p.set_defaults(quat_smoothing=True)

    p.add_argument("--show-obstacles", action="store_true", help="If NPZ has obstacles (n_obs,3), render them.")
    p.add_argument("--obstacle-height", type=float, default=3.0, help="Height (m) for obstacle cylinder pillars.")

    p.add_argument("--show-ground", action="store_true", help="Render a visual-only ground plane via user geoms.")

    p.add_argument(
        "--use-sls-feedback",
        action="store_true",
        help="If NPZ has Phi_u, apply u = u_nom + Phi_u*w_history (w sampled) and disturbed state steps.",
    )
    p.add_argument("--E-scale", type=float, default=0.1, help="Diagonal value for E on the first k diagonals.")
    p.add_argument("--E-first-k", type=int, default=2, help="Number of leading diagonals to set to E-scale.")
    return p.parse_args()


def main():
    jax.config.update("jax_enable_x64", True)

    args = parse_args()
    layout = LayoutSpec(mode=args.layout, offset=int(args.offset))

    run_viewer_model_rollout(
        npz_path=str(args.npz),
        realtime=bool(args.realtime),
        stride=int(args.stride),
        start_index=int(args.start),
        quat_sign_smoothing=bool(args.quat_smoothing),
        layout=layout,
        show_obstacles=bool(args.show_obstacles),
        obstacle_height=float(args.obstacle_height),
        show_ground=bool(args.show_ground),
        use_sls_feedback=bool(args.use_sls_feedback),
        E_scale=float(args.E_scale),
        E_first_k=int(args.E_first_k),
    )


if __name__ == "__main__":
    main()
