"""
render_h1_rollout_via_model_dynamics.py

Render a saved H1 rollout NPZ by *re-simulating* it using the SAME dynamics
function used inside MPC, then visualizing the resulting states in an
interactive MuJoCo viewer.

Workflow:
  - Load X, U, parameter from NPZ
  - Build dynamics = partial(config.dynamics, model, mjx_model, contact_id, body_id, n_joints, dt)
  - Rollout x_sim[0]=X[0], then x_sim[t+1]=dynamics(x_sim[t], u[t], t, parameter=parameter)
  - Render x_sim[t] in mujoco.viewer (write qpos/qvel into data, mj_forward, sync)

Assumptions:
  - config_h1 provides: model_path, n_joints, dt, contact_frame, body_name, dynamics(...)
  - NPZ contains:
      X: (N+1, nx)
      U: (N, nu)
      parameter: (>=N+1, p_dim) or at least indexable by t used in dynamics
    (optional) obstacles: (n_obs, 3) as [cx, cy, r] for overlay

Run:
  python render_h1_rollout_via_model_dynamics.py --npz mpc_data/h1_mpc_rollout.npz
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
import numpy as np

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


def infer_disturbance(E_i: np.ndarray,
                      x_next_act: np.ndarray,
                      x_next_nom: np.ndarray) -> np.ndarray:
    """
    Solve E_i w = (x_next_act - x_next_nom) in least-squares sense.
    E_i: (n, n_w) or (n, n) in your setup.
    """
    rhs = x_next_act - x_next_nom
    # least-squares / pseudoinverse (stable for debug)
    w_i, *_ = np.linalg.lstsq(E_i, rhs, rcond=None)
    return w_i

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

    qpos = np.array(x_row[start : start + nq], dtype=np.float64, copy=True)
    qvel = np.array(x_row[start + nq : start + nq + nv], dtype=np.float64, copy=True)
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
    # (Optional) JAX cache knobs like you used elsewhere
    # jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    # jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    # jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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
        raise RuntimeError(f"Failed to resolve some contact_frame geoms: {getattr(config,'contact_frame',None)}")
    if any(int(i) < 0 for i in body_id):
        raise RuntimeError(f"Failed to resolve some body_name bodies: {getattr(config,'body_name',None)}")

    dynamics = partial(
        config.dynamics,
        model, mjx_model,
        contact_id, body_id,
        n_joints, dt
    )

    @jax.jit
    def step_dynamics(x, u, t, parameter):
        # Keep keyword "parameter=" to match your wrapper call style
        return dynamics(x, u, t, parameter=parameter)

    return model, n_joints, dt, step_dynamics


# -----------------------------
# Rollout via model dynamics (optionally compare to stored X)
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

    # JAX arrays
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

def sls_control_from_history(U_nom: np.ndarray,
                            Phi_u: np.ndarray,
                            w_hist: np.ndarray,
                            i: int) -> np.ndarray:
    """
    U_nom: (N, m)
    Phi_u: typically (N, N+1, m, n_w)  (or something equivalent)
    w_hist: (N, n_w) but only entries [0..i-1] are used
    """
    u = U_nom[i].copy()

    # Sum_{j < i} Phi_u[i, j+1] @ w_hist[j]
    # (j+1 matches your snippet)
    if i > 0:
        # robust loop (fast enough for debug)
        for j in range(i):
            u += Phi_u[i, j + 1] @ w_hist[j]
    return u


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

    N = int(U.shape[0])
    nx = int(X.shape[1])

    # Build dynamics + MuJoCo model (model is used for rendering too)
    model, n_joints, dt, step_dynamics = build_h1_step_dynamics()

    # Re-simulate from X[0]
    print(f"Loaded NPZ: {npz_path}")
    print(f"Arrays: {list(data_npz.files)}")
    print(f"Stored X shape: {X.shape}, U shape: {U.shape}, parameter shape: {parameter.shape}")
    print(f"Using config.model_path: {config.model_path}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}")
    print(f"n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"layout={layout}")
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

    # Optional quick one-step consistency check vs stored X
    # (Useful to detect you’re not matching the same state layout/parameter semantics)
    if X.shape[0] >= 2:
        err0 = np.linalg.norm(X_sim[1, :min(nx, X.shape[1])] - X[1, :min(nx, X.shape[1])])
        print(f"Sanity: ||X_sim[1]-X_stored[1]|| = {err0:.6e} (first min(nx) dims)")

    # Rendering uses its own MjData
    render_data = mujoco.MjData(model)

    # Obstacles (optional)
    obs = None
    if show_obstacles and ("obstacles" in data_npz.files):
        obs = np.asarray(data_npz["obstacles"], dtype=np.float64)
        print(f"Obstacle overlay enabled: obstacles shape={obs.shape}")
    elif show_obstacles:
        print("Obstacle overlay enabled, but NPZ has no 'obstacles' array.")

    prev_quat: Optional[np.ndarray] = None
    T = int(X_sim.shape[0])
    i = max(0, min(int(start_index), T - 1))

    print("Close the viewer window to exit.")
    counter = 0
    with mujoco.viewer.launch_passive(model, render_data) as viewer:
        while viewer.is_running():
            # Reset per-frame overlay geoms
            clear_user_geoms(viewer)

            # --- Ground plane (same as teleport viewer) ---
            add_ground_plane(
                viewer,
                z=0.0,              # world Z height
                size_xy=50.0,       # half-extent in X/Y
                rgba=(0.7, 0.7, 0.7, 1.0),
            )

            # --- Obstacles (same as teleport viewer) ---
            if show_obstacles and ("obstacles" in data_npz.files):
                obs = np.asarray(data_npz["obstacles"], dtype=np.float64)
                for k in range(obs.shape[0]):
                    add_cylinder_pillar(
                        viewer,
                        pos_xyz=np.array([obs[k, 0], obs[k, 1], 0.0]),
                        radius=float(obs[k, 2]),
                        height=float(obstacle_height),
                    )

            # Render simulated state at i
            qpos, qvel = extract_qpos_qvel(
                X_sim[i], model=model, n_joints=n_joints, layout=layout
            )
            print(counter)
            counter+=1
            # Normalize / smooth base quaternion
            if qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if quat_sign_smoothing and (prev_quat is not None):
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            # Write into MuJoCo buffers
            nq_write = min(qpos.shape[0], render_data.qpos.shape[0])
            nv_write = min(qvel.shape[0], render_data.qvel.shape[0])
            render_data.qpos[:nq_write] = qpos[:nq_write]
            render_data.qvel[:nv_write] = qvel[:nv_write]

            mujoco.mj_forward(model, render_data)
            viewer.sync()

            # Advance
            i_next = i + int(stride)
            if i_next >= T:
                i_next = 0
                counter =0 
                prev_quat = None
            i = i_next

            if realtime:
                time.sleep(float(dt) * float(stride))



def parse_args():
    p = argparse.ArgumentParser(description="Render H1 rollout by re-simulating with config.dynamics(...) then viewing.")
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
        help=(
            "How to read qpos/qvel out of each simulated x[t]. "
            "'prefix' assumes x[:nq+nv]=[qpos,qvel] with nq=7+n_joints. "
            "'offset' uses --offset. "
            "'model_dims_prefix' uses model.nq/nv from the prefix."
        ),
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
    return p.parse_args()


def main():
    # Keep your typical x64 setting for dynamics fidelity
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
    )


if __name__ == "__main__":
    main()
