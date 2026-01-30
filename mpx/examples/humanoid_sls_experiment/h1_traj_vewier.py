"""
teleport_viewer_h1_rollout.py

Interactive MuJoCo viewer for a saved rollout NPZ produced by your H1 MPC script
(mpc_data/h1_mpc_rollout.npz). The robot is "teleported" each frame to the stored
state rather than stepped via dynamics.

Differences vs the Go2 viewer:
  - Loads config from mpx.config.config_h1
  - Default NPZ path matches your save_path: mpc_data/h1_mpc_rollout.npz
  - Uses config.model_path by default (your MJCF)
  - Robustly supports two common X layouts:
      (A) X starts with [qpos, qvel] (MuJoCo floating base order)
      (B) X is your full MPC state, but contains qpos/qvel as the *first* (nq+nv) entries
          (this is what we assume unless you pass --layout explicit)

Run:
  python teleport_viewer_h1_rollout.py
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
    Defines how to extract [qpos, qvel] from one X row.

    mode:
      - "prefix": X_row[:(nq+nv)] is [qpos,qvel]
      - "offset": X_row[offset:offset+(nq+nv)] is [qpos,qvel]
      - "model_dims_prefix": use model.nq/model.nv, slice from prefix
    """
    mode: str = "prefix"
    offset: int = 0


def extract_qpos_qvel(
    X_row: np.ndarray,
    model: mujoco.MjModel,
    n_joints: int,
    layout: LayoutSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    X_row = np.asarray(X_row)

    if layout.mode == "model_dims_prefix":
        nq = int(model.nq)
        nv = int(model.nv)
        start = 0
    else:
        # Your previous convention for floating-base + joints
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

        if layout.mode == "prefix":
            start = 0
        elif layout.mode == "offset":
            start = int(layout.offset)
        else:
            raise ValueError(f"Unknown layout.mode={layout.mode}")

    need = start + nq + nv
    if X_row.shape[0] < need:
        raise ValueError(
            f"X row too small for requested layout: need at least {need} entries "
            f"(start={start}, nq={nq}, nv={nv}), got {X_row.shape[0]}."
        )

    qpos = np.array(X_row[start : start + nq], dtype=np.float64, copy=True)
    qvel = np.array(X_row[start + nq : start + nq + nv], dtype=np.float64, copy=True)
    return qpos, qvel


# -----------------------------
# Optional obstacle overlay (cylinder pillars)
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
        size=np.array([float(radius - 0.3), float(height) / 2.0, 0.0], dtype=np.float64),
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
    rgba=(0.6, 0.6, 0.6, 1.0),
) -> None:
    """
    Inject a visual-only ground plane.

    MuJoCo plane geom:
      - size = [half_x, half_y, 0]
      - pos  = center of plane
    """
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
# Main viewer loop
# -----------------------------
def run_viewer(
    npz_path: str,
    model_path: str,
    *,
    n_joints: int,
    dt: float,
    realtime: bool,
    start_index: int,
    stride: int,
    quat_sign_smoothing: bool,
    layout: LayoutSpec,
    show_obstacles: bool,
    obstacle_height: float,
) -> None:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model XML: {model_path}")

    data_npz = np.load(npz_path)
    if "X" not in data_npz.files:
        raise RuntimeError(f"NPZ missing 'X'. Found arrays: {list(data_npz.files)}")

    X = np.asarray(data_npz["X"])
    T = int(X.shape[0])

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Basic info
    print(f"Loaded NPZ: {npz_path}")
    print(f"Arrays: {list(data_npz.files)}")
    print(f"X shape: {X.shape} (T={T})")
    print(f"Model: {model_path}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}")
    print(f"config.n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"layout={layout}")
    if show_obstacles:
        has_obs = "obstacles" in data_npz.files
        print(f"obstacles overlay: enabled (npz_has_obstacles={has_obs})")
    print("Close the viewer window to exit.")

    prev_quat: Optional[np.ndarray] = None
    i = max(0, min(int(start_index), T - 1))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Optional obstacles overlay:
            # - If NPZ has 'obstacles', prefer that
            # - Otherwise, fall back to your hardcoded obstacle from the script (if desired)
            clear_user_geoms(viewer)

            # --- Ground plane ---
            add_ground_plane(
                viewer,
                z=0.0,              # world Z height
                size_xy=50.0,       # half-extent in X/Y
                rgba=(0.7, 0.7, 0.7, 1.0),
            )

            # --- Obstacles ---
            if show_obstacles and "obstacles" in data_npz.files:
                obs = np.asarray(data_npz["obstacles"], dtype=np.float64)
                for k in range(obs.shape[0]):
                    add_cylinder_pillar(
                        viewer,
                        pos_xyz=np.array([obs[k, 0], obs[k, 1], 0.0]),
                        radius=float(obs[k, 2]),
                        height=float(obstacle_height),
                    )

            # Teleport to state i
            qpos, qvel = extract_qpos_qvel(X[i], model=model, n_joints=n_joints, layout=layout)

            # Normalize / smooth base quaternion at qpos[3:7] for floating base
            if qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if quat_sign_smoothing and (prev_quat is not None):
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            # Write into MuJoCo buffers (write what we have; remainder untouched)
            nq_write = min(qpos.shape[0], data.qpos.shape[0])
            nv_write = min(qvel.shape[0], data.qvel.shape[0])
            data.qpos[:nq_write] = qpos[:nq_write]
            data.qvel[:nv_write] = qvel[:nv_write]

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Advance
            i_next = i + int(stride)
            if i_next >= T:
                i_next = 0
                prev_quat = None
            i = i_next

            if realtime:
                time.sleep(float(dt) * float(stride))


def parse_args():
    p = argparse.ArgumentParser(
        description="Teleport-render a saved H1 MPC rollout in an interactive MuJoCo viewer."
    )

    p.add_argument(
        "--npz",
        type=str,
        default=os.path.join("mpc_data", "h1_mpc_rollout.npz"),
        help="Path to rollout NPZ (must contain array 'X').",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(getattr(config, "model_path", "")),
        help="Path to MuJoCo XML. Defaults to config.model_path.",
    )

    p.add_argument("--realtime", action="store_true", help="Sleep dt each frame (scaled by stride).")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="Render as fast as possible.")
    p.set_defaults(realtime=True)

    p.add_argument("--start", type=int, default=0, help="Start index into X.")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (1=every state).")

    # Layout controls
    p.add_argument(
        "--layout",
        type=str,
        default="prefix",
        choices=["prefix", "offset", "model_dims_prefix"],
        help=(
            "How to read qpos/qvel out of each X[t]. "
            "'prefix' assumes X[t][:nq+nv] = [qpos,qvel]. "
            "'offset' uses --offset. "
            "'model_dims_prefix' uses model.nq/nv from the prefix."
        ),
    )
    p.add_argument("--offset", type=int, default=0, help="Offset into X[t] if --layout=offset.")

    p.add_argument(
        "--no-quat-smoothing",
        dest="quat_smoothing",
        action="store_false",
        help="Disable quaternion sign smoothing (may cause visual popping).",
    )
    p.set_defaults(quat_smoothing=True)

    # Obstacles overlay
    p.add_argument(
        "--show-obstacles",
        action="store_true",
        help="If NPZ contains 'obstacles' (n_obs,3), render them as cylinder pillars.",
    )
    p.add_argument(
        "--obstacle-height",
        type=float,
        default=3.0,
        help="Height (meters) for obstacle pillar visualization.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    layout = LayoutSpec(mode=args.layout, offset=int(args.offset))

    # Pull defaults from config (matches your MPC script)
    n_joints = int(getattr(config, "n_joints", 0))
    dt = float(getattr(config, "dt", 0.02))

    run_viewer(
        npz_path=str(args.npz),
        model_path=str(args.model),
        n_joints=n_joints,
        dt=dt,
        realtime=bool(args.realtime),
        start_index=int(args.start),
        stride=int(args.stride),
        quat_sign_smoothing=bool(args.quat_smoothing),
        layout=layout,
        show_obstacles=bool(args.show_obstacles),
        obstacle_height=float(args.obstacle_height),
    )


if __name__ == "__main__":
    main()
