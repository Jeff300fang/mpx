"""
teleport_viewer_go2_rollout.py

Interactive MuJoCo viewer for a saved rollout NPZ (X, U, ...), where the robot is
"teleported" each frame to the stored state rather than being stepped via dynamics.

Key idea:
  - Set data.qpos / data.qvel directly from X[i]
  - Call mujoco.mj_forward(model, data)
  - viewer.sync()

This is ideal for visually debugging MPC rollouts.

Assumptions (matches your earlier script conventions):
  - X has shape (N+1, nx)
  - The first (nq+nv) entries of X correspond to [qpos, qvel] in MuJoCo floating-base order:
        qpos = [base_pos(3), base_quat(4), joint_pos(n_joints)]
        qvel = [base_lin_vel(3), base_ang_vel(3), joint_vel(n_joints)]
  - Uses model_path, n_joints, dt from mpx.config.config_go2

If your X layout differs, adjust `extract_qpos_qvel(...)`.

Run:
  python teleport_viewer_go2_rollout.py
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np

import mujoco
import mujoco.viewer

# Must match the rollout-producing config
import mpx.config.config_go2 as config


# -----------------------------
# Helpers
# -----------------------------
def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion q in-place-safe manner (returns new array)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q
    return q / n


def align_quat_sign(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """
    Prevent visual "popping" by ensuring quaternion sign continuity.
    If dot(q, q_ref) < 0, flip q.
    """
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    return -q if float(np.dot(q, q_ref)) < 0.0 else q


def extract_qpos_qvel(
    X_row: np.ndarray,
    n_joints: int,
    model: mujoco.MjModel,
    *,
    use_model_dims: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract qpos, qvel from one row of X.

    Default behavior follows your earlier convention:
      nq = 7 + n_joints
      nv = 6 + n_joints

    If your MuJoCo model uses different nq/nv and your X matches model.nq/model.nv exactly,
    set use_model_dims=True.
    """
    if use_model_dims:
        nq = int(model.nq)
        nv = int(model.nv)
    else:
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

    if X_row.shape[0] < nq + nv:
        raise ValueError(
            f"X has insufficient dimension for nq+nv. "
            f"Need at least {nq+nv}, got {X_row.shape[0]}."
        )

    qpos = np.array(X_row[:nq], dtype=np.float64, copy=True)
    qvel = np.array(X_row[nq:nq + nv], dtype=np.float64, copy=True)
    return qpos, qvel


# -----------------------------
# Main viewer
# -----------------------------
def run_viewer(
    npz_path: str,
    *,
    model_path: str,
    n_joints: int,
    dt: float,
    realtime: bool,
    start_index: int,
    stride: int,
    use_model_dims: bool,
    quat_sign_smoothing: bool,
):
    assert os.path.exists(npz_path), f"Missing NPZ: {npz_path}"
    assert os.path.exists(model_path), f"Missing model XML: {model_path}"

    data_npz = np.load(npz_path)
    if "X" not in data_npz.files:
        raise RuntimeError(f"NPZ missing 'X'. Found arrays: {data_npz.files}")

    X = data_npz["X"]
    T = X.shape[0]

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Basic info
    print(f"Loaded NPZ: {npz_path}")
    print(f"X shape: {X.shape}")
    print(f"Model: {model_path}")
    print(f"Configured n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"Model dims: model.nq={model.nq}, model.nv={model.nv}")
    print(f"Using dims from {'model' if use_model_dims else 'convention (7+n_joints, 6+n_joints)'}")
    print("Close the viewer window to exit.")

    # Precompute qpos/qvel indices once (for smoothing reference)
    prev_quat = None

    # Viewer loop (teleport)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = int(start_index)
        i = max(0, min(i, T - 1))

        while viewer.is_running():
            # Teleport to frame i
            qpos, qvel = extract_qpos_qvel(
                X[i], n_joints=n_joints, model=model, use_model_dims=use_model_dims
            )

            # Normalize base quaternion in qpos[3:7] if present
            # (MuJoCo floating base uses pos(3) then quat(4) at indices 3:7)
            if qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])

                if quat_sign_smoothing and (prev_quat is not None):
                    q = align_quat_sign(q, prev_quat)

                qpos[3:7] = q
                prev_quat = q.copy()

            # Assign into data
            # Note: if you use_model_dims=False, nq/nv may be smaller than model.nq/nv.
            # We write what we have; the rest remains as previously set. If your model
            # truly expects full qpos/qvel, use_model_dims=True and ensure X matches.
            nq_write = min(qpos.shape[0], data.qpos.shape[0])
            nv_write = min(qvel.shape[0], data.qvel.shape[0])

            data.qpos[:nq_write] = qpos[:nq_write]
            data.qvel[:nv_write] = qvel[:nv_write]

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Advance index
            i_next = i + int(stride)
            if i_next >= T:
                # Loop
                i_next = 0
                prev_quat = None
            i = i_next

            if realtime:
                time.sleep(float(dt) * float(stride))


def parse_args():
    p = argparse.ArgumentParser(description="Teleport-render a saved rollout X in an interactive MuJoCo viewer.")
    p.add_argument("--npz", type=str, default=os.path.join("mpc_data", "go2_mpc_rollout.npz"),
                   help="Path to rollout npz (must contain array 'X').")
    p.add_argument("--realtime", action="store_true", help="Sleep dt each frame (scaled by stride).")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="Do not sleep; render as fast as possible.")
    p.set_defaults(realtime=True)

    p.add_argument("--start", type=int, default=0, help="Start index into X.")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (1 = every state, 2 = every other, ...).")

    p.add_argument("--use-model-dims", action="store_true",
                   help="Use model.nq/model.nv to slice X (requires X layout matches model exactly).")

    p.add_argument("--no-quat-smoothing", dest="quat_smoothing", action="store_false",
                   help="Disable quaternion sign smoothing (may cause visual popping).")
    p.set_defaults(quat_smoothing=True)

    return p.parse_args()


def main():
    args = parse_args()

    run_viewer(
        npz_path=args.npz,
        model_path=config.model_path,
        n_joints=int(config.n_joints),
        dt=float(config.dt),
        realtime=bool(args.realtime),
        start_index=int(args.start),
        stride=int(args.stride),
        use_model_dims=bool(args.use_model_dims),
        quat_sign_smoothing=bool(args.quat_smoothing),
    )


if __name__ == "__main__":
    main()
