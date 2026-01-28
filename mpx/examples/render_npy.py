"""
teleport_viewer_go2_rollout_npy.py

Interactive MuJoCo viewer for a saved rollout NPY (X array), where the robot is
"teleported" each frame to the stored state rather than being stepped via dynamics.

Key idea:
  - Set data.qpos / data.qvel directly from X[i]
  - Call mujoco.mj_forward(model, data)
  - viewer.sync()

Adds:
  - Injected ground plane into MJCF before loading (so you always see a floor)

Supports two NPY formats:
  1) Plain ndarray saved via np.save("X.npy", X) -> loads directly as X
  2) Pickled object saved via np.save("rollout.npy", {"X": X, ...}, allow_pickle=True)
     -> use --key X (or another key) to extract

Run:
  python teleport_viewer_go2_rollout_npy.py --npy path/to/X.npy
"""

from __future__ import annotations

import os
import time
import argparse
import tempfile
from typing import Any

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


def load_X_from_npy(npy_path: str, *, key: str | None, allow_pickle: bool) -> np.ndarray:
    """
    Load X from a .npy file.

    - If the file contains a numeric ndarray: returns it directly.
    - If the file contains an object array (e.g., dict saved with allow_pickle=True),
      extracts X via `key`.
    """
    assert os.path.exists(npy_path), f"Missing NPY: {npy_path}"

    arr = np.load(npy_path, allow_pickle=bool(allow_pickle))

    # Typical case: arr is numeric ndarray with ndim >= 2
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return np.asarray(arr)

    # Object case: either a 0-d object array containing dict, or object array itself.
    if key is None:
        raise RuntimeError(
            "Loaded an object from .npy (dtype=object). Provide --key to extract X, e.g. --key X"
        )

    obj: Any
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == ():
            obj = arr.item()
        else:
            obj = arr
    else:
        obj = arr

    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"Key '{key}' not found in dict. Available keys: {list(obj.keys())}")
        X = obj[key]
        return np.asarray(X)

    raise RuntimeError(
        f"Unsupported object type loaded from .npy: {type(obj)}. "
        "Expected numeric ndarray or dict-like object."
    )


def inject_ground_plane(xml: str, *, ground_z: float = 0.0, half_size_xy: float = 50.0) -> str:
    """
    Insert a plane geom into the first <worldbody> block.

    Notes:
      - We do NOT need contacts for visualization, but it doesn't hurt.
      - rgba controls visual color.
      - size for plane: (half-length-x, half-length-y, unused-thickness)
    """
    ground_geom = f"""
    <geom name="ground" type="plane"
          pos="0 0 {ground_z}"
          size="{half_size_xy} {half_size_xy} 0.1"
          rgba="0.8 0.8 0.8 1"
          contype="1" conaffinity="1"
          friction="1 0.005 0.0001"/>
    """
    if "<worldbody>" not in xml:
        raise RuntimeError("MJCF has no <worldbody> tag; cannot inject ground plane safely.")
    return xml.replace("<worldbody>", "<worldbody>\n" + ground_geom, 1)


def load_model_with_ground(model_path: str, *, ground_z: float, half_size_xy: float) -> mujoco.MjModel:
    """
    Load MJCF from `model_path`, inject a ground plane, write to a temp XML placed in the
    same directory as the original model (so MuJoCo can resolve relative assets), and load it.
    """
    assert os.path.exists(model_path), f"Missing model XML: {model_path}"

    with open(model_path, "r") as f:
        xml = f.read()

    xml = inject_ground_plane(xml, ground_z=ground_z, half_size_xy=half_size_xy)

    model_dir = os.path.dirname(os.path.abspath(model_path))
    fd, tmp_xml_path = tempfile.mkstemp(prefix="go2_with_ground_", suffix=".xml", dir=model_dir)
    os.close(fd)
    with open(tmp_xml_path, "w") as f:
        f.write(xml)

    # Load by PATH so asset relative paths resolve correctly.
    model = mujoco.MjModel.from_xml_path(tmp_xml_path)

    # If you want to keep the temp XML for debugging, comment out the unlink.
    # Leaving it in place is harmless too; it's in the model directory.
    # os.unlink(tmp_xml_path)

    return model


def _swap_lr_leg_blocks_12(x12: np.ndarray) -> np.ndarray:
    """
    Permute 12 joint entries arranged as 4 legs * 3 joints.

    Your statement:
      saved order:   FR, FL, BR, BL
      model expects: FL, FR, BL, BR

    If each leg has 3 joints, indices are blocks:
      0:3   -> leg0
      3:6   -> leg1
      6:9   -> leg2
      9:12  -> leg3

    Mapping: [FR, FL, BR, BL] -> [FL, FR, BL, BR]
      new block0 (FL) = old block1 (FL)
      new block1 (FR) = old block0 (FR)
      new block2 (BL) = old block3 (BL)
      new block3 (BR) = old block2 (BR)

    i.e., swap blocks (0 <-> 1) and (2 <-> 3).
    """
    x12 = np.asarray(x12)
    if x12.shape[0] != 12:
        raise ValueError(f"Expected 12 joint entries, got {x12.shape[0]}")

    out = np.empty_like(x12)
    out[0:3] = x12[3:6]     # FL <- old FL block
    out[3:6] = x12[0:3]     # FR <- old FR block
    out[6:9] = x12[9:12]    # BL <- old BL block
    out[9:12] = x12[6:9]    # BR <- old BR block
    return out


def extract_qpos_qvel(
    X_row: np.ndarray,
    n_joints: int,
    model: mujoco.MjModel,
    *,
    use_model_dims: bool = False,
    swap_lr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract qpos, qvel from one row of X.

    Default behavior follows your earlier convention:
      nq = 7 + n_joints
      nv = 6 + n_joints

    If your MuJoCo model uses different nq/nv and your X matches model.nq/model.nv exactly,
    set use_model_dims=True.

    swap_lr:
      Applies a leg-block permutation to the joint entries (and joint velocities)
      to match your model joint ordering.
    """
    if use_model_dims:
        nq = int(model.nq)
        nv = int(model.nv)
    else:
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

    X_row = np.asarray(X_row)
    if X_row.shape[0] < nq + nv:
        raise ValueError(
            f"X has insufficient dimension for nq+nv. Need at least {nq+nv}, got {X_row.shape[0]}."
        )

    qpos = np.array(X_row[:nq], dtype=np.float64, copy=True)
    qvel = np.array(X_row[nq:nq + nv], dtype=np.float64, copy=True)

    # Joint positions are qpos[7:7+n_joints], joint vels are qvel[6:6+n_joints].
    if swap_lr and int(n_joints) == 12 and qpos.shape[0] >= 19 and qvel.shape[0] >= 18:
        qpos[7:19] = _swap_lr_leg_blocks_12(qpos[7:19])
        qvel[6:18] = _swap_lr_leg_blocks_12(qvel[6:18])

    return qpos, qvel


# -----------------------------
# Main viewer
# -----------------------------
def run_viewer(
    npy_path: str,
    *,
    model_path: str,
    n_joints: int,
    dt: float,
    realtime: bool,
    start_index: int,
    stride: int,
    use_model_dims: bool,
    quat_sign_smoothing: bool,
    key: str | None,
    allow_pickle: bool,
    swap_lr: bool,
    ground_z: float,
    ground_half_size: float,
):
    assert os.path.exists(npy_path), f"Missing NPY: {npy_path}"
    assert os.path.exists(model_path), f"Missing model XML: {model_path}"

    X = load_X_from_npy(npy_path, key=key, allow_pickle=allow_pickle)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (T, nx). Got shape {X.shape} (ndim={X.ndim}).")

    T = int(X.shape[0])

    # Load model with injected ground plane
    model = load_model_with_ground(model_path, ground_z=ground_z, half_size_xy=ground_half_size)
    data = mujoco.MjData(model)

    # Basic info
    print(f"Loaded NPY: {npy_path}")
    print(f"X shape: {X.shape}")
    print(f"Model: {model_path} (ground injected)")
    print(f"Configured n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"Model dims: model.nq={model.nq}, model.nv={model.nv}")
    print(f"Using dims from {'model' if use_model_dims else 'convention (7+n_joints, 6+n_joints)'}")
    print(f"swap_lr={swap_lr}, ground_z={ground_z}, ground_half_size={ground_half_size}")
    print("Close the viewer window to exit.")

    prev_quat = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = int(start_index)
        i = max(0, min(i, T - 1))

        while viewer.is_running():
            qpos, qvel = extract_qpos_qvel(
                X[i],
                n_joints=n_joints,
                model=model,
                use_model_dims=use_model_dims,
                swap_lr=swap_lr,
            )
            print("Odom :",i, X[i, 0:3])
            # print("q1:", X[1, 7:19])
            # print("dq:", X[1, 25: 37])
            # Normalize base quaternion in qpos[3:7] if present
            if qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if quat_sign_smoothing and (prev_quat is not None):
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            nq_write = min(qpos.shape[0], data.qpos.shape[0])
            nv_write = min(qvel.shape[0], data.qvel.shape[0])

            data.qpos[:nq_write] = qpos[:nq_write]
            data.qvel[:nv_write] = qvel[:nv_write]

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Advance index (your current test loop)
            print("Current:", i)
            i_next = i + int(stride)
            if i_next >= X.shape[0]:
                i_next = 0
                prev_quat = None
            i = i_next

            if realtime:
                time.sleep(0.1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Teleport-render a saved rollout X from a .npy file in an interactive MuJoCo viewer (with ground plane)."
    )
    p.add_argument(
        "--npy",
        type=str,
        default=os.path.join("mpc_data", "go2_mpc_rollout_X.npy"),
        help="Path to .npy containing X, or a pickled dict containing X (use --key).",
    )
    p.add_argument(
        "--key",
        type=str,
        default=None,
        help="If the .npy contains a pickled dict/object, extract X using this key (e.g. X).",
    )
    p.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading pickled objects from .npy (required if you saved dict/object).",
    )

    p.add_argument("--realtime", action="store_true", help="Sleep dt each frame (scaled by stride).")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="Do not sleep; render as fast as possible.")
    p.set_defaults(realtime=True)

    p.add_argument("--start", type=int, default=0, help="Start index into X.")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (1 = every state, 2 = every other, ...).")

    p.add_argument(
        "--use-model-dims",
        action="store_true",
        help="Use model.nq/model.nv to slice X (requires X layout matches model exactly).",
    )

    p.add_argument(
        "--no-quat-smoothing",
        dest="quat_smoothing",
        action="store_false",
        help="Disable quaternion sign smoothing (may cause visual popping)."
    )
    p.set_defaults(quat_smoothing=True)

    p.add_argument(
        "--no-swap-lr",
        dest="swap_lr",
        action="store_false",
        help="Disable FR/FL and BR/BL leg-block swap (use raw joint order from X)."
    )
    p.set_defaults(swap_lr=False)

    # Ground plane options
    p.add_argument("--ground-z", type=float, default=0.0, help="Z height of the injected ground plane.")
    p.add_argument("--ground-half-size", type=float, default=50.0, help="Half-size (X/Y) of the injected ground plane.")

    return p.parse_args()


def main():
    args = parse_args()

    run_viewer(
        npy_path=args.npy,
        model_path=config.model_path,
        n_joints=int(config.n_joints),
        dt=float(config.dt),
        realtime=bool(args.realtime),
        start_index=int(args.start),
        stride=int(args.stride),
        use_model_dims=bool(args.use_model_dims),
        quat_sign_smoothing=bool(args.quat_smoothing),
        key=(str(args.key) if args.key is not None else None),
        allow_pickle=bool(args.allow_pickle),
        swap_lr=bool(args.swap_lr),
        ground_z=float(args.ground_z),
        ground_half_size=float(args.ground_half_size),
    )


if __name__ == "__main__":
    main()
