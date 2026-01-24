"""
teleport_viewer_go2_rollout_npy.py

Interactive MuJoCo viewer for a saved rollout NPY (X array), where the robot is
"teleported" each frame to the stored state rather than being stepped via dynamics.

Key idea:
  - Set data.qpos / data.qvel directly from X[i]
  - Call mujoco.mj_forward(model, data)
  - viewer.sync()

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
        # Often a 0-d object array wrapping dict
        if arr.shape == ():
            obj = arr.item()
        else:
            # Less common; might already be list/dict-like
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

    X_row = np.asarray(X_row)
    if X_row.shape[0] < nq + nv:
        raise ValueError(
            f"X has insufficient dimension for nq+nv. Need at least {nq+nv}, got {X_row.shape[0]}."
        )

    qpos = np.array(X_row[:nq], dtype=np.float64, copy=True)
    qvel = np.array(X_row[nq:nq + nv], dtype=np.float64, copy=True)
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
):
    assert os.path.exists(npy_path), f"Missing NPY: {npy_path}"
    assert os.path.exists(model_path), f"Missing model XML: {model_path}"

    X = load_X_from_npy(npy_path, key=key, allow_pickle=allow_pickle)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (T, nx). Got shape {X.shape} (ndim={X.ndim}).")

    T = int(X.shape[0])

    # model = mujoco.MjModel.from_xml_path(model_path)
    # Create a derived XML that includes a ground plane
    model_path_with_ground = inject_ground_plane_xml(
        model_path,
        plane_size=(50.0, 50.0, 0.1),
        plane_z=0.0,
        rgba=(0.7, 0.7, 0.7, 1.0),
        friction=(1.0, 0.005, 0.0001),
        add_material=True,
    )

    model = mujoco.MjModel.from_xml_path(model_path_with_ground)

    data = mujoco.MjData(model)

    # Basic info
    print(f"Loaded NPY: {npy_path}")
    print(f"X shape: {X.shape}")
    print(f"Model: {model_path}")
    print(f"Configured n_joints={n_joints}, dt={dt}, stride={stride}, realtime={realtime}")
    print(f"Model dims: model.nq={model.nq}, model.nv={model.nv}")
    print(f"Using dims from {'model' if use_model_dims else 'convention (7+n_joints, 6+n_joints)'}")
    print("Close the viewer window to exit.")

    prev_quat = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = int(start_index)
        i = max(0, min(i, T - 1))

        while viewer.is_running():
            qpos, qvel = extract_qpos_qvel(
                X[i], n_joints=n_joints, model=model, use_model_dims=use_model_dims
            )

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

            # Advance index
            print("Current:", i)
            i_next = i + int(stride)
            if i_next >= X.shape[0]:
                i_next = 0
                prev_quat = None
            #     prev_quat = None
            # if i_next >= T:
            #     i_next = 0
            #     prev_quat = None
            i = i_next

            if realtime:
                # time.sleep(float(dt) * float(stride))
                time.sleep(0.1)


import tempfile
from pathlib import Path

def inject_ground_plane_xml(
    base_xml_path: str,
    *,
    plane_size: tuple[float, float, float] = (50.0, 50.0, 0.1),
    plane_z: float = 0.0,
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
    add_material: bool = True,
) -> str:
    """
    Create a temporary MuJoCo XML that is the same as base_xml_path but with a ground plane
    inserted into <worldbody>. Returns the path to the generated XML.

    Notes:
      - This is string-based injection; it assumes the XML contains a <worldbody> ... </worldbody>.
      - If your XML already has a ground plane, you may want to detect and skip insertion.
    """
    base_xml_path = str(base_xml_path)
    assert os.path.exists(base_xml_path), f"Missing model XML: {base_xml_path}"

    xml = Path(base_xml_path).read_text(encoding="utf-8")

    if "<worldbody" not in xml:
        raise RuntimeError("Could not find <worldbody> in XML; cannot inject plane safely.")

    # Basic "already has plane" heuristic (optional)
    if 'type="plane"' in xml or "type='plane'" in xml:
        # If you want to always add another plane, remove this guard.
        return base_xml_path

    material_block = ""
    material_name = "injected_ground_mat"
    if add_material:
        # Ensure there is an <asset> block; if not, create one under <mujoco>.
        if "<asset" in xml:
            # Insert material just before </asset>
            material_block = f'\n    <material name="{material_name}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>\n'
            xml = xml.replace("</asset>", material_block + "  </asset>", 1)
        else:
            # Create an <asset> block right after <mujoco ...>
            # This is a simple heuristic: insert after the first '>' of <mujoco ...>
            idx = xml.find(">")
            if idx == -1:
                raise RuntimeError("Malformed XML: cannot find end of <mujoco ...> tag.")
            asset_block = (
                "\n  <asset>\n"
                f'    <material name="{material_name}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>\n'
                "  </asset>\n"
            )
            xml = xml[: idx + 1] + asset_block + xml[idx + 1 :]

    # Construct plane geom. For MuJoCo plane geoms, `size` is half-extent in x/y; z is unused but required.
    plane_geom = (
        "\n    <!-- Injected ground plane -->\n"
        f'    <geom name="injected_ground" type="plane" pos="0 0 {plane_z}" '
        f'size="{plane_size[0]} {plane_size[1]} {plane_size[2]}" '
        f'friction="{friction[0]} {friction[1]} {friction[2]}" '
        + (f'material="{material_name}" ' if add_material else f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" ')
        + '/>\n'
    )

    # Insert plane geom right after the opening <worldbody ...> tag.
    # Find "<worldbody" then the next ">" and inject after it.
    wb_start = xml.find("<worldbody")
    wb_tag_end = xml.find(">", wb_start)
    if wb_tag_end == -1:
        raise RuntimeError("Malformed XML: <worldbody> tag not closed with '>'.")

    xml = xml[: wb_tag_end + 1] + plane_geom + xml[wb_tag_end + 1 :]

    # Write to a temp file
    base_dir = Path(base_xml_path).resolve().parent
    out_path = base_dir / (Path(base_xml_path).stem + "_with_ground.xml")
    out_path.write_text(xml, encoding="utf-8")
    return str(out_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Teleport-render a saved rollout X from a .npy file in an interactive MuJoCo viewer."
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
        help="Disable quaternion sign smoothing (may cause visual popping).",
    )
    p.set_defaults(quat_smoothing=True)

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
    )


if __name__ == "__main__":
    main()
