"""
render_first_state_all_X.py

Load every X_*.npy saved by RunQuadConstrained (your node), take the FIRST timestep
state (X[0]) from each file, and render them sequentially in a MuJoCo viewer by
"teleporting" (setting qpos/qvel + mj_forward).

NOW INCLUDES: Ghost trail timelapse across files.
- Keeps a history of past qpos/qvel (last --ghost-len files)
- During each --hold window, renders multiple passes with alpha blending

Usage:
  python render_first_state_all_X.py \
      --x-dir /home/jeff/logs/X3 \
      --pattern "X_*.npy" \
      --hold 0.25 \
      --ghost-len 25

If you saved dict objects (allow_pickle=True):
  python render_first_state_all_X.py --x-dir ... --allow-pickle --key X
"""

from __future__ import annotations

import os
import time
import glob
import argparse
import tempfile
from collections import deque
from typing import Any, Iterable

import numpy as np
import mujoco
import mujoco.viewer

# Must match the rollout-producing config (same as your viewer)
import mpx.config.config_go2 as config


# -----------------------------
# Helpers (same style as your viewer)
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


def load_X_from_npy(npy_path: str, *, key: str | None, allow_pickle: bool) -> np.ndarray:
    assert os.path.exists(npy_path), f"Missing NPY: {npy_path}"
    arr = np.load(npy_path, allow_pickle=bool(allow_pickle))

    # Numeric ndarray case
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return np.asarray(arr)

    # Object case: dict, etc.
    if key is None:
        raise RuntimeError(
            f"{npy_path}: loaded dtype=object. Provide --key (e.g. --key X) and --allow-pickle."
        )

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        obj = arr.item() if arr.shape == () else arr
    else:
        obj = arr

    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"{npy_path}: key '{key}' not in dict. Keys: {list(obj.keys())}")
        return np.asarray(obj[key])

    raise RuntimeError(f"{npy_path}: unsupported object type {type(obj)}; expected numeric ndarray or dict.")


def inject_ground_plane(xml: str, *, ground_z: float = 0.0, half_size_xy: float = 50.0) -> str:
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


def inject_obstacles_as_cylinders(
    xml: str,
    obstacles: np.ndarray,
    *,
    height: float = 1.0,
    rgba=(1.0, 0.0, 0.0, 1.0),
) -> str:
    """
    Inject vertical cylinder geoms for obstacles.
    obstacles: array of shape (N, 3) with [x, y, radius]
    """
    if obstacles.size == 0:
        return xml

    geoms = []
    for i, (x, y, r) in enumerate(np.asarray(obstacles)):
        geoms.append(f"""
        <geom name="obstacle_{i}"
              type="cylinder"
              pos="{x} {y} {height/2}"
              size="{r} {height/2}"
              rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
              contype="0"
              conaffinity="0"/>
        """)

    block = "\n".join(geoms)

    if "<worldbody>" not in xml:
        raise RuntimeError("MJCF has no <worldbody> tag")

    return xml.replace("<worldbody>", "<worldbody>\n" + block, 1)


def load_model_with_ground(model_path: str, *, ground_z: float, half_size_xy: float) -> mujoco.MjModel:
    assert os.path.exists(model_path), f"Missing model XML: {model_path}"
    with open(model_path, "r") as f:
        xml = f.read()

    obstacles = np.array([
        [1.2, -0.3, 0.3],
        [2.7,  0.75, 0.3],
    ])
    xml = inject_ground_plane(xml, ground_z=ground_z, half_size_xy=half_size_xy)
    xml = inject_obstacles_as_cylinders(
        xml,
        obstacles=obstacles,
        height=1.5,
        rgba=(1.0, 0.2, 0.2, 0.5),
    )

    model_dir = os.path.dirname(os.path.abspath(model_path))
    fd, tmp_xml_path = tempfile.mkstemp(prefix="go2_with_ground_", suffix=".xml", dir=model_dir)
    os.close(fd)
    with open(tmp_xml_path, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(tmp_xml_path)
    return model


def _swap_lr_leg_blocks_12(x12: np.ndarray) -> np.ndarray:
    """
    Swap 12 joint entries arranged as 4 legs*3 joints:
      [FR, FL, BR, BL] -> [FL, FR, BL, BR]
    """
    x12 = np.asarray(x12)
    if x12.shape[0] != 12:
        raise ValueError(f"Expected 12 joint entries, got {x12.shape[0]}")
    out = np.empty_like(x12)
    out[0:3] = x12[3:6]
    out[3:6] = x12[0:3]
    out[6:9] = x12[9:12]
    out[9:12] = x12[6:9]
    return out


def extract_qpos_qvel(
    X_row: np.ndarray,
    n_joints: int,
    model: mujoco.MjModel,
    *,
    use_model_dims: bool = False,
    swap_lr: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if use_model_dims:
        nq = int(model.nq)
        nv = int(model.nv)
    else:
        nq = 7 + int(n_joints)
        nv = 6 + int(n_joints)

    X_row = np.asarray(X_row)
    if X_row.shape[0] < nq + nv:
        raise ValueError(f"X row too short: need >= {nq+nv}, got {X_row.shape[0]}")

    qpos = np.array(X_row[:nq], dtype=np.float64, copy=True)
    qvel = np.array(X_row[nq:nq + nv], dtype=np.float64, copy=True)

    if swap_lr and int(n_joints) == 12 and qpos.shape[0] >= 19 and qvel.shape[0] >= 18:
        qpos[7:19] = _swap_lr_leg_blocks_12(qpos[7:19])
        qvel[6:18] = _swap_lr_leg_blocks_12(qvel[6:18])

    return qpos, qvel


# -----------------------------
# Ghost rendering
# -----------------------------
def _geom_name(model: mujoco.MjModel, gid: int) -> str:
    # Returns '' if unnamed
    try:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    except Exception:
        return ""


def _build_geom_mask_for_robot_only(model: mujoco.MjModel) -> np.ndarray:
    """
    Exclude ground + injected obstacles (by name).
    Everything else is considered "robot" for ghosting purposes.
    """
    mask = np.ones((model.ngeom,), dtype=bool)
    for g in range(model.ngeom):
        name = _geom_name(model, g)
        if name == "ground" or name.startswith("obstacle_"):
            mask[g] = False
    return mask


def _alpha_schedule(i: int, n: int, a_min: float, a_max: float, power: float = 2.0) -> float:
    """
    i in [0..n-1], older -> smaller alpha, newer -> larger alpha
    """
    if n <= 1:
        return float(a_max)
    t = float(i) / float(n - 1)
    t = t ** float(power)
    return float(a_min + (a_max - a_min) * t)


def render_ghost_trail(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ghost_qpos_list: Iterable[np.ndarray],
    ghost_qvel_list: Iterable[np.ndarray] | None,
    *,
    orig_rgba: np.ndarray,
    robot_geom_mask: np.ndarray | None,
    alpha_min: float,
    alpha_max: float,
    alpha_power: float,
):
    """
    Multi-pass render: set state to each ghost, adjust alpha, viewer.sync().
    Finally restore original colors (orig_rgba must be copied by caller).
    """
    # Save current state
    qpos_now = data.qpos.copy()
    qvel_now = data.qvel.copy()

    ghosts_qpos = list(ghost_qpos_list)
    ghosts_qvel = list(ghost_qvel_list) if ghost_qvel_list is not None else None

    n = len(ghosts_qpos)
    if n == 0:
        # nothing to do
        return

    # Oldest -> newest
    for i in range(n):
        data.qpos[:] = ghosts_qpos[i]
        if ghosts_qvel is not None:
            data.qvel[:] = ghosts_qvel[i]
        else:
            data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        alpha = _alpha_schedule(i, n, alpha_min, alpha_max, power=alpha_power)

        # Set alpha only on masked geoms if provided
        model.geom_rgba[:] = orig_rgba
        if robot_geom_mask is None:
            model.geom_rgba[:, 3] = alpha
        else:
            model.geom_rgba[robot_geom_mask, 3] = alpha

        viewer.sync()

    # Restore current pose opaque (and restore original RGBA)
    data.qpos[:] = qpos_now
    data.qvel[:] = qvel_now
    mujoco.mj_forward(model, data)
    model.geom_rgba[:] = orig_rgba
    viewer.sync()


# -----------------------------
# Main
# -----------------------------
def run_sequence(
    x_dir: str,
    pattern: str,
    *,
    model_path: str,
    n_joints: int,
    hold_s: float,
    loop: bool,
    use_model_dims: bool,
    quat_sign_smoothing: bool,
    allow_pickle: bool,
    key: str | None,
    swap_lr: bool,
    ground_z: float,
    ground_half_size: float,
    max_files: int | None,
    # ghost params
    ghost_len: int,
    ghost_alpha_min: float,
    ghost_alpha_max: float,
    ghost_alpha_power: float,
    ghost_fps: float,
    ghost_robot_only: bool,
    render_every: int,
):
    x_dir = os.path.abspath(x_dir)
    files = sorted(glob.glob(os.path.join(x_dir, pattern)))
    if not files:
        raise RuntimeError(f"No files matched: dir={x_dir}, pattern={pattern}")

    if max_files is not None:
        files = files[: max(0, int(max_files))]

    model = load_model_with_ground(model_path, ground_z=ground_z, half_size_xy=ground_half_size)
    data = mujoco.MjData(model)

    print(f"X dir: {x_dir}")
    print(f"Matched {len(files)} files (pattern={pattern})")
    print(f"Model dims: nq={model.nq}, nv={model.nv}")
    print(f"Using dims from {'model' if use_model_dims else 'convention (7+n_joints, 6+n_joints)'}")
    print(f"hold_s={hold_s}, loop={loop}, swap_lr={swap_lr}")
    print(f"ghost_len={ghost_len}, alpha=[{ghost_alpha_min},{ghost_alpha_max}], power={ghost_alpha_power}, fps={ghost_fps}")
    print("Close the viewer to exit.\n")

    prev_quat = None

    def set_overhead_camera(viewer, center=(0.0, 0.0, 0.0), height=6.0):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array(center, dtype=np.float64)
        viewer.cam.distance = height
        viewer.cam.azimuth = 0.0
        viewer.cam.elevation = -90.0

    # Ghost buffers
    ghost_qpos = deque(maxlen=max(0, int(ghost_len)))
    ghost_qvel = deque(maxlen=max(0, int(ghost_len)))

    # Cache original geom colors once (and we’ll restore each pass)
    orig_rgba = model.geom_rgba.copy()

    # Precompute which geoms to fade if ghost_robot_only
    robot_geom_mask = _build_geom_mask_for_robot_only(model) if ghost_robot_only else None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        idx = 0
        model.vis.headlight.active = 0
        set_overhead_camera(
            viewer,
            center=(0.0, 0.0, 0.3),
            height=6.0,
        )

        while viewer.is_running():
            if idx >= len(files):
                if loop:
                    idx = 0
                    prev_quat = None
                    ghost_qpos.clear()
                    ghost_qvel.clear()
                else:
                    break

            path = files[idx]
            if (idx % render_every) != 0:
                idx += 1
                continue
            try:
                X = load_X_from_npy(path, key=key, allow_pickle=allow_pickle)
                if X.ndim != 2 or X.shape[0] < 1:
                    raise ValueError(f"X must be 2D with T>=1. Got shape {X.shape}.")
                qpos, qvel = extract_qpos_qvel(
                    X[0],
                    n_joints=n_joints,
                    model=model,
                    use_model_dims=use_model_dims,
                    swap_lr=swap_lr,
                )
            except Exception as e:
                print(f"[SKIP] {os.path.basename(path)}: {e}")
                idx += 1
                continue

            # Normalize quaternion + sign smoothing
            if qpos.shape[0] >= 7:
                q = normalize_quat(qpos[3:7])
                if quat_sign_smoothing and (prev_quat is not None):
                    q = align_quat_sign(q, prev_quat)
                qpos[3:7] = q
                prev_quat = q.copy()

            # Teleport current pose
            nq_write = min(qpos.shape[0], data.qpos.shape[0])
            nv_write = min(qvel.shape[0], data.qvel.shape[0])
            data.qpos[:nq_write] = qpos[:nq_write]
            data.qvel[:nv_write] = qvel[:nv_write]

            mujoco.mj_forward(model, data)

            # Update ghost history AFTER setting current pose:
            # We want the ghost trail to include the current pose as the newest entry.
            if ghost_len > 0:
                ghost_qpos.append(data.qpos.copy())
                ghost_qvel.append(data.qvel.copy())

            base = qpos[:3] if qpos.shape[0] >= 3 else np.array([np.nan, np.nan, np.nan])
            print(f"[{idx+1:04d}/{len(files):04d}] {os.path.basename(path)}  base={base}")

            # Hold window: render ghost trail at ghost_fps
            dt_frame = 1.0 / max(1.0, float(ghost_fps))
            t_end = time.time() + float(hold_s)
            while viewer.is_running() and time.time() < t_end:
                if ghost_len > 0 and len(ghost_qpos) > 0:
                    render_ghost_trail(
                        viewer,
                        model,
                        data,
                        ghost_qpos_list=list(ghost_qpos),     # oldest -> newest (deque order)
                        ghost_qvel_list=list(ghost_qvel),
                        orig_rgba=orig_rgba,
                        robot_geom_mask=robot_geom_mask,
                        alpha_min=float(ghost_alpha_min),
                        alpha_max=float(ghost_alpha_max),
                        alpha_power=float(ghost_alpha_power),
                    )
                else:
                    # Fallback: just render current pose
                    model.geom_rgba[:] = orig_rgba
                    viewer.sync()

                # keep UI responsive but don't spam GPU
                time.sleep(dt_frame)

            idx += 1


def parse_args():
    p = argparse.ArgumentParser(
        description="Render the first state (X[0]) from every saved X_*.npy in a directory, sequentially, in MuJoCo."
    )
    p.add_argument("--x-dir", type=str, required=True, help="Directory containing X_*.npy files (e.g., /home/jeff/logs/X3).")
    p.add_argument("--pattern", type=str, default="X_*.npy", help="Glob pattern within x-dir (default: X_*.npy).")

    p.add_argument("--hold", type=float, default=0.25, help="Seconds to hold each file's first state before advancing.")
    p.add_argument("--loop", action="store_true", help="Loop back to the first file after finishing.")
    p.add_argument("--max-files", type=int, default=None, help="Optional cap on number of files to render.")

    p.add_argument("--allow-pickle", action="store_true", help="Allow loading pickled objects from .npy (dict, etc).")
    p.add_argument("--key", type=str, default=None, help="If loading a dict/object .npy, extract X via this key (e.g. X).")

    p.add_argument("--use-model-dims", action="store_true", help="Use model.nq/model.nv to slice X rows.")
    p.add_argument("--swap-lr", action="store_true", help="Apply [FR,FL,BR,BL]->[FL,FR,BL,BR] leg-block swap to joints/vels.")
    p.add_argument("--no-quat-smoothing", dest="quat_smoothing", action="store_false", help="Disable quaternion sign smoothing.")
    p.set_defaults(quat_smoothing=True)

    p.add_argument("--ground-z", type=float, default=0.0, help="Z height of injected ground plane.")
    p.add_argument("--ground-half-size", type=float, default=50.0, help="Half-size (X/Y) of injected ground plane.")

    # Ghost trail args
    p.add_argument("--ghost-len", type=int, default=25, help="Number of ghost poses to show (0 disables).")
    p.add_argument("--ghost-alpha-min", type=float, default=0.03, help="Alpha of oldest ghost.")
    p.add_argument("--ghost-alpha-max", type=float, default=0.35, help="Alpha of newest ghost.")
    p.add_argument("--ghost-alpha-power", type=float, default=2.0, help="Fade curve power (>1 fades older ghosts faster).")
    p.add_argument("--ghost-fps", type=float, default=60.0, help="Render rate during hold window.")
    p.add_argument("--ghost-robot-only", action="store_true", help="Only fade robot geoms (exclude ground/obstacles by name).")

    p.add_argument("--render-every", type=int, default=1,
               help="Only render every N files (frames).")

    return p.parse_args()


def main():
    args = parse_args()
    run_sequence(
        x_dir=args.x_dir,
        pattern=args.pattern,
        model_path=config.model_path,
        n_joints=int(config.n_joints),
        hold_s=float(args.hold),
        loop=bool(args.loop),
        use_model_dims=bool(args.use_model_dims),
        quat_sign_smoothing=bool(args.quat_smoothing),
        allow_pickle=bool(args.allow_pickle),
        key=(str(args.key) if args.key is not None else None),
        swap_lr=bool(args.swap_lr),
        ground_z=float(args.ground_z),
        ground_half_size=float(args.ground_half_size),
        max_files=(int(args.max_files) if args.max_files is not None else None),
        ghost_len=int(args.ghost_len),
        ghost_alpha_min=float(args.ghost_alpha_min),
        ghost_alpha_max=float(args.ghost_alpha_max),
        ghost_alpha_power=float(args.ghost_alpha_power),
        ghost_fps=float(args.ghost_fps),
        ghost_robot_only=bool(args.ghost_robot_only),
        render_every=int(args.render_every)
    )


if __name__ == "__main__":
    main()
