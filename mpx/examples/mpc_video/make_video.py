#!/usr/bin/env python3
import os
import shutil
import subprocess

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpx.utils.fast_sls_visual import get_trajectory_tubes

# -----------------------------------------------------------------------------
# Matplotlib style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,
})
plt.rcParams.update({
    "font.size": 20,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FILE_NAME = "h1_mpc_rollout_log.npz"

# If True: interactive playback respects wall_time (variable delays)
PLAY_REALTIME_INTERACTIVE = True

# If True: export a VFR MP4 that respects wall_time using ffmpeg concat durations
EXPORT_VFR_MP4 = True
VFR_MP4_PATH = "h1_mpc_box_tubes_revealed_realtime.mp4"

FRAMES_DIR = "frames_h1_realtime"
FFMPEG = "ffmpeg"  # or absolute path to ffmpeg if needed

# Clamp step delays to avoid 0ms frames or huge pauses
DT_MIN = 1e-3   # seconds
DT_MAX = 0.5    # seconds

# Obstacles (x, y, r)
obstacles = jnp.array([
    [2.0,  0.2,  0.43],
    [1.7, -1.1,  0.43],
])

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
data = np.load(FILE_NAME)

X_list = data["X"]          # (N, T, nx)
Phi_x_list = data["Phi_x"]  # (N, T, nx, nx) or similar
N, T, nx = X_list.shape

# Executed trajectory: reveal progressively
actual_xy_full = X_list[:, 0, :2]  # (N, 2)

# wall_time support
wall_time = None
if "wall_time" in data.files:
    wall_time = data["wall_time"].astype(np.float64)
    wall_time = wall_time - wall_time[0]  # normalize to start at 0

# -----------------------------------------------------------------------------
# Figure setup
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

legend_handles = [
    Line2D([0], [0], color="tab:orange", linewidth=2.5, linestyle="-",
           label="Executed"),
    Line2D([0], [0], color="tab:orange", linewidth=2.5, linestyle="--",
           label="Planned"),
    Line2D([0], [0], marker="o", color="tab:blue", linestyle="None",
           markersize=6, label="Current"),
    Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.18,
          label="Robust tube"),
    Patch(facecolor="tab:red", edgecolor="tab:red", alpha=0.35,
          label="Obstacle"),
]



ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.32),
    ncol=3,                        # 2 rows
    frameon=True,                 # <-- border on
    framealpha=1.0,               # solid white background
    fancybox=False,               # square corners (IEEE-style)
    fontsize=12,
    handlelength=2.2,
    columnspacing=1.6,
    labelspacing=0.9,
)
# Obstacles
for (ox, oy, r) in np.array(obstacles):
    ax.add_patch(Circle((ox, oy), r, facecolor="tab:red", edgecolor="tab:red", alpha=0.35))

# Actual trajectory (solid)
(actual_line,) = ax.plot([], [], "-", color="tab:orange", linewidth=2.5)

# Planned trajectory (dashed)
(plan_line,) = ax.plot([], [], "--", color="tab:orange", linewidth=2.5)

# Current executed point
(curr_pt,) = ax.plot([], [], "o", color="tab:blue", markersize=6)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$p_x$")
ax.set_ylabel(r"$p_y$")

# Fix bounds from all data so axes don't jump
all_xy = X_list[:, :, :2].reshape(-1, 2)
xmin, ymin = all_xy.min(axis=0)
xmax, ymax = all_xy.max(axis=0)
pad = 0.5
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# Tube boxes: pre-create T rectangles
tube_boxes = []
for _ in range(T):
    rect = Rectangle(
        (0.0, 0.0), 1e-6, 1e-6,
        facecolor="tab:blue", edgecolor="tab:blue",
        alpha=0.18, linewidth=1.0
    )
    rect.set_visible(False)
    ax.add_patch(rect)
    tube_boxes.append(rect)

# -----------------------------------------------------------------------------
# Animation update logic
# -----------------------------------------------------------------------------
def init_artists():
    actual_line.set_data([], [])
    plan_line.set_data([], [])
    curr_pt.set_data([], [])
    for r in tube_boxes:
        r.set_visible(False)

def update_frame(i: int):
    # Reveal executed/actual only up to i
    actual_xy = actual_xy_full[: i + 1]
    actual_line.set_data(actual_xy[:, 0], actual_xy[:, 1])
    curr_pt.set_data([actual_xy[-1, 0]], [actual_xy[-1, 1]])

    # Current plan and tubes
    X = X_list[i]          # (T, nx)
    plan_xy = X[:, :2]     # (T, 2)
    plan_line.set_data(plan_xy[:, 0], plan_xy[:, 1])

    tubes = np.asarray(get_trajectory_tubes(Phi_x_list[i]))  # (T, nx)
    tubes_xy = tubes[:, :2]

    lower_xy = plan_xy - tubes_xy
    upper_xy = plan_xy + tubes_xy

    for t in range(T):
        x0, y0 = lower_xy[t]
        x1, y1 = upper_xy[t]
        w = max(0.0, float(x1 - x0))
        h = max(0.0, float(y1 - y0))
        tube_boxes[t].set_xy((float(x0), float(y0)))
        tube_boxes[t].set_width(w)
        tube_boxes[t].set_height(h)
        tube_boxes[t].set_visible(True)

# -----------------------------------------------------------------------------
# Real-time interactive playback (variable delays)
# -----------------------------------------------------------------------------
def run_realtime_interactive():
    if wall_time is None:
        raise RuntimeError(
            "wall_time not found in NPZ. Re-run your logger with wall_time saved, "
            "or set PLAY_REALTIME_INTERACTIVE=False."
        )

    dt = np.diff(wall_time)
    dt = np.clip(dt, DT_MIN, DT_MAX)
    dt_ms = (dt * 1000.0).astype(int)

    init_artists()
    update_frame(0)
    fig.canvas.draw_idle()

    frame_idx = 0
    timer_obj = fig.canvas.new_timer(interval=int(dt_ms[0]) if N > 1 else 50)

    def on_timer():
        nonlocal frame_idx, timer_obj
        frame_idx += 1
        if frame_idx >= N:
            timer_obj.stop()
            return

        update_frame(frame_idx)
        fig.canvas.draw_idle()

        if frame_idx < N - 1:
            timer_obj.interval = int(dt_ms[frame_idx])
        else:
            timer_obj.interval = 50

    timer_obj.add_callback(on_timer)
    timer_obj.start()
    plt.show()

# -----------------------------------------------------------------------------
# Export a variable-timing MP4 (VFR) using ffmpeg concat durations
# -----------------------------------------------------------------------------
def export_vfr_mp4():
    if wall_time is None:
        raise RuntimeError("wall_time not found in NPZ; cannot export real-time VFR MP4.")

    if shutil.which(FFMPEG) is None:
        raise RuntimeError(f"ffmpeg not found on PATH (looked for '{FFMPEG}').")

    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Use absolute paths everywhere to avoid cwd / relative path issues.
    frames_dir_abs = os.path.abspath(FRAMES_DIR)
    out_mp4_abs = os.path.abspath(VFR_MP4_PATH)

    dt = np.diff(wall_time)
    dt = np.clip(dt, DT_MIN, DT_MAX)

    # Render PNG frames
    init_artists()
    for i in range(N):
        update_frame(i)
        fig.canvas.draw()
        frame_path = os.path.join(frames_dir_abs, f"frame_{i:05d}.png")
        fig.savefig(frame_path, dpi=160)
        print(f"[render] {frame_path}")

    # Build concat file with absolute file paths.
    # Note: concat demuxer interprets paths relative to the concat file,
    # so absolute paths is the simplest robust approach.
    concat_path = os.path.join(frames_dir_abs, "concat.txt")
    with open(concat_path, "w") as f:
        for i in range(N - 1):
            frame_path = os.path.join(frames_dir_abs, f"frame_{i:05d}.png")
            f.write(f"file '{frame_path}'\n")
            f.write(f"duration {float(dt[i]):.9f}\n")
        last_frame = os.path.join(frames_dir_abs, f"frame_{N-1:05d}.png")
        f.write(f"file '{last_frame}'\n")
        f.write(f"file '{last_frame}'\n")  # ensures last frame displays

    # Encode MP4 with variable frame timing.
    # -fps_mode vfr replaces deprecated -vsync vfr.
    cmd = [
        FFMPEG,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_path,
        "-fps_mode", "vfr",
        "-pix_fmt", "yuv420p",
        out_mp4_abs,
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Saved {out_mp4_abs}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if EXPORT_VFR_MP4:
        export_vfr_mp4()

    if PLAY_REALTIME_INTERACTIVE:
        run_realtime_interactive()
    else:
        init_artists()
        for i in range(N):
            update_frame(i)
            plt.pause(0.05)
        plt.show()
