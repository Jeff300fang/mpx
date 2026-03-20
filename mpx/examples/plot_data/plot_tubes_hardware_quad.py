#!/usr/bin/env python3
import os
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mpx.utils.fast_sls_visual import get_trajectory_tubes
from matplotlib.patches import Rectangle, Circle

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
folder = "/home/jeff/logs/XRTI2"
folder_phi = "/home/jeff/logs/Phi_xRTI2"

num_files = 1200
output_mp4 = "xrti_box_tubes.mp4"   # saved in current folder
frames_dir = "frames_xrti"
ffmpeg_bin = "ffmpeg"
obstacle_center = np.array([1.2, -0.5])
obstacle_radius = 0.6
fps = 20
dpi = 160

# -----------------------------------------------------------------------------
# Load all data first
# -----------------------------------------------------------------------------
all_plans = []
all_upper = []
all_lower = []

for i in range(num_files):
    x_file = os.path.join(folder, f"X_{i:05d}.npy")
    phi_file = os.path.join(folder_phi, f"Phi_x_{i:05d}.npy")

    if not os.path.exists(x_file):
        print(f"Missing X file: {x_file}")
        continue
    if not os.path.exists(phi_file):
        print(f"Missing Phi_x file: {phi_file}")
        continue

    X = np.load(x_file)
    Phi_x = np.load(phi_file)

    tube_width = np.asarray(get_trajectory_tubes(Phi_x))   # expected shape ~ (T, nx)
    plans_xy = np.asarray(X[:, :2])
    upper_xy = plans_xy + tube_width[:, :2]
    lower_xy = plans_xy - tube_width[:, :2]

    all_plans.append(plans_xy)
    all_upper.append(upper_xy)
    all_lower.append(lower_xy)

    print(f"Loaded {i:05d}: {plans_xy.shape[0]} points")

N = len(all_plans)
if N == 0:
    raise RuntimeError("No valid X / Phi_x pairs were loaded.")

T = all_plans[0].shape[0]

# -----------------------------------------------------------------------------
# Figure setup
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

ax.add_patch(
    Circle(
        obstacle_center,
        obstacle_radius,
        facecolor="tab:red",
        edgecolor="tab:red",
        alpha=0.35,
        zorder=1,
    )
)

legend_handles = [
    Line2D([0], [0], color="tab:orange", linewidth=2.5, linestyle="-",
           label="Executed prefix"),
    Line2D([0], [0], color="tab:orange", linewidth=2.5, linestyle="--",
           label="Planned"),
    Line2D([0], [0], marker="o", color="tab:blue", linestyle="None",
           markersize=6, label="Current"),
    Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.18,
          label="Robust tube"),
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2,
    frameon=True,
    framealpha=1.0,
    fancybox=False,
    fontsize=12,
    handlelength=2.2,
    columnspacing=1.6,
    labelspacing=0.9,
)

# Global bounds so axes do not jump
all_pts = np.vstack(
    [pts for triple in zip(all_plans, all_upper, all_lower) for pts in triple]
)
xmin, ymin = all_pts.min(axis=0)
xmax, ymax = all_pts.max(axis=0)
pad = 0.5

ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_aspect("auto")
ax.set_xlabel(r"$p_x$")
ax.set_ylabel(r"$p_y$")

# Executed path: show first point of each plan up to current frame
executed_xy_full = np.array([p[0] for p in all_plans])

(actual_line,) = ax.plot([], [], "-", color="tab:orange", linewidth=2.5)
(plan_line,) = ax.plot([], [], "--", color="tab:orange", linewidth=2.5)
(curr_pt,) = ax.plot([], [], "o", color="tab:blue", markersize=6)

# Pre-create tube rectangles
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
# Animation logic
# -----------------------------------------------------------------------------
def init_artists():
    actual_line.set_data([], [])
    plan_line.set_data([], [])
    curr_pt.set_data([], [])
    for r in tube_boxes:
        r.set_visible(False)

def update_frame(i: int):
    # Executed prefix = first state from each file up to i
    actual_xy = executed_xy_full[:i + 1]
    actual_line.set_data(actual_xy[:, 0], actual_xy[:, 1])
    curr_pt.set_data([actual_xy[-1, 0]], [actual_xy[-1, 1]])

    # Current planned trajectory
    plans_xy = all_plans[i]
    upper_xy = all_upper[i]
    lower_xy = all_lower[i]

    plan_line.set_data(plans_xy[:, 0], plans_xy[:, 1])

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
# Export MP4
# -----------------------------------------------------------------------------
def export_mp4():
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg not found on PATH (looked for '{ffmpeg_bin}').")

    os.makedirs(frames_dir, exist_ok=True)
    frames_dir_abs = os.path.abspath(frames_dir)
    out_mp4_abs = os.path.abspath(output_mp4)

    init_artists()
    for i in range(N):
        update_frame(i)
        fig.canvas.draw()
        frame_path = os.path.join(frames_dir_abs, f"frame_{i:05d}.png")
        fig.savefig(frame_path, dpi=dpi)
        print(f"[render] {frame_path}")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir_abs, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4_abs,
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Saved {out_mp4_abs}")

def plot_obstacle_distance(all_plans, all_upper, all_lower,
                           obstacle_center,
                           obstacle_radius,
                           save_path="distance_vs_tube.pdf"):
    N = len(all_plans)

    dist = np.zeros(N)
    for i in range(N):
        p = all_plans[i][0]
        dist[i] = np.linalg.norm(p - obstacle_center)

    tube_upper = np.zeros(N)
    tube_lower = np.zeros(N)
    for i in range(N):
        up = all_upper[i][-1]
        lo = all_lower[i][-1]
        width = np.linalg.norm(up - lo) / 2.0
        tube_upper[i] = dist[i] + width
        tube_lower[i] = dist[i] - width

    dt = 0.02
    t = np.arange(N) * dt

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(t, dist, linewidth=2.5, label="Distance to obstacle")

    ax.fill_between(
        t,
        tube_lower,
        tube_upper,
        alpha=0.25,
        label="Tube bound"
    )

    ax.axhline(
        obstacle_radius,
        linestyle="--",
        linewidth=2,
        color="tab:blue",
        label=f"Obstacle radius ({obstacle_radius:.1f} m)"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    ax.margins(x=0.0, y=0.1)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=True,
        fontsize=16,
        handlelength=2.0,
        columnspacing=1.4
    )

    fig.subplots_adjust(bottom=0.30)
    fig.savefig(save_path, dpi=200)
    print("Saved:", save_path)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # export_mp4()
    plot_obstacle_distance(
        all_plans,
        all_upper,
        all_lower,
        obstacle_center,
        obstacle_radius
    )