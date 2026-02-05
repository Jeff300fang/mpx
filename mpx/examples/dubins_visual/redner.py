"""
render_saved_dubins_mpc_replay_deepreach_style.py

DeepReach-style rendering for a SINGLE executed trajectory `xs` (closed-loop),
plus (optionally) per-step plans and tube boxes saved from your MPC experiment.

Outputs:
  1) xs_tubes_centers.png   (static)
  2) replay.mp4             (animated)

Expected NPZ keys (any subset works):
  Required:
    - xs  (T,3)  OR  X (T,3)          executed trajectory [px, py, theta]
    - dt  scalar

  Obstacles (either form):
    - obstacles (K,3) with [cx,cy,r]
      OR
    - centers (K,2) and radii (K,)

  Optional for tubes/plans (per MPC step t):
    - plans_xy  (n_steps, N+1, 2)
    - lowers_xy (n_steps, N+1, 2)
    - uppers_xy (n_steps, N+1, 2)

Run:
  python render_saved_dubins_mpc_replay_deepreach_style.py \
      --npz dubins_mpc_rollout.npz \
      --sprite racecar_topdown.png \
      --out_dir rendered \
      --tube_stride 2 --box_stride 2
"""

from __future__ import annotations

import argparse
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import animation
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D


# =============================================================================
# DeepReach-style RC params (NO system latex dependency)
# =============================================================================
mpl.rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,
})
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =============================================================================
# DeepReach palette / style constants
# =============================================================================
DEEPREACH_ORANGE = "#ff7f0e"
OBSTACLE_FACE = "red"
OBSTACLE_EDGE = "darkred"
TUBE_FACE = "tab:blue"

EXEC_LINEWIDTH = 3.0
EXEC_ALPHA_PNG = 0.45
EXEC_ALPHA_MP4 = 0.40

PLAN_LW = 2.5
PLAN_ALPHA = 0.90

OBSTACLE_ALPHA = 0.25
OBSTACLE_LW = 1.5

TUBE_ALPHA = 0.10
GRID_ALPHA = 0.30

GOAL_TOL = 0.05  # for "freeze at goal" behavior (optional)


# =============================================================================
# Helpers: annotations / decorations
# =============================================================================
def draw_labeled_point(ax, xy, label, *, color="black", marker="o",
                       text_dx=0.03, text_dy=0.03, fontsize=16,
                       ha="left", va="bottom"):
    ax.scatter(xy[0], xy[1], s=70, marker=marker, color=color, zorder=8)
    ax.text(
        xy[0] + text_dx, xy[1] + text_dy, label,
        fontsize=fontsize, ha=ha, va=va, zorder=9
    )


def draw_obstacles(ax, centers: np.ndarray, radii: np.ndarray):
    for c, r in zip(centers, radii):
        circ = plt.Circle(
            (float(c[0]), float(c[1])),
            float(r),
            facecolor=OBSTACLE_FACE,
            edgecolor=OBSTACLE_EDGE,
            alpha=OBSTACLE_ALPHA,
            linewidth=OBSTACLE_LW,
            zorder=5,
        )
        ax.add_patch(circ)


# =============================================================================
# Tube helpers
# =============================================================================
def make_tube_patch_collection(
    lowers_xy_step: np.ndarray,   # (N+1,2)
    uppers_xy_step: np.ndarray,   # (N+1,2)
    tube_stride: int = 1,
    alpha: float = TUBE_ALPHA,
    facecolor: str = TUBE_FACE,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    zorder: int = 2,
) -> PatchCollection:
    lo = np.asarray(lowers_xy_step)
    up = np.asarray(uppers_xy_step)

    stride = max(1, int(tube_stride))
    rects: list[Rectangle] = []
    for k in range(0, lo.shape[0], stride):
        w = up[k, 0] - lo[k, 0]
        h = up[k, 1] - lo[k, 1]
        if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
            continue
        rects.append(Rectangle((float(lo[k, 0]), float(lo[k, 1])), float(w), float(h)))

    return PatchCollection(
        rects,
        match_original=False,
        alpha=alpha,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )


# =============================================================================
# Sprite helpers (imshow + Affine2D)
# =============================================================================
def load_sprite_rgba(path: str) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    elif img.shape[-1] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], dtype=img.dtype)])
    return img


def make_sprite_artist(
    ax: plt.Axes,
    img_rgba: np.ndarray,
    x: float, y: float, theta: float,
    length: float,
    width: float,
    alpha: float,
    zorder: int = 6,
    interpolation: str = "nearest",
):
    extent = (-length / 2.0, length / 2.0, -width / 2.0, width / 2.0)
    im = ax.imshow(
        img_rgba,
        extent=extent,
        origin="upper",
        interpolation=interpolation,
        alpha=alpha,
        zorder=zorder,
    )
    tf = Affine2D().rotate(theta).translate(x, y)
    im.set_transform(tf + ax.transData)
    return {"im": im, "tf": tf}


def update_sprite_artist(sprite, x: float, y: float, theta: float) -> None:
    tf: Affine2D = sprite["tf"]
    tf.clear()
    tf.rotate(theta)
    tf.translate(x, y)


# =============================================================================
# Data loading utilities
# =============================================================================
def load_npz_data(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)

    # executed trajectory: prefer 'xs', else 'X'
    if "xs" in d:
        xs = np.asarray(d["xs"])
    elif "X" in d:
        xs = np.asarray(d["X"])
    else:
        raise KeyError("NPZ must contain 'xs' (T,3) or 'X' (T,3).")

    if xs.ndim != 2 or xs.shape[1] != 3:
        raise ValueError(f"executed xs/X must be (T,3). Got {xs.shape}")

    # obstacles: prefer 'obstacles', else 'centers'+'radii', else empty
    if "obstacles" in d:
        obstacles = np.asarray(d["obstacles"])
        if obstacles.size == 0:
            centers = np.zeros((0, 2), dtype=float)
            radii = np.zeros((0,), dtype=float)
        else:
            if obstacles.ndim != 2 or obstacles.shape[1] != 3:
                raise ValueError(f"'obstacles' must be (K,3). Got {obstacles.shape}")
            centers = obstacles[:, :2]
            radii = obstacles[:, 2]
    else:
        centers = np.asarray(d["centers"]) if "centers" in d else np.zeros((0, 2), dtype=float)
        radii = np.asarray(d["radii"]) if "radii" in d else np.zeros((0,), dtype=float)

    # optional per-step plans/tubes
    plans_xy = np.asarray(d["plans_xy"]) if "plans_xy" in d else None
    lowers_xy = np.asarray(d["lowers_xy"]) if "lowers_xy" in d else None
    uppers_xy = np.asarray(d["uppers_xy"]) if "uppers_xy" in d else None

    # dt required
    if "dt" not in d:
        raise KeyError("NPZ must contain 'dt' (scalar).")
    dt = float(np.asarray(d["dt"]))

    return xs, centers, radii, plans_xy, lowers_xy, uppers_xy, dt


# =============================================================================
# Static plot: xs + (optional) tubes + obstacles
# =============================================================================
def plot_xs_tubes_centers(
    xs: np.ndarray,                 # (T,3)
    centers: np.ndarray,            # (K,2)
    radii: np.ndarray,              # (K,)
    lowers_xy_step: np.ndarray | None,  # (N+1,2)
    uppers_xy_step: np.ndarray | None,  # (N+1,2)
    tube_stride: int,
    margin: float,
    filename: str,
    dpi: int = 300,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    start_label: str = "Start",
    goal_label: str = "Goal",
    goal_xy: tuple[float, float] | None = None,
):
    xs = np.asarray(xs)
    px = xs[:, 0]
    py = xs[:, 1]

    # axis limits
    if xlim is None or ylim is None:
        all_x = [px.ravel()]
        all_y = [py.ravel()]
        if lowers_xy_step is not None and uppers_xy_step is not None:
            all_x.append(np.asarray(lowers_xy_step)[:, 0].ravel())
            all_x.append(np.asarray(uppers_xy_step)[:, 0].ravel())
            all_y.append(np.asarray(lowers_xy_step)[:, 1].ravel())
            all_y.append(np.asarray(uppers_xy_step)[:, 1].ravel())
        if centers.size:
            all_x.append(centers[:, 0].ravel())
            all_y.append(centers[:, 1].ravel())
        all_x = np.concatenate(all_x) if len(all_x) else px
        all_y = np.concatenate(all_y) if len(all_y) else py
        xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
        ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    # Start / Goal markers
    start_xy = (float(px[0]), float(py[0]))
    draw_labeled_point(ax, start_xy, start_label, marker="o", text_dx=0.03, text_dy=0.04)

    if goal_xy is None:
        goal_xy = (float(px[-1]), float(py[-1]))
    draw_labeled_point(ax, goal_xy, goal_label, marker="*", text_dx=-0.06, text_dy=0.05)

    # Obstacles
    if centers.size and radii.size == centers.shape[0]:
        draw_obstacles(ax, centers, radii)

    # Tubes
    if lowers_xy_step is not None and uppers_xy_step is not None:
        tube_pc = make_tube_patch_collection(
            lowers_xy_step, uppers_xy_step,
            tube_stride=tube_stride,
            alpha=TUBE_ALPHA,
            facecolor=TUBE_FACE,
            zorder=2,
        )
        ax.add_collection(tube_pc)

    # Executed path
    ax.plot(px, py, color=DEEPREACH_ORANGE, linewidth=EXEC_LINEWIDTH, alpha=EXEC_ALPHA_PNG, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$", fontsize=20)
    ax.set_ylabel(r"$p_y$", fontsize=20)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Legend
    legend_handles = [
        Patch(facecolor=OBSTACLE_FACE, edgecolor=OBSTACLE_EDGE, alpha=OBSTACLE_ALPHA, label="Obstacle"),
        Line2D([], [], color=DEEPREACH_ORANGE, linewidth=EXEC_LINEWIDTH, alpha=0.8, label="Executed\ntrajectory"),
    ]
    if lowers_xy_step is not None and uppers_xy_step is not None:
        legend_handles.insert(1, Patch(facecolor=TUBE_FACE, edgecolor="none", alpha=TUBE_ALPHA, label="Tubes"))

    ax.legend(handles=legend_handles, loc="lower right", fontsize=16, framealpha=0.9)

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MP4: animated xs + (optional) per-step plan + per-step tube boxes + obstacles + sprite
# =============================================================================
def save_xs_replay_mp4(
    xs: np.ndarray,                     # (T,3)
    centers: np.ndarray,                # (K,2)
    radii: np.ndarray,                  # (K,)
    plans_xy: np.ndarray | None,        # (n_steps,N+1,2)
    lowers_xy: np.ndarray | None,       # (n_steps,N+1,2)
    uppers_xy: np.ndarray | None,       # (n_steps,N+1,2)
    sprite_rgba: np.ndarray,
    filename: str,
    dt: float,
    fps: int | None,
    box_stride: int,
    tube_stride_static: int,
    margin: float,
    sprite_length: float,
    sprite_width: float,
    sprite_alpha: float,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
):
    xs = np.asarray(xs)
    T = xs.shape[0]

    # If we have plans/tubes, animation frames = n_steps; else frames = T
    have_plans = (plans_xy is not None)
    have_tubes = (lowers_xy is not None and uppers_xy is not None)

    if have_plans:
        plans_xy = np.asarray(plans_xy)
        n_steps = plans_xy.shape[0]
    else:
        n_steps = T

    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))

    # axis limits
    all_x = [xs[:, 0].ravel()]
    all_y = [xs[:, 1].ravel()]
    if have_plans:
        all_x.append(plans_xy[:, :, 0].ravel())
        all_y.append(plans_xy[:, :, 1].ravel())
    if have_tubes:
        all_x.append(np.asarray(lowers_xy)[:, :, 0].ravel())
        all_x.append(np.asarray(uppers_xy)[:, :, 0].ravel())
        all_y.append(np.asarray(lowers_xy)[:, :, 1].ravel())
        all_y.append(np.asarray(uppers_xy)[:, :, 1].ravel())
    if centers.size:
        all_x.append(centers[:, 0].ravel())
        all_y.append(centers[:, 1].ravel())

    all_x = np.concatenate(all_x) if len(all_x) else xs[:, 0]
    all_y = np.concatenate(all_y) if len(all_y) else xs[:, 1]

    if xlim is None or ylim is None:
        xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
        ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$p_x$", fontsize=20)
    ax.set_ylabel(r"$p_y$", fontsize=20)
    ax.grid(True, alpha=GRID_ALPHA)

    # Obstacles
    if centers.size and radii.size == centers.shape[0]:
        draw_obstacles(ax, centers, radii)

    # Start / Goal labels
    start_xy = (float(xs[0, 0]), float(xs[0, 1]))
    draw_labeled_point(ax, start_xy, "Start", marker="o", text_dx=0.03, text_dy=0.04)

    if goal_xy is None:
        goal_xy = (float(xs[-1, 0]), float(xs[-1, 1]))
    draw_labeled_point(ax, goal_xy, "Goal", marker="*", text_dx=-0.06, text_dy=0.05)

    goal_xy_np = np.array(goal_xy, dtype=float)
    goal_tol2 = float(GOAL_TOL) ** 2

    # Optional: static tube overlay (useful if you don’t have per-step tubes)
    tube_pc_static = None
    if (not have_tubes) and have_plans:
        # if user saved only a single plan/tube step elsewhere, ignore; keep clean
        pass
    elif (not have_tubes) and (have_plans is False):
        pass

    # Artists
    (exec_line,) = ax.plot([], [], lw=EXEC_LINEWIDTH, alpha=EXEC_ALPHA_MP4, color=DEEPREACH_ORANGE, zorder=3)
    (plan_line,) = ax.plot([], [], lw=PLAN_LW, ls="--", alpha=PLAN_ALPHA, color=DEEPREACH_ORANGE, zorder=4)
    plan_line.set_visible(bool(have_plans))

    tube_boxes = PatchCollection([], alpha=TUBE_ALPHA, match_original=False, zorder=2)
    ax.add_collection(tube_boxes)
    tube_boxes.set_visible(bool(have_tubes))

    # Sprite (car)
    x0, y0, th0 = float(xs[0, 0]), float(xs[0, 1]), float(xs[0, 2])
    car = make_sprite_artist(
        ax, sprite_rgba,
        x0, y0, th0,
        length=sprite_length,
        width=sprite_width,
        alpha=sprite_alpha,
        zorder=6,
        interpolation="nearest",
    )

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    reached = False
    freeze_pose = np.array([x0, y0, th0], dtype=float)

    def init():
        exec_line.set_data([], [])
        if have_plans:
            plan_line.set_data([], [])
        if have_tubes:
            tube_boxes.set_paths([])
        step_text.set_text("")
        return (exec_line, plan_line, tube_boxes, car["im"], step_text)

    def update(t: int):
        nonlocal reached, freeze_pose

        # executed trail: show xs up to index t (clamped)
        xt = min(int(t), T - 1)
        exec_line.set_data(xs[:xt + 1, 0], xs[:xt + 1, 1])

        # per-step plan/tubes: use step t (clamped)
        if have_plans:
            st = min(int(t), n_steps - 1)
            pl = plans_xy[st]
            plan_line.set_data(pl[:, 0], pl[:, 1])

        if have_tubes:
            st = min(int(t), lowers_xy.shape[0] - 1)
            lo = lowers_xy[st]
            up = uppers_xy[st]

            rects = []
            stride = max(int(box_stride), 1)
            for k in range(0, lo.shape[0], stride):
                w = up[k, 0] - lo[k, 0]
                h = up[k, 1] - lo[k, 1]
                if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                    continue
                rects.append(Rectangle((float(lo[k, 0]), float(lo[k, 1])), float(w), float(h)))
            tube_boxes.set_paths(rects)

        # sprite pose (freeze if reaches goal tolerance)
        x_t = float(xs[xt, 0])
        y_t = float(xs[xt, 1])
        th_t = float(xs[xt, 2])

        if not reached:
            dx = x_t - goal_xy_np[0]
            dy = y_t - goal_xy_np[1]
            if (dx * dx + dy * dy) <= goal_tol2:
                reached = True
                freeze_pose = np.array([x_t, y_t, th_t], dtype=float)

        if reached:
            update_sprite_artist(car, float(freeze_pose[0]), float(freeze_pose[1]), float(freeze_pose[2]))
        else:
            update_sprite_artist(car, x_t, y_t, th_t)

        step_text.set_text(f"t = {t}/{n_steps-1}")

        artists = [exec_line, car["im"], step_text]
        if have_plans:
            artists.append(plan_line)
        if have_tubes:
            artists.append(tube_boxes)
        if tube_pc_static is not None:
            artists.append(tube_pc_static)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        blit=True,
        interval=interval_ms,
    )

    if not animation.FFMpegWriter.isAvailable():
        plt.close(fig)
        raise RuntimeError("ffmpeg not available. Install ffmpeg (or switch to GIF with PillowWriter).")

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer, dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--sprite", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="rendered")

    # tube / plan options
    ap.add_argument("--tube_step", type=int, default=0, help="Which MPC step to show in the static PNG (if tubes exist).")
    ap.add_argument("--tube_stride", type=int, default=2, help="Stride for rectangles in static PNG.")
    ap.add_argument("--box_stride", type=int, default=2, help="Stride for rectangles per frame in MP4.")

    # rendering options
    ap.add_argument("--no_mp4", action="store_true")
    ap.add_argument("--sprite_alpha", type=float, default=0.8)
    ap.add_argument("--sprite_length", type=float, default=0.18)
    ap.add_argument("--sprite_width", type=float, default=0.09)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)

    # goal marker override (optional)
    ap.add_argument("--goal_xy", type=float, nargs=2, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    xs, centers, radii, plans_xy, lowers_xy, uppers_xy, dt = load_npz_data(args.npz)

    # choose a tube step for static png if available
    lo_step = None
    up_step = None
    if lowers_xy is not None and uppers_xy is not None:
        tube_step = int(max(0, min(args.tube_step, lowers_xy.shape[0] - 1)))
        lo_step = lowers_xy[tube_step]
        up_step = uppers_xy[tube_step]

    sprite_rgba = load_sprite_rgba(args.sprite)

    goal_xy = tuple(args.goal_xy) if args.goal_xy is not None else None

    # PNG
    png_path = os.path.join(args.out_dir, "xs_tubes_centers.png")
    plot_xs_tubes_centers(
        xs=xs,
        centers=centers,
        radii=radii,
        lowers_xy_step=lo_step,
        uppers_xy_step=up_step,
        tube_stride=args.tube_stride,
        margin=args.margin,
        filename=png_path,
        dpi=300,
        xlim=tuple(args.xlim) if args.xlim is not None else None,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        goal_xy=goal_xy,
    )
    print(f"[Saved] {png_path}")

    # MP4
    if not args.no_mp4:
        mp4_path = os.path.join(args.out_dir, "replay.mp4")
        save_xs_replay_mp4(
            xs=xs,
            centers=centers,
            radii=radii,
            plans_xy=plans_xy,
            lowers_xy=lowers_xy,
            uppers_xy=uppers_xy,
            sprite_rgba=sprite_rgba,
            filename=mp4_path,
            dt=dt,
            fps=args.fps,
            box_stride=args.box_stride,
            tube_stride_static=args.tube_stride,
            margin=args.margin,
            sprite_length=args.sprite_length,
            sprite_width=args.sprite_width,
            sprite_alpha=args.sprite_alpha,
            xlim=tuple(args.xlim) if args.xlim is not None else None,
            ylim=tuple(args.ylim) if args.ylim is not None else None,
            goal_xy=goal_xy,
        )
        print(f"[Saved] {mp4_path}")


if __name__ == "__main__":
    main()
