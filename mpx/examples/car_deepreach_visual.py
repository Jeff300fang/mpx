"""
render_saved_dubins_rollouts_racecar_deepreach_style.py
DeepReach-style rendering for:
  1) rollouts_tubes_centers.png
  2) rollouts.mp4

Run:
  python render_saved_dubins_rollouts_racecar_deepreach_style.py \
      --npz testing_rollouts/sls_vs_deepreach.npz \
      --sprite racecar_topdown.png \
      --out_dir rendered \
      --tube_step 0 --tube_stride 1
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
from matplotlib.ticker import MultipleLocator

GOAL_TOL = 0.1
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

ROLL_LINEWIDTH = 3.0
ROLL_ALPHA_PNG = 0.40
ROLL_ALPHA_MP4 = 0.35

OBSTACLE_ALPHA = 0.25
OBSTACLE_LW = 1.5

TUBE_ALPHA = 0.10

GRID_ALPHA = 0.30

# Start/Goal (DeepReach-like)
START_XY = (-0.75, -0.75)
GOAL_XY  = (1.0, 0.4)   # note: in your DeepReach plot you used (1.0, 0.4); adjust if desired

# =============================================================================
# Helpers: annotations / decorations
# =============================================================================
def draw_labeled_point(ax, xy, label, *, color="black", marker="o", text_dx=0.03, text_dy=0.03):
    ax.scatter(xy[0], xy[1], s=70, marker=marker, color=color, zorder=8)
    ax.text(
        xy[0] + text_dx, xy[1] + text_dy, label,
        fontsize=16, ha="left", va="bottom", zorder=9
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
    interpolation: str = "nearest",   # clipart crisp
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

def set_sprite_visible(sprite, visible: bool) -> None:
    sprite["im"].set_visible(visible)

# =============================================================================
# Static plot: rollouts + tubes + obstacles (DeepReach-style)
# =============================================================================
def plot_rollouts_tubes_centers(
    xs: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lowers_xy_step: np.ndarray | None,
    uppers_xy_step: np.ndarray | None,
    tube_stride: int,
    margin: float,
    filename: str,
    dpi: int = 300,
    xlim: tuple[float, float] | None = (-0.8, 1.2),
    ylim: tuple[float, float] | None = (-1.0, 0.6),
):
    xs = np.asarray(xs)
    if xs.ndim == 2 and xs.shape[1] == 3:
        xs = xs[None, :, :]
    if xs.ndim != 3 or xs.shape[2] != 3:
        raise ValueError(f"xs must be (n_rollouts,T,3) or (T,3). Got {xs.shape}")

    n_rollouts = xs.shape[0]

    # Use fixed DeepReach-like limits unless user wants auto
    if xlim is None or ylim is None:
        all_x = [xs[:, :, 0].ravel()]
        all_y = [xs[:, :, 1].ravel()]
        if lowers_xy_step is not None and uppers_xy_step is not None:
            all_x.append(lowers_xy_step[:, 0].ravel()); all_x.append(uppers_xy_step[:, 0].ravel())
            all_y.append(lowers_xy_step[:, 1].ravel()); all_y.append(uppers_xy_step[:, 1].ravel())
        if centers.size:
            all_x.append(centers[:, 0].ravel()); all_y.append(centers[:, 1].ravel())
        all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
        ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # Labels like your DeepReach plot
    draw_labeled_point(ax, START_XY, "Start", marker="o", text_dx=0.03, text_dy=0.04)
    draw_labeled_point(ax, GOAL_XY,  "Goal",  marker="*", text_dx=-0.06, text_dy=0.05)

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

    # Rollouts
    for i in range(n_rollouts):
        ax.plot(
            xs[i, :, 0],
            xs[i, :, 1],
            color=DEEPREACH_ORANGE,
            linewidth=ROLL_LINEWIDTH,
            alpha=ROLL_ALPHA_PNG,
            zorder=3,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$", fontsize=20)
    ax.set_ylabel(r"$p_y$", fontsize=20)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Legend (DeepReach-like)
    legend_handles = [
        Patch(facecolor=OBSTACLE_FACE, edgecolor=OBSTACLE_EDGE, alpha=OBSTACLE_ALPHA, label="Obstacle"),
        Line2D([], [], color=DEEPREACH_ORANGE, linewidth=ROLL_LINEWIDTH, alpha=0.75, label="DeepReach\nrollouts"),
    ]
    if lowers_xy_step is not None and uppers_xy_step is not None:
        legend_handles.insert(
            1,
            Patch(facecolor=TUBE_FACE, edgecolor="none", alpha=TUBE_ALPHA, label="Tubes")
        )

    ax.legend(handles=legend_handles, loc="lower right", fontsize=16, framealpha=0.9, ncol=1)

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# MP4: animated rollouts + tubes + obstacles + racecars (DeepReach-style)
# =============================================================================
def save_rollouts_mp4(
    xs: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lowers_xy_step: np.ndarray | None,
    uppers_xy_step: np.ndarray | None,
    sprite_rgba: np.ndarray,
    filename: str,
    dt: float,
    fps: int | None,
    tube_stride: int,
    margin: float,
    sprite_length: float,
    sprite_width: float,
    sprite_alpha: float,
    xlim: tuple[float, float] | None = (-0.8, 1.2),
    ylim: tuple[float, float] | None = (-1.0, 0.6),
):
    xs = np.asarray(xs)
    if xs.ndim != 3 or xs.shape[2] != 3:
        raise ValueError(f"xs must be (n_rollouts,T,3). Got {xs.shape}")

    n_rollouts, T, _ = xs.shape

    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))
    freeze_frames = int(round(1.0 * fps)) 

    # Fixed DeepReach-like limits (keeps your look consistent)
    if xlim is None or ylim is None:
        all_x = [xs[:, :, 0].ravel()]
        all_y = [xs[:, :, 1].ravel()]
        if lowers_xy_step is not None and uppers_xy_step is not None:
            all_x.append(lowers_xy_step[:, 0].ravel()); all_x.append(uppers_xy_step[:, 0].ravel())
            all_y.append(lowers_xy_step[:, 1].ravel()); all_y.append(uppers_xy_step[:, 1].ravel())
        if centers.size:
            all_x.append(centers[:, 0].ravel()); all_y.append(centers[:, 1].ravel())
        all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
        ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$p_x$", fontsize=20)
    ax.set_ylabel(r"$p_y$", fontsize=20)
    ax.grid(True, alpha=GRID_ALPHA)

    # Start/Goal labels
    draw_labeled_point(ax, START_XY, "Start", marker="o", text_dx=0.03, text_dy=0.04)
    draw_labeled_point(ax, GOAL_XY,  "Goal",  marker="*", text_dx=-0.06, text_dy=0.05)

    # Tubes (static)
    tube_pc = None
    if lowers_xy_step is not None and uppers_xy_step is not None:
        tube_pc = make_tube_patch_collection(
            lowers_xy_step, uppers_xy_step,
            tube_stride=tube_stride,
            alpha=TUBE_ALPHA,
            facecolor=TUBE_FACE,
            zorder=2,
        )
        ax.add_collection(tube_pc)

    # Obstacles
    if centers.size and radii.size == centers.shape[0]:
        draw_obstacles(ax, centers, radii)

    # Rollout lines + cars (DeepReach orange)
    lines = []
    cars = []
    goal_xy = np.array(GOAL_XY, dtype=float)
    goal_tol2 = float(GOAL_TOL) ** 2

    reached = np.zeros((n_rollouts,), dtype=bool)
    freeze_pose = np.zeros((n_rollouts, 3), dtype=float)
    freeze_t = -np.ones((n_rollouts,), dtype=int) 
    for i in range(n_rollouts):
        (ln,) = ax.plot([], [], lw=ROLL_LINEWIDTH, alpha=ROLL_ALPHA_MP4, color=DEEPREACH_ORANGE, zorder=3)
        lines.append(ln)

        valid0 = np.where(
            np.isfinite(xs[i, :, 0]) & np.isfinite(xs[i, :, 1]) & np.isfinite(xs[i, :, 2])
        )[0]
        if valid0.size:
            k0 = int(valid0[0])
            x0, y0, th0 = xs[i, k0]
        else:
            x0, y0, th0 = 0.0, 0.0, 0.0

        car = make_sprite_artist(
            ax, sprite_rgba,
            float(x0), float(y0), float(th0),
            length=sprite_length,
            width=sprite_width,
            alpha=sprite_alpha,
            zorder=6,
            interpolation="nearest",
        )
        set_sprite_visible(car, False)
        cars.append(car)

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        for ln in lines:
            ln.set_data([], [])
        for car in cars:
            set_sprite_visible(car, False)
        step_text.set_text("")
        artists = [*lines, step_text]
        for car in cars:
            artists.append(car["im"])
        if tube_pc is not None:
            artists.append(tube_pc)
        return artists

    def update(t: int):
        t = min(t, T - 1)
        for i in range(n_rollouts):
            # If already reached, freeze the trail at freeze_t[i]
            if reached[i]:
                tt = int(freeze_t[i])
                lines[i].set_data(xs[i, :tt + 1, 0], xs[i, :tt + 1, 1])

                update_sprite_artist(cars[i], freeze_pose[i, 0], freeze_pose[i, 1], freeze_pose[i, 2])
                set_sprite_visible(cars[i], True)
                continue

            # Not reached yet: extend trail up to current t
            lines[i].set_data(xs[i, : t + 1, 0], xs[i, : t + 1, 1])

            # If current state is valid, update car normally
            if np.isfinite(xs[i, t, 0]) and np.isfinite(xs[i, t, 1]) and np.isfinite(xs[i, t, 2]):
                x_t = float(xs[i, t, 0])
                y_t = float(xs[i, t, 1])
                th_t = float(xs[i, t, 2])

                # check goal
                dx = x_t - goal_xy[0]
                dy = y_t - goal_xy[1]
                if (dx * dx + dy * dy) <= goal_tol2:
                    reached[i] = True
                    freeze_t[i] = t
                    freeze_pose[i, :] = np.array([x_t, y_t, th_t], dtype=float)

                    # immediately freeze the line this frame
                    lines[i].set_data(xs[i, : t + 1, 0], xs[i, : t + 1, 1])

                update_sprite_artist(cars[i], x_t, y_t, th_t)
                set_sprite_visible(cars[i], True)
            else:
                set_sprite_visible(cars[i], False)

        artists = [*lines, step_text]
        for car in cars:
            artists.append(car["im"])
        if tube_pc is not None:
            artists.append(tube_pc)
        return artists



    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T + freeze_frames,
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
    ap.add_argument("--tube_step", type=int, default=0)
    ap.add_argument("--tube_stride", type=int, default=1)
    ap.add_argument("--no_mp4", action="store_true")
    ap.add_argument("--sprite_alpha", type=float, default=0.8)   # DeepReach-friendly subtle
    ap.add_argument("--sprite_length", type=float, default=0.18)
    ap.add_argument("--sprite_width", type=float, default=0.09)
    ap.add_argument("--xlim", type=float, nargs=2, default=[-0.8, 1.2])
    ap.add_argument("--ylim", type=float, nargs=2, default=[-1.0, 0.6])
    ap.add_argument("--fps", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=False)

    xs = d["xs"]                       # (N_ROLLOUTS, T, 3) NaN-padded
    centers = d["centers"]             # (K,2)
    radii = d["radii"]                 # (K,)
    lowers_xy = d["lowers_xy"]         # (n_plans, N+1, 2)
    uppers_xy = d["uppers_xy"]         # (n_plans, N+1, 2)
    dt = float(d["dt"])

    tube_step = int(max(0, min(args.tube_step, lowers_xy.shape[0] - 1)))
    lo_step = lowers_xy[tube_step]
    up_step = uppers_xy[tube_step]

    sprite_rgba = load_sprite_rgba(args.sprite)

    # PNG
    png_path = os.path.join(args.out_dir, "rollouts_tubes_centers.png")
    plot_rollouts_tubes_centers(
        xs=xs,
        centers=centers,
        radii=radii,
        lowers_xy_step=lo_step,
        uppers_xy_step=up_step,
        tube_stride=args.tube_stride,
        margin=0.2,
        filename=png_path,
        dpi=300,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
    )
    print(f"[Saved] {png_path}")

    # MP4
    if not args.no_mp4:
        mp4_path = os.path.join(args.out_dir, "rollouts.mp4")
        save_rollouts_mp4(
            xs=xs,
            centers=centers,
            radii=radii,
            lowers_xy_step=lo_step,
            uppers_xy_step=up_step,
            sprite_rgba=sprite_rgba,
            filename=mp4_path,
            dt=dt,
            fps=args.fps,
            tube_stride=args.tube_stride,
            margin=0.2,
            sprite_length=args.sprite_length,
            sprite_width=args.sprite_width,
            sprite_alpha=args.sprite_alpha,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
        )
        print(f"[Saved] {mp4_path}")

if __name__ == "__main__":
    main()
