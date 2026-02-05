import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D
from matplotlib import animation

# ============================
# Hard-coded paths
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NPZ_PATH   = os.path.join(SCRIPT_DIR, "rollouts_xy_data_test.npz")
PDF_OUT    = os.path.join(SCRIPT_DIR, "deepreach.pdf")
MP4_OUT    = os.path.join(SCRIPT_DIR, "deepreach.mp4")

# PNG sprite (transparent) to render as the "car"
SPRITE_PNG = os.path.join(SCRIPT_DIR, "car_removed.png")


import matplotlib as mpl

mpl.rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,        # keep False unless you want full LaTeX
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


def maybe(npz, key, default=None):
    return npz[key] if key in npz.files else default


def draw_circle(ax, center, radius):
    c = plt.Circle(
        center,
        radius,
        facecolor="red",
        edgecolor="darkred",
        alpha=0.25,
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(c)


def draw_goal(ax, goal_point, goal_tol):
    g = plt.Circle(
        goal_point,
        goal_tol,
        facecolor="gold",
        edgecolor="goldenrod",
        alpha=0.25,
        linewidth=1.5,
        zorder=6,
    )
    ax.add_patch(g)
    ax.text(
        goal_point[0] + goal_tol + 0.02,
        goal_point[1] + 0.02,
        "Goal tol",
        fontsize=9,
    )


def draw_init_box(ax, init_center, init_half_extents):
    cx, cy = init_center
    hx, hy = init_half_extents
    r = plt.Rectangle(
        (cx - hx, cy - hy),
        2 * hx,
        2 * hy,
        facecolor="lightgreen",
        alpha=0.25,
        edgecolor="forestgreen",
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(r)


def draw_labeled_point(ax, xy, label, *, color="black", marker="o", text_dx=0.03, text_dy=0.03):
    ax.scatter(
        xy[0],
        xy[1],
        s=70,
        marker=marker,
        color=color,
        zorder=8,
    )
    ax.text(
        xy[0] + text_dx,
        xy[1] + text_dy,
        label,
        fontsize=16,
        ha="left",
        va="bottom",
        zorder=9,
    )


# ============================
# Sprite helpers (PNG + Affine2D)
# ============================
def load_sprite_rgba(path: str) -> np.ndarray:
    img = plt.imread(path)
    # Ensure RGBA
    if img.ndim == 2:
        img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    elif img.shape[-1] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], dtype=img.dtype)])
    return img


def make_sprite_artist(
    ax,
    img_rgba: np.ndarray,
    x: float,
    y: float,
    theta: float,
    length: float,
    width: float,
    alpha: float,
    zorder: int = 10,
    interpolation: str = "nearest",  # crisp for clipart
):
    """
    Create a car sprite centered at (x,y) rotated by theta.
    Size is controlled in WORLD units via (length,width).
    """
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


def update_sprite_artist(sprite, x: float, y: float, theta: float):
    tf: Affine2D = sprite["tf"]
    tf.clear()
    tf.rotate(theta)
    tf.translate(x, y)


def set_sprite_visible(sprite, visible: bool):
    sprite["im"].set_visible(visible)


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")
    if not os.path.exists(SPRITE_PNG):
        raise FileNotFoundError(
            f"Sprite PNG not found: {SPRITE_PNG}\n"
            f"Put racecar_topdown.png next to this script (or change SPRITE_PNG)."
        )

    print(f"[Loading] {NPZ_PATH}")
    npz = np.load(NPZ_PATH, allow_pickle=True)

    paths = npz["paths"]          # (N, T, 3)
    lengths = npz["lengths"]      # (N,)

    goal_point = maybe(npz, "goal_point")
    goal_tol = maybe(npz, "goal_tol")
    obs_center = maybe(npz, "obs_center")
    obs_radius = maybe(npz, "obs_radius")
    init_center = maybe(npz, "init_center")
    init_half_extents = maybe(npz, "init_half_extents")

    N, T, D = paths.shape
    assert D == 3, f"Expected paths[...,3], got {paths.shape}"

    # ============================
    # 1) Save the STATIC PDF (your original plot)
    # ============================
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    draw_labeled_point(ax, (-0.75, -0.75), "Start", color="black", marker="o", text_dx=0.03, text_dy=0.04)
    draw_labeled_point(ax, (1.0, 0.4), "Goal", color="black", marker="*", text_dx=-0.06, text_dy=0.05)

    if obs_center is not None and obs_radius is not None:
        draw_circle(ax, (float(obs_center[0]), float(obs_center[1])), float(obs_radius))

    if init_center is not None and init_half_extents is not None:
        draw_init_box(ax, init_center, init_half_extents)

    if goal_point is not None and goal_tol is not None:
        draw_goal(ax, (float(goal_point[0]), float(goal_point[1])), float(goal_tol))

    for i in range(N):
        Li = int(lengths[i])
        if Li <= 0:
            continue
        traj = paths[i, :Li]
        x, y = traj[:, 0], traj[:, 1]
        ax.plot(x, y, color="#ff7f0e", linewidth=3.0, alpha=0.4, zorder=2)

        # crash criterion
        if x[-1] < 0.8:
            ax.scatter(x[-1], y[-1], marker="X", s=80, color="darkred", zorder=7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$p_x$", fontsize=20)
    ax.set_ylabel("$p_y$", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.8, 1.2)
    ax.set_ylim(-1, 0.6)

    legend_handles = [
        Patch(facecolor="red", edgecolor="darkred", alpha=0.25, label="Obstacle"),
        Line2D([], [], marker="X", linestyle="None", color="darkred", markersize=8, label="Crashes"),
        Line2D([], [], color="#ff7f0e", linewidth=3.0, alpha=0.75, label="DeepReach\nrollouts"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=16, ncol=1, framealpha=0.9)

    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {PDF_OUT}")

    # ============================
    # 2) Save an MP4 animation using the PNG sprite
    # ============================
    # Load sprite once
    sprite_rgba = load_sprite_rgba(SPRITE_PNG)

    # Animation parameters (tune as you like)
    dt = float(maybe(npz, "dt", 0.1))
    fps = max(10, int(round(1.0 / dt)))
    sprite_alpha = 0.8
    sprite_length = 0.18  # <-- make smaller by decreasing these
    sprite_width  = 0.09

    # Precompute max length to set frames
    max_L = int(np.max(lengths)) if len(lengths) else 0
    max_L = max(1, min(max_L, T))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Same decorations as PDF
    draw_labeled_point(ax, (-0.75, -0.75), "Start", color="black", marker="o", text_dx=0.03, text_dy=0.04)
    draw_labeled_point(ax, (1.0, 0.4), "Goal", color="black", marker="*", text_dx=-0.06, text_dy=0.05)

    if obs_center is not None and obs_radius is not None:
        draw_circle(ax, (float(obs_center[0]), float(obs_center[1])), float(obs_radius))
    if init_center is not None and init_half_extents is not None:
        draw_init_box(ax, init_center, init_half_extents)
    if goal_point is not None and goal_tol is not None:
        draw_goal(ax, (float(goal_point[0]), float(goal_point[1])), float(goal_tol))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$p_x$", fontsize=20)
    ax.set_ylabel("$p_y$", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.8, 1.2)
    ax.set_ylim(-1, 0.6)

    # Create line artists and sprite artists (one per rollout)
    lines = []
    cars = []
    crash_markers = []  # X markers at final point if crash (appear when rollout completes)

    for i in range(N):
        (ln,) = ax.plot([], [], color="#ff7f0e", linewidth=3.0, alpha=0.4, zorder=2)
        lines.append(ln)

        # Initialize sprite at first valid point (if any)
        Li = int(lengths[i])
        if Li > 0:
            x0, y0, th0 = paths[i, 0, 0], paths[i, 0, 1], paths[i, 0, 2]
        else:
            x0, y0, th0 = 0.0, 0.0, 0.0

        car = make_sprite_artist(
            ax,
            sprite_rgba,
            float(x0), float(y0), float(th0),
            length=sprite_length,
            width=sprite_width,
            alpha=sprite_alpha,
            zorder=10,
            interpolation="nearest",
        )
        set_sprite_visible(car, False)
        cars.append(car)

        # Crash marker artist (hidden until completion)
        mk = ax.scatter([], [], marker="X", s=80, color="darkred", zorder=7)
        mk.set_visible(False)
        crash_markers.append(mk)

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        for ln in lines:
            ln.set_data([], [])
        for car in cars:
            set_sprite_visible(car, False)
        for mk in crash_markers:
            mk.set_visible(False)
        title.set_text("")
        artists = [*lines, title]
        for car in cars:
            artists.append(car["im"])
        artists.extend(crash_markers)
        return artists

    def update(k: int):
        # k is "time index" into each rollout
        for i in range(N):
            Li = int(lengths[i])
            if Li <= 0:
                lines[i].set_data([], [])
                set_sprite_visible(cars[i], False)
                crash_markers[i].set_visible(False)
                continue

            kk = min(k, Li - 1)
            traj = paths[i, :kk + 1]
            x, y, th = traj[:, 0], traj[:, 1], traj[:, 2]
            lines[i].set_data(x, y)

            # Move sprite to current point
            update_sprite_artist(cars[i], float(x[-1]), float(y[-1]), float(th[-1]))
            set_sprite_visible(cars[i], True)

            # If rollout finished (k >= Li-1), show crash marker if criterion met
            if k >= Li - 1:
                if float(paths[i, Li - 1, 0]) < 0.8:
                    crash_markers[i].set_offsets([[float(paths[i, Li - 1, 0]), float(paths[i, Li - 1, 1])]])
                    crash_markers[i].set_visible(True)
                else:
                    crash_markers[i].set_visible(False)
            else:
                crash_markers[i].set_visible(False)

        # title.set_text(f"DeepReach rollouts (frame {k+1}/{max_L})")
        artists = [*lines, title]
        for car in cars:
            artists.append(car["im"])
        artists.extend(crash_markers)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_L,
        init_func=init,
        blit=True,
        interval=int(round(1000.0 / fps)),
    )

    if not animation.FFMpegWriter.isAvailable():
        plt.close(fig)
        raise RuntimeError("ffmpeg not available. Install ffmpeg to save MP4, or switch to GIF.")

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(MP4_OUT, writer=writer, dpi=200)
    plt.close(fig)

    print(f"[Saved] {MP4_OUT}")


if __name__ == "__main__":
    main()
