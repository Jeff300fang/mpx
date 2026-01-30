import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================
# Hard-coded NPZ path
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "rollouts_xy_data_test.npz")
PDF_OUT = os.path.join(SCRIPT_DIR, "deepreach.pdf")


plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,   # embed fonts in PDF
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

def draw_labeled_point(ax, xy, label, *, color="black", marker="o"):
    ax.scatter(
        xy[0],
        xy[1],
        s=60,
        marker=marker,
        color=color,
        zorder=8,
    )
    ax.text(
        xy[0] + 0.03,
        xy[1] + 0.03,
        label,
        fontsize=10,
        ha="left",
        va="bottom",
        zorder=9,
    )


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

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

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    draw_labeled_point(ax, (-0.75, -0.75), "Start", color="black")
    draw_labeled_point(ax, (1.0, 0.4), "Goal", color="black", marker="*")

    # Decorations
    if obs_center is not None and obs_radius is not None:
        draw_circle(
            ax,
            (float(obs_center[0]), float(obs_center[1])),
            float(obs_radius),
        )

    if init_center is not None and init_half_extents is not None:
        draw_init_box(ax, init_center, init_half_extents)

    if goal_point is not None and goal_tol is not None:
        draw_goal(
            ax,
            (float(goal_point[0]), float(goal_point[1])),
            float(goal_tol),
        )

    # Plot trajectories
    for i in range(N):
        Li = int(lengths[i])
        if Li <= 0:
            continue

        traj = paths[i, :Li]
        x, y = traj[:, 0], traj[:, 1]

        ax.plot(
            x,
            y,
            color="#ff7f0e",
            linewidth=3.0,
            alpha=0.4,
            zorder=2,
        )

        # ---- crash criterion ----
        if x[-1] < 0.8:
            ax.scatter(
                x[-1],
                y[-1],
                marker="X",
                s=80,
                color="darkred",
                zorder=7,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1.2)
    ax.set_ylim(-1, 0.6)

    legend_handles = [
        Patch(
            facecolor="red",
            edgecolor="darkred",
            alpha=0.25,
            label="Obstacle",
        ),
        Line2D(
            [],
            [],
            marker="X",
            linestyle="None",
            color="darkred",
            markersize=8,
            label="Crashes",
        ),
        Line2D(
            [],
            [],
            color="#ff7f0e",
            linewidth=3.0,
            alpha=0.75,
            label="DeepReach rollouts",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        framealpha=0.9,
    )
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {PDF_OUT}")


if __name__ == "__main__":
    main()
