import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection

# ============================
# Hard-coded NPZ path
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "sls_vs_deepreach.npz")  # <-- adjust if needed
PDF_OUT = os.path.join(SCRIPT_DIR, "deepreach_with_tubes.pdf")

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
        fontsize=12,
        ha="left",
        va="bottom",
        zorder=9,
    )


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    print(f"[Loading] {NPZ_PATH}")
    npz = np.load(NPZ_PATH, allow_pickle=True)

    # --- Required arrays (from your saver) ---
    xs = np.asarray(npz["xs"])                 # (N_rollouts, T, 3) or (T, 3)
    centers = np.asarray(npz["centers"])       # (K, 2)
    radii = np.asarray(npz["radii"]).reshape(-1)

    lowers_xy = np.asarray(npz["lowers_xy"])   # (n_steps, N+1, 2)
    uppers_xy = np.asarray(npz["uppers_xy"])   # (n_steps, N+1, 2)
    plans_xy  = maybe(npz, "plans_xy", None)   # optional

    # If you saved only one tube set, step_idx=0 is correct
    step_idx = 0
    step_idx = int(np.clip(step_idx, 0, lowers_xy.shape[0] - 1))

    # Normalize xs to (N_rollouts, T, 3)
    if xs.ndim == 2 and xs.shape[1] == 3:
        xs = xs[None, :, :]
    if xs.ndim != 3 or xs.shape[2] != 3:
        raise ValueError(f"Expected xs shape (N_rollouts, T, 3). Got {xs.shape}")

    n_rollouts, T, _ = xs.shape

    # Your desired annotations
    start_xy = (-0.75, -0.75)
    goal_xy = (1.0, 0.4)

    # --- Styling ---
    rollout_color = "#ff7f0e"   # muted orange
    crash_color = "darkred"
    tube_face = "tab:blue"      # tubes color (fill)
    tube_alpha = 0.12
    tube_stride = 1             # increase to 2/3 if you want fewer rectangles

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # --- Start / Goal annotations ---
    draw_labeled_point(ax, start_xy, "Start", color="black", marker="o")
    draw_labeled_point(ax, goal_xy, "Goal", color="black", marker="*", text_dx=0.03, text_dy=0.03)

    # --- Obstacles ---
    if centers.size and radii.size:
        for c, r in zip(centers, radii):
            draw_circle(ax, (float(c[0]), float(c[1])), float(r))

    # --- Tubes (from lowers/uppers at chosen step) ---
    lo = lowers_xy[step_idx]  # (N+1, 2)
    up = uppers_xy[step_idx]  # (N+1, 2)

    rects = []
    stride = max(1, int(tube_stride))
    for k in range(0, lo.shape[0], stride):
        w = up[k, 0] - lo[k, 0]
        h = up[k, 1] - lo[k, 1]
        if not np.isfinite(w) or not np.isfinite(h):
            continue
        if w <= 0.0 or h <= 0.0:
            continue
        rects.append(Rectangle((lo[k, 0], lo[k, 1]), w, h))

    tube_boxes = PatchCollection(
        rects,
        facecolor=tube_face,
        edgecolor="none",
        alpha=tube_alpha,
        zorder=1,
    )
    ax.add_collection(tube_boxes)

    # --- Optional: planned nominal (if present) ---

    # --- Rollouts + crash markers ---
    for i in range(n_rollouts):
        x = xs[i, :, 0]
        y = xs[i, :, 1]

        ax.plot(
            x,
            y,
            color=rollout_color,
            linewidth=3.0,
            alpha=0.40,
            zorder=2,
        )

    # --- Axes styling ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)

    # Match your framing (edit if desired)
    ax.set_xlim(-1.0, 1.2)
    ax.set_ylim(-1.0, 0.6)

    # --- Legend (Obstacle / Crashes / DeepReach rollouts / Robust tubes) ---
    # You asked earlier for 3 items; now that we're plotting tubes, I’m including it as a 4th.
    # If you want ONLY 3 entries, tell me which one to drop.
    legend_handles = [
        Patch(facecolor="red", edgecolor="darkred", alpha=0.25, label="Obstacle"),
        Line2D([], [], marker="X", linestyle="None", color=crash_color, markersize=9, label="Crashes"),
        Line2D([], [], color=rollout_color, linewidth=3.0, alpha=0.75, label="GPU-SLS rollouts"),
        Patch(facecolor=tube_face, edgecolor="none", alpha=min(0.35, tube_alpha * 3.0), label="Robust tubes"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,               # 2 columns to avoid a super-wide legend now that there are 4 items
        framealpha=0.9,
    )

    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {PDF_OUT}")


if __name__ == "__main__":
    main()
