#!/usr/bin/env python3
"""
plot_npz_tubes.py

Load:
  - X          (T, nx)   planned / stored trajectory
  - tube_sizes (T, nx)   half-width tube sizes per state dim
  - X_rollout  (T, nx)   executed / rollout trajectory

Then plot planned ± tube band (shaded) with actual overlaid.

Usage:
  python plot_npz_tubes.py

Edit the USER CONFIG section below.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =========================
# USER CONFIG
# =========================
NPZ_PATH = "data.npz"
OUT_PATH = "tube_band_grid.pdf"   # .png or .pdf
DT = 0.02                         # time step for x-axis

# Which state indices to plot:
# - set IDX = [0, 1, 2] etc
# - or set IDX = None to plot first K
IDX = [0, 1, 2, 26, 27, 28]          # example: base_x, base_y if that matches your layout
MAX_DIMS_IF_NONE = 12 # used only if IDX is None

NCOLS = 3             # grid columns
FIG_W = 14            # inches
ROW_H = 3.2           # inches per row

# Visual knobs
BAND_ALPHA = 0.18
LINEWIDTH = 1.6
MARK_VIOLATIONS = True   # mark points where actual exits the band
VIOL_MARKERSIZE = 3.0
# =========================


def save_band_grid(
    plan: np.ndarray,
    actual: np.ndarray,
    tube: np.ndarray,
    dt: float,
    idx: np.ndarray,
    out_path: str,
    ncols: int,
    suptitle: str,
):
    plan = np.asarray(plan)
    actual = np.asarray(actual)
    tube = np.asarray(tube)
    idx = np.asarray(idx, dtype=int)

    if plan.ndim != 2 or actual.ndim != 2 or tube.ndim != 2:
        raise ValueError(f"Expected 2D arrays. got {plan.shape=} {actual.shape=} {tube.shape=}")

    T = min(plan.shape[0], actual.shape[0], tube.shape[0])
    if T <= 1:
        raise ValueError(f"Not enough timesteps (T={T}).")

    plan = plan[:T]
    actual = actual[:T]
    tube = tube[:T]

    # Slice dims
    plan_plot = plan[:, idx]
    act_plot = actual[:, idx]
    tube_plot = tube[:, idx]  # half widths

    lower = plan_plot - tube_plot
    upper = plan_plot + tube_plot

    n_idx = idx.size
    nrows = int(math.ceil(n_idx / ncols))
    t = np.arange(T) * float(dt)

    fig_h = ROW_H * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, fig_h), sharex=True)
    axes = np.atleast_2d(axes)

    for j in range(n_idx):
        r = j // ncols
        c = j % ncols
        ax = axes[r, c]

        # Band
        ax.fill_between(
            t,
            lower[:, j],
            upper[:, j],
            alpha=BAND_ALPHA,
            label="planned ± tube" if j == 0 else None,
        )

        # Planned centerline
        ax.plot(
            t,
            plan_plot[:, j],
            linestyle="--",
            linewidth=LINEWIDTH,
            label="planned" if j == 0 else None,
        )

        # Actual
        ax.plot(
            t,
            act_plot[:, j],
            linewidth=LINEWIDTH,
            label="actual" if j == 0 else None,
        )

        # Optional: mark violations
        if MARK_VIOLATIONS:
            viol = (act_plot[:, j] < lower[:, j]) | (act_plot[:, j] > upper[:, j])
            if np.any(viol):
                ax.plot(
                    t[viol],
                    act_plot[viol, j],
                    linestyle="None",
                    marker="o",
                    markersize=VIOL_MARKERSIZE,
                )

        ax.set_title(f"state[{idx[j]}]", fontsize=10)
        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Turn off unused axes
    for j in range(n_idx, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    # Figure-level legend (pull from first axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(suptitle, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PATH}")

    data = np.load(NPZ_PATH)
    for k in ["X", "tube_sizes", "X_rollout"]:
        if k not in data.files:
            raise RuntimeError(f"NPZ missing key '{k}'. Found keys: {list(data.files)}")

    X = np.asarray(data["X"], dtype=np.float64)
    tube_sizes = np.asarray(data["tube_sizes"], dtype=np.float64)
    X_rollout = np.asarray(data["X_rollout"], dtype=np.float64)

    T = min(X.shape[0], tube_sizes.shape[0], X_rollout.shape[0])
    nx = X.shape[1]
    print(f"Loaded {NPZ_PATH}: X{X.shape}, tube_sizes{tube_sizes.shape}, X_rollout{X_rollout.shape}")
    print(f"Using T={T}, nx={nx}")

    if IDX is None:
        idx = np.arange(min(nx, int(MAX_DIMS_IF_NONE)))
    else:
        idx = np.array(IDX, dtype=int)
        if np.any(idx < 0) or np.any(idx >= nx):
            raise ValueError(f"IDX has out-of-range entries for nx={nx}: {idx.tolist()}")

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    save_band_grid(
        plan=X[:T],
        actual=X_rollout[:T],
        tube=tube_sizes[:T],
        dt=float(DT),
        idx=idx,
        out_path=OUT_PATH,
        ncols=int(NCOLS),
        suptitle="Planned ± tube band with executed rollout",
    )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
