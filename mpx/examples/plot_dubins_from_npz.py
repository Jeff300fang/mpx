#!/usr/bin/env python3
"""
plot_dubins_from_npz.py

Load a saved MPC rollout NPZ file and plot a single frame
(planned trajectory, executed trajectory, robust tubes, obstacles).

This script ALWAYS saves a PDF.

Usage:
  python plot_dubins_from_npz.py dubins_mpc_rollout.npz
  python plot_dubins_from_npz.py dubins_mpc_rollout.npz --step 20
  python plot_dubins_from_npz.py dubins_mpc_rollout.npz --show
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D



plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,   # embed fonts in PDF
    "ps.fonttype": 42,
})



def plot_frame(
    *,
    xs,
    plans_xy,
    lowers_xy,
    uppers_xy,
    centers,
    radii,
    step: int,
    box_stride: int = 1,
    margin: float = 0.5,
):
    """
    Render a single MPC frame.

    xs        : (T, 3)
    plans_xy  : (T, N+1, 2)
    lowers_xy : (T, N+1, 2)
    uppers_xy : (T, N+1, 2)
    centers   : (K, 2)
    radii     : (K,)
    step      : frame index to render
    """

    T = plans_xy.shape[0]
    step = int(np.clip(step, 0, T - 1))
    t_next = min(step + 1, xs.shape[0] - 1)

    # --- axis limits ---
    all_px = np.concatenate([
        xs[:, 0],
        plans_xy[:, :, 0].ravel(),
        lowers_xy[:, :, 0].ravel(),
        uppers_xy[:, :, 0].ravel(),
        centers[:, 0] if centers.size else np.array([]),
    ])
    all_py = np.concatenate([
        xs[:, 1],
        plans_xy[:, :, 1].ravel(),
        lowers_xy[:, :, 1].ravel(),
        uppers_xy[:, :, 1].ravel(),
        centers[:, 1] if centers.size else np.array([]),
    ])

    xmin, xmax = all_px.min() - margin, all_px.max() + margin
    ymin, ymax = all_py.min() - margin, all_py.max() + margin

    # --- figure ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # --- obstacles ---
    for c, r in zip(centers, radii):
        ax.add_patch(
            plt.Circle(
                (float(c[0]), float(c[1])),
                float(r),
                color="red",
                alpha=0.35,
            )
        )

    # --- executed trajectory ---
    ax.plot(
        xs[: t_next + 1, 0],
        xs[: t_next + 1, 1],
        lw=2,
        label="Executed Trajectory",
    )

    # --- planned trajectory ---
    ax.plot(
        plans_xy[step, :, 0],
        plans_xy[step, :, 1],
        lw=2,
        ls="--",
        label="Planned Trajectory",
    )

    # --- tube rectangles ---
    lo = lowers_xy[step]
    up = uppers_xy[step]

    rects = []
    stride = max(1, int(box_stride))
    for k in range(0, lo.shape[0], stride):
        w = up[k, 0] - lo[k, 0]
        h = up[k, 1] - lo[k, 1]
        if not np.isfinite(w) or not np.isfinite(h):
            continue
        if w <= 0 or h <= 0:
            continue
        rects.append(Rectangle((lo[k, 0], lo[k, 1]), w, h))

    tube_boxes = PatchCollection(
        rects,
        alpha=0.20,
    )
    ax.add_collection(tube_boxes)

    # --- legend proxy for tubes ---
    tube_proxy = Patch(
        facecolor="tab:blue",
        alpha=0.50,
        edgecolor="none",
        label="Robust tubes",
    )

    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()

    # handles[0] = Executed
    # handles[1] = Planned

    # Dummy spacer to occupy column 2 of row 2
    spacer = Line2D([], [], linestyle="none", marker=None, label="")

    handles = [
        handles[0],  # row 1, col 1
        handles[1],  # row 1, col 2
        tube_proxy,  # row 2, col 1
        spacer,      # row 2, col 2 (empty)
    ]

    labels = [
        labels[0],
        labels[1],
        "Robust tubes",
        "",
    ]

    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=2,
        framealpha=0.9,
        handlelength=2.0,
        columnspacing=1.5,
    )





    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=str, help="Path to dubins_mpc_rollout.npz")
    parser.add_argument("--step", type=int, default=-1, help="MPC step to plot (default: last)")
    parser.add_argument("--stride", type=int, default=1, help="Tube box stride")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    args = parser.parse_args()

    data = np.load(args.npz)

    xs        = data["xs"]
    plans_xy = data["plans_xy"]
    lowers   = data["lowers_xy"]
    uppers   = data["uppers_xy"]
    centers  = data["centers"]
    radii    = data["radii"]

    if args.step < 0:
        step = plans_xy.shape[0] - 1
        pdf_name = "big_dubins.pdf"
    else:
        step = args.step
        pdf_name = f"frame_step_{step}.pdf"

    fig = plot_frame(
        xs=xs,
        plans_xy=plans_xy,
        lowers_xy=lowers,
        uppers_xy=uppers,
        centers=centers,
        radii=radii,
        step=step,
        box_stride=args.stride,
    )

    fig.savefig(pdf_name, bbox_inches="tight")
    print(f"Saved PDF → {pdf_name}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
