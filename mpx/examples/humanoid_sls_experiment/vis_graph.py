#!/usr/bin/env python3
"""
plot_tubes_vs_diff_xy_pdf.py

Loads:
  - tubes_x, diff_x, tubes_y, diff_y

Renders:
  - Left:  X (tube vs diff)
  - Right: Y (tube vs diff)

Saves:
  - tubes_vs_diff_xy.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to data.npz")
    ap.add_argument("--dt", type=float, default=None, help="Optional timestep for time axis (s)")
    ap.add_argument("--out", default="tubes_vs_diff_xy.pdf", help="Output PDF filename")
    args = ap.parse_args()

    data = np.load(args.npz)
    required = ["tubes_x", "diff_x", "tubes_y", "diff_y"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"NPZ missing keys {missing}. Found: {data.files}")

    tubes_x = np.asarray(data["tubes_x"], dtype=float).ravel()
    diff_x  = np.asarray(data["diff_x"],  dtype=float).ravel()
    tubes_y = np.asarray(data["tubes_y"], dtype=float).ravel()
    diff_y  = np.asarray(data["diff_y"],  dtype=float).ravel()

    T = min(len(tubes_x), len(diff_x), len(tubes_y), len(diff_y))
    tubes_x, diff_x = tubes_x[:T], diff_x[:T]
    tubes_y, diff_y = tubes_y[:T], diff_y[:T]

    # x-axis
    if args.dt is None:
        x = np.arange(T)
        xlabel = "Step"
    else:
        x = np.arange(T) * float(args.dt)
        xlabel = "Time [s]"

    # ---- Side-by-side figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # X plot
    axes[0].plot(x, tubes_x, label=r"$p_x$ tube size", linewidth=1.8)
    axes[0].plot(x, diff_x, label=r"$p_x$" + " deviation from \nplanned trajectory", linewidth=1.8)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True)
    # axes[0].legend()
    axes[0].legend()
    # Y plot
    axes[1].plot(x, tubes_y, label=r"$p_y$ tube size", linewidth=1.8)
    axes[1].plot(x, diff_y, label=r"$p_y$ deviation from" + "\nplanned trajectory", linewidth=1.8)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PDF: {args.out}")


if __name__ == "__main__":
    main()
