#!/usr/bin/env python3
"""
plot_npz_tubes.py

Load from data.npz:
  - X            (T, nx)                 planned / stored trajectory
  - tube_sizes   (T, nx)                 half-width tube sizes per state dim
  - Either:
      - X_rollout   (T, nx)              single executed rollout
    or
      - X_rollouts  (R, T, nx)           multiple executed rollouts

Then plot planned ± tube band (shaded) with actual rollout(s) overlaid.

This version:
  - Supports BOTH single-rollout and multi-rollout NPZ outputs.
  - For multi-rollouts, overlays multiple trajectories (optionally a subset).
  - Can optionally plot an envelope (min/max) or quantile band across rollouts.

Usage:
  python plot_npz_tubes.py

Edit the USER CONFIG section below.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 20,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,   # embed fonts in PDF
    "ps.fonttype": 42,
})



# =========================
# USER CONFIG
# =========================
NPZ_PATH = "data.npz"
OUT_PATH = "tube_band_grid.pdf"   # .png or .pdf
DT = 0.05                         # time step for x-axis

# Which state indices to plot:
# - set IDX = [0, 1, 2] etc
# - or set IDX = None to plot first K
IDX = [0, 1, 2, 26, 27, 28]
MAX_DIMS_IF_NONE = 12  # used only if IDX is None

NCOLS = 3              # grid columns
FIG_W = 14             # inches
ROW_H = 3.2            # inches per row

# Visual knobs
BAND_ALPHA = 0.18
LINEWIDTH = 1.6
MARK_VIOLATIONS = True         # mark points where actual exits the band
VIOL_MARKERSIZE = 2.8

# Multi-rollout plotting controls
# If X_rollouts exists:
PLOT_MAX_ROLLOUTS = 30         # overlay at most this many rollouts (for readability)
ROLLOUT_STRIDE = 1             # take every k-th rollout (after optional cap)
ROLLOUT_ALPHA = 0.35           # alpha for each rollout line
HIGHLIGHT_ROLLOUT_IDX = None      # highlight one rollout thicker (None to disable)

# Optional: draw rollout envelope / quantiles across rollouts
DRAW_ENVELOPE = False          # if True, fill between min/max across rollouts
ENVELOPE_ALPHA = 0.12

DRAW_QUANTILES = False         # if True, fill between quantiles across rollouts
Q_LO = 0.10
Q_HI = 0.90
QUANT_ALPHA = 0.14
# =========================


def _select_rollouts(X_rollouts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_sel, sel_indices) where:
      X_sel: (R_sel, T, nx)
      sel_indices: (R_sel,) original rollout indices
    """
    R = int(X_rollouts.shape[0])
    stride = max(int(ROLLOUT_STRIDE), 1)
    cand = np.arange(0, R, stride, dtype=int)

    if cand.size > int(PLOT_MAX_ROLLOUTS):
        cand = cand[: int(PLOT_MAX_ROLLOUTS)]

    return X_rollouts[cand], cand


def save_band_grid_multi(
    plan: np.ndarray,                 # (T, nx)
    tube: np.ndarray,                 # (T, nx)
    actuals: np.ndarray,              # (R, T, nx) or (T, nx)
    dt: float,
    idx: np.ndarray,
    out_path: str,
    ncols: int,
):
    plan = np.asarray(plan, dtype=np.float64)
    tube = np.asarray(tube, dtype=np.float64)
    idx = np.asarray(idx, dtype=int)

    if plan.ndim != 2 or tube.ndim != 2:
        raise ValueError(f"Expected plan and tube to be 2D. got {plan.shape=} {tube.shape=}")

    # Normalize actual(s) into shape (R, T, nx)
    actuals = np.asarray(actuals, dtype=np.float64)
    if actuals.ndim == 2:
        actuals = actuals[None, ...]  # (1, T, nx)
    elif actuals.ndim != 3:
        raise ValueError(f"Expected actuals to be 2D or 3D. got {actuals.shape=}")

    T = min(plan.shape[0], tube.shape[0], actuals.shape[1])
    if T <= 1:
        raise ValueError(f"Not enough timesteps (T={T}).")

    plan = plan[:T]
    tube = tube[:T]
    actuals = actuals[:, :T, :]

    nx = int(plan.shape[1])
    if tube.shape[1] != nx or actuals.shape[2] != nx:
        raise ValueError(f"Dimension mismatch: plan nx={nx}, tube nx={tube.shape[1]}, actual nx={actuals.shape[2]}")

    # Slice dims
    plan_plot = plan[:, idx]                      # (T, D)
    tube_plot = tube[:, idx]                      # (T, D)
    act_plot = actuals[:, :, idx]                 # (R, T, D)

    lower = plan_plot - tube_plot                 # (T, D)
    upper = plan_plot + tube_plot                 # (T, D)

    R = int(act_plot.shape[0])
    D = int(idx.size)

    nrows = int(math.ceil(D / ncols))
    t = np.arange(T) * float(dt)

    fig_h = ROW_H * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, fig_h), sharex=True)
    axes = np.atleast_2d(axes)

    # Decide which rollout to highlight (in selected-index space)
    hi = HIGHLIGHT_ROLLOUT_IDX
    if hi is not None:
        hi = int(hi)
        if hi < 0 or hi >= R:
            hi = None

    indx_mapping = [r"$p_x$ [m]",
                    r"$p_y$ [m]",
                    r"$p_z$ [m]",
                    r"$v_x$ [m/s]",
                    r"$v_y$ [m/s]",
                    r"$v_z$ [m/s]"]

    for j in range(D):
        r = j // ncols
        c = j % ncols
        ax = axes[r, c]

        # Planned tube band
        ax.fill_between(
            t,
            lower[:, j],
            upper[:, j],
            alpha=BAND_ALPHA,
            label="Robust Tube" if j == 0 else None,
        )

        # Optional envelope across rollouts (min/max)
        if DRAW_ENVELOPE and R > 1:
            mn = np.min(act_plot[:, :, j], axis=0)
            mx = np.max(act_plot[:, :, j], axis=0)
            ax.fill_between(
                t,
                mn,
                mx,
                alpha=ENVELOPE_ALPHA,
                label="rollout min/max" if j == 0 else None,
            )

        # Optional quantile band across rollouts
        if DRAW_QUANTILES and R > 1:
            qlo = np.quantile(act_plot[:, :, j], Q_LO, axis=0)
            qhi = np.quantile(act_plot[:, :, j], Q_HI, axis=0)
            ax.fill_between(
                t,
                qlo,
                qhi,
                alpha=QUANT_ALPHA,
                label=f"rollout q[{Q_LO:.2f},{Q_HI:.2f}]" if j == 0 else None,
            )

        # Planned centerline
        ax.plot(
            t,
            plan_plot[:, j],
            linestyle="--",
            linewidth=LINEWIDTH,
            label="Planned Trajectory" if j == 0 else None,
        )

        # Overlay rollouts
        for k in range(R):
            lw = LINEWIDTH
            a = ROLLOUT_ALPHA
            lab = None

            if (hi is not None) and (k == hi):
                lw = LINEWIDTH * 2.0
                a = 0.95
                lab = "highlight rollout" if j == 0 else None

            ax.plot(
                t,
                act_plot[k, :, j],
                linewidth=lw,
                alpha=a,
                label=lab,
                 color="tab:orange",
            )

        # Optional: mark violations (per rollout)
        if MARK_VIOLATIONS:
            for k in range(R):
                viol = (act_plot[k, :, j] < lower[:, j]) | (act_plot[k, :, j] > upper[:, j])
                if np.any(viol):
                    ax.plot(
                        t[viol],
                        act_plot[k, viol, j],
                        linestyle="None",
                        marker="o",
                        markersize=VIOL_MARKERSIZE,
                        alpha=min(1.0, ROLLOUT_ALPHA + 0.2),
                    )

        ax.set_ylabel(indx_mapping[j])
        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Turn off unused axes
    for j in range(D, nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        axes[rr, cc].axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    # Figure-level legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Add proxy handle for rollouts (orange lines)
    rollout_handle = Line2D(
        [0], [0],
        color="tab:orange",
        linewidth=LINEWIDTH,
        label="Rollouts",
    )

    handles.append(rollout_handle)
    labels.append("Rollouts")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),        # single row
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=2.8,
        columnspacing=1.8,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PATH}")

    data = np.load(NPZ_PATH)

    # Required keys
    for k in ["X", "tube_sizes"]:
        if k not in data.files:
            raise RuntimeError(f"NPZ missing key '{k}'. Found keys: {list(data.files)}")

    X = np.asarray(data["X"], dtype=np.float64)
    tube_sizes = np.asarray(data["tube_sizes"], dtype=np.float64)

    # Single vs multi-rollout
    has_multi = "X_rollouts" in data.files
    has_single = "X_rollout" in data.files

    if not (has_multi or has_single):
        raise RuntimeError(
            "NPZ must contain either 'X_rollout' (single) or 'X_rollouts' (multi).\n"
            f"Found keys: {list(data.files)}"
        )

    if has_multi:
        X_rollouts = np.asarray(data["X_rollouts"], dtype=np.float64)  # (R, T, nx)
        if X_rollouts.ndim != 3:
            raise ValueError(f"X_rollouts must be 3D (R,T,nx), got {X_rollouts.shape}")

        # Select subset for plotting (readability)
        X_sel, sel_idx = _select_rollouts(X_rollouts)

        T = min(X.shape[0], tube_sizes.shape[0], X_sel.shape[1])
        nx = int(X.shape[1])
        print(f"Loaded {NPZ_PATH}: X{X.shape}, tube_sizes{tube_sizes.shape}, X_rollouts{X_rollouts.shape}")
        print(f"Selecting rollouts: {sel_idx.tolist()}  -> X_sel{X_sel.shape}")
        print(f"Using T={T}, nx={nx}")

        actuals_for_plot = X_sel[:T if X_sel.shape[1] == T else X_sel.shape[0]]

        # (R, T, nx)
        actuals_for_plot = X_sel[:, :T, :]

    else:
        X_rollout = np.asarray(data["X_rollout"], dtype=np.float64)  # (T, nx)
        T = min(X.shape[0], tube_sizes.shape[0], X_rollout.shape[0])
        nx = int(X.shape[1])
        print(f"Loaded {NPZ_PATH}: X{X.shape}, tube_sizes{tube_sizes.shape}, X_rollout{X_rollout.shape}")
        print(f"Using T={T}, nx={nx}")

        actuals_for_plot = X_rollout[:T]  # (T, nx)

    # Choose indices
    if IDX is None:
        idx = np.arange(min(nx, int(MAX_DIMS_IF_NONE)))
    else:
        idx = np.array(IDX, dtype=int)
        if np.any(idx < 0) or np.any(idx >= nx):
            raise ValueError(f"IDX has out-of-range entries for nx={nx}: {idx.tolist()}")

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    save_band_grid_multi(
        plan=X[:T],
        tube=tube_sizes[:T],
        actuals=actuals_for_plot,
        dt=float(DT),
        idx=idx,
        out_path=OUT_PATH,
        ncols=int(NCOLS),
    )
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
