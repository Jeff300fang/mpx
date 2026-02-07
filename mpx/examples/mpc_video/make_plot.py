#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mpx.utils.fast_sls_visual import get_trajectory_tubes

# -----------------------------------------------------------------------------
# Matplotlib style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": False,
})
plt.rcParams.update({
    "font.size": 18,
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
FILE_NAME = "h1_mpc_rollout_log.npz"
OUT_PDF = "h1_overlay_tubes_every60.pdf"

# Obstacles (x, y, r)
obstacles = jnp.array([
    [2.0,  0.2,  0.43],
    [1.7, -1.1,  0.43],
])

# Which MPC solves to overlay (in *time*, i over N)
SOLVE_STRIDE = 60
START_SOLVE = 0

# Opacity schedule across solve index k (k=0 => 0.40, ..., k=6 => 1.00)
ALPHAS = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]

# Draw fewer horizon boxes if you want speed; set to 1 to draw every t
HORIZON_STRIDE = 2

PAD = 0.5

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
data = np.load(FILE_NAME)
X_list = data["X"]          # (N, T, nx)
Phi_x_list = data["Phi_x"]  # (N, T, nx, nx) or similar
N, T, nx = X_list.shape
actual_xy = X_list[:, 0, :2] 
# -----------------------------------------------------------------------------
# Select which solves to overlay
# -----------------------------------------------------------------------------
solve_idxs = list(range(START_SOLVE, N, SOLVE_STRIDE))
if len(solve_idxs) == 0:
    raise RuntimeError("No solve indices selected. Check START_SOLVE / SOLVE_STRIDE / N.")

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

# Obstacles
for (ox, oy, r) in np.asarray(obstacles):
    ax.add_patch(Circle((ox, oy), r, facecolor="tab:red", edgecolor="tab:red",
                        alpha=0.35, zorder=1))
ax.plot(
    actual_xy[:, 0],
    actual_xy[:, 1],
    "-",
    color="tab:orange",
    linewidth=3.0,
    label="Executed",
    zorder=5,
)
# -----------------------------------------------------------------------------
# Plot: for each selected solve i, draw ALL tube boxes along its horizon
# -----------------------------------------------------------------------------
# We’ll also accumulate extents so axis limits definitely include everything.
mins = []
maxs = []

for k, i in enumerate(solve_idxs):
    alpha = ALPHAS[k] if k < len(ALPHAS) else ALPHAS[-1]

    X = X_list[i]                         # (T, nx)
    plan_xy = np.asarray(X[:, :2])        # (T, 2)
    tubes = np.asarray(get_trajectory_tubes(Phi_x_list[i]))  # (T, nx)
    tubes_xy = tubes[:, :2]

    lower_xy = plan_xy - tubes_xy
    upper_xy = plan_xy + tubes_xy

    # Optionally plot the plan itself (light, behind tubes)
    ax.plot(plan_xy[:, 0], plan_xy[:, 1],
            linestyle="--", linewidth=2.0, color="tab:orange",
            alpha=min(0.25, alpha), zorder=2)

    # Draw ALL horizon boxes for this solve
    for t in range(0, T, HORIZON_STRIDE):
        x0, y0 = lower_xy[t]
        x1, y1 = upper_xy[t]
        w = float(max(0.0, x1 - x0))
        h = float(max(0.0, y1 - y0))

        ax.add_patch(
            Rectangle((float(x0), float(y0)), w, h,
                      facecolor="tab:blue", edgecolor="tab:blue",
                      linewidth=0.8, alpha=alpha, zorder=3)
        )

    mins.append(np.nanmin(lower_xy, axis=0))
    maxs.append(np.nanmax(upper_xy, axis=0))

# -----------------------------------------------------------------------------
# Axis limits: include all overlaid tube envelopes + obstacles
# -----------------------------------------------------------------------------
mins = np.vstack(mins)
maxs = np.vstack(maxs)
xy_min = np.nanmin(mins, axis=0)
xy_max = np.nanmax(maxs, axis=0)

obs_xy = np.asarray(obstacles)[:, :2]
obs_r  = np.asarray(obstacles)[:, 2:3]
obs_min = np.min(obs_xy - obs_r, axis=0)
obs_max = np.max(obs_xy + obs_r, axis=0)

xmin = min(float(xy_min[0]), float(obs_min[0])) - PAD
xmax = max(float(xy_max[0]), float(obs_max[0])) + PAD
ymin = min(float(xy_min[1]), float(obs_min[1])) - PAD
ymax = max(float(xy_max[1]), float(obs_max[1])) + PAD

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$p_x$")
ax.set_ylabel(r"$p_y$")

# -----------------------------------------------------------------------------
# Legend (bottom, 2 rows, bordered)
# -----------------------------------------------------------------------------
legend_handles = [
    Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.60,
          label=f"Robust tubes"),
    Line2D([0], [0], color="tab:orange", linewidth=2.0, linestyle="--",
           label="Planned Trajectory"),
    Patch(facecolor="tab:red", edgecolor="tab:red", alpha=0.35,
          label="Obstacle"),
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=True,
    framealpha=1.0,
    fancybox=False,
    handlelength=2.2,
    columnspacing=1.6,
    labelspacing=0.9,
)
plt.subplots_adjust(bottom=0.26)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved: {OUT_PDF}")
