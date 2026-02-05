#!/usr/bin/env python3
"""
render_legend_only.py

Render a standalone legend for Dubins MPC plots (no data, legend only).

Outputs:
  - legend.png
  - legend.pdf   (vector, recommended for papers)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

# -----------------------------
# Styling (match your main script)
# -----------------------------
plt.rcParams.update({
    "font.size": 25,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,   # embed fonts in PDF
    "ps.fonttype": 42,
})
# -----------------------------
# Legend handles
# -----------------------------
handles = [
    Line2D(
        [], [], color="tab:orange", lw=2,
        label="Executed (closed-loop)"
    ),
    Line2D(
        [], [], color="tab:orange", lw=2, ls="--",
        label="Planned (open-loop)"
    ),
    Line2D(
        [], [], color="tab:blue", marker="o", linestyle="None",
        markersize=7, label="Current state"
    ),
    Line2D(
        [], [], color="tab:orange", marker="x", linestyle="None",
        markersize=8, label="End of plan"
    ),
    Patch(
        facecolor="tab:blue", alpha=0.20,
        label="Robust tube (state uncertainty)"
    ),
    Patch(
        facecolor="tab:red", alpha=0.35,
        label="Obstacle"
    ),
]

# -----------------------------
# Figure (legend-only canvas)
# -----------------------------
fig = plt.figure(figsize=(6.0, 1.6))
ax = fig.add_subplot(111)
ax.axis("off")

legend = ax.legend(
    handles=handles,
    loc="center",
    ncol=3,              # adjust: 2–4 depending on slide/paper
    frameon=True,
    framealpha=0.95,
    edgecolor="black",
    handlelength=2.2,
    columnspacing=1.5,
)

# -----------------------------
# Save
# -----------------------------
fig.savefig("legend.png", dpi=300, bbox_inches="tight")
fig.savefig("legend.pdf", bbox_inches="tight")
plt.close(fig)

print("[Saved] legend.png, legend.pdf")
