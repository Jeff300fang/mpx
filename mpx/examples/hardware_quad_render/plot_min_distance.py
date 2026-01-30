import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# -----------------------------
# Matplotlib styling (paper-ready)
# -----------------------------
plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,   # embed fonts (Type 42)
    "ps.fonttype": 42,
})

# -----------------------------
# Load data
# -----------------------------
min_distance = np.load("minimum_distance_np.npy")

dt = 0.01
t = np.arange(len(min_distance)) * dt

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(t, min_distance, label="Min distance to obstacle")
plt.axhline(0.6, linestyle="--", linewidth=2, label="Obstacle radius (0.6 m)")

ax = plt.gca()

# X-axis ticks and limits
ax.xaxis.set_major_locator(MultipleLocator(2.0))
ax.set_xlim(t.min(), t.max())

# Major grid lines on both axes
ax.grid(True, axis="y")
ax.grid(True, axis="x")

plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
plt.legend()
plt.tight_layout()

# -----------------------------
# Save to PDF
# -----------------------------
plt.savefig("minimum_distance_vs_time.pdf", bbox_inches="tight")

# Optional: also display interactively
# plt.show()
