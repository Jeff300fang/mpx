import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter

# -----------------------------
# User config
# -----------------------------
DATA_PATH = "minimum_distance_np.npy"
OUT_MP4 = "minimum_distance_vs_time_anim.mp4"

dt_data = 0.02           # for x-axis time units
ANIM_DURATION_S = 27.05   # <-- total animation duration in seconds (set this)

# -----------------------------
# Matplotlib styling (paper-ready)
# -----------------------------
plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------
# Load data
# -----------------------------
min_distance = np.load(DATA_PATH).astype(float)
n = len(min_distance)
t = np.arange(n) * dt_data

# If you have N samples, there are (N-1) time intervals between them.
interval_ms = 1000.0 * ANIM_DURATION_S / max(n - 1, 1)
fps = 1000.0 / interval_ms if interval_ms > 0 else 30.0

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))

(line,) = ax.plot([], [], linewidth=2, label="Min distance to obstacle")
(vline,) = ax.plot([0, 0], [0, 1], linewidth=2)  # current-time cursor
ax.axhline(0.6, linestyle="--", linewidth=2, label="Obstacle radius (0.6 m)")

ax.xaxis.set_major_locator(MultipleLocator(5.0))
ax.set_xlim(t.min(), t.max())

y_min = min(min_distance.min(), 0.6) - 0.05
y_max = max(min_distance.max(), 0.6) + 0.05
ax.set_ylim(y_min, y_max)

ax.grid(True, axis="y")
ax.grid(True, axis="x")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance [m]")
ax.legend()
fig.tight_layout()

# -----------------------------
# Animation
# -----------------------------
def init():
    line.set_data([], [])
    vline.set_data([t[0], t[0]], [y_min, y_max])
    return line, vline

def update(i):
    line.set_data(t[: i + 1], min_distance[: i + 1])
    vline.set_data([t[i], t[i]], [y_min, y_max])
    return line, vline

anim = FuncAnimation(
    fig,
    update,
    frames=n,
    init_func=init,
    interval=interval_ms,  # <-- sets real-time pacing
    blit=True,
    repeat=False,
)

# Save
writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=5000)
anim.save(OUT_MP4, writer=writer)
print(f"Saved: {OUT_MP4} (duration ~ {ANIM_DURATION_S:.2f}s, fps ~ {fps:.2f})")
