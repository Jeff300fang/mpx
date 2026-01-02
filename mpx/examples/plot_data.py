import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------
# Raw data: (x, y, z)
# Missing values are represented as np.nan
# ------------------------------------------------------------
data = np.array([
    [10, 10,   3.00],
    [10, 25,   3.87],
    [10, 50,   4.40],
    [10, 100,  5.00],
    [10, 500,  5.85],
    [10, 1000, 4.66],
    [10, 2000, 7.93],
    [10, 3000, 9.9],

    [25, 10,   4.07],
    [25, 25,   4.4],
    [25, 50,   6.29],
    [25, 100,  8.8],
    [25, 500,  np.nan],
    [25, 1000, np.nan],
    [25, 2000, np.nan],
    [25, 3000, np.nan],

    [50, 10,   4.08],
    [50, 25,   5.44],
    [50, 50,   6],
    [50, 100,  7],
    [50, 500,  12.05],
    [50, 1000, 14.89],
    [50, 2000, 21.6],
    [50, 3000, 29.2],

    [75, 10,   4.88],
    [75, 25,   6.24],
    [75, 50,   7.05],
    [75, 100,  9.3],
    [75, 500,  18.07],
    [75, 1000, np.nan],
    [75, 2000, np.nan],
    [75, 3000, np.nan],
])

# ------------------------------------------------------------
# Filter out rows with missing z-values
# ------------------------------------------------------------
mask = ~np.isnan(data[:, 2])
x = data[mask, 0]
y = data[mask, 1]
z = data[mask, 2]

# ------------------------------------------------------------
# Create 3D scatter plot
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")

ax.scatter(x, y, z, s=70)

ax.set_xlabel("X")

ax.invert_xaxis() 
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter (Observed Data Only)")

plt.tight_layout()

# ------------------------------------------------------------
# Save to file
# ------------------------------------------------------------
output_path = "3d_scatter.png"
plt.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Saved plot to {output_path}")
