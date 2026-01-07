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
    [25, 500,  9.3],
    [25, 1000, 10.79],
    [25, 2000, 11.7],
    [25, 3000, 15.8],

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
    [75, 1000, 30.89],
    [75, 2000, 48],
    [75, 3000, 72.98],

    [100, 10,   5.3],
    [100, 25,   8.23],
    [100, 50,   11.89],
    [100, 100,  16.5],
    [100, 500,  40.95],
    [100, 1000, 72.97],
    [100, 2000, 138],

])

# ------------------------------------------------------------
# Filter out rows with missing z-values
# ------------------------------------------------------------
mask = ~np.isnan(data[:, 2])
x = data[mask, 0]
y = data[mask, 1]
z = data[mask, 2]
x = np.log10(x)
y = np.log10(y)
z = np.log10(z)

# ------------------------------------------------------------
# Create 3D scatter plot
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25, azim=40)
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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

XY = np.column_stack([x, y])
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=True)
XY_poly = poly.fit_transform(XY)

model = LinearRegression().fit(XY_poly, z)

# Evaluate on grid
Xi, Yi = np.meshgrid(
    np.linspace(x.min(), x.max(), 50),
    np.linspace(y.min(), y.max(), 50)
)
XYi = np.column_stack([Xi.ravel(), Yi.ravel()])
Zi = model.predict(poly.transform(XYi)).reshape(Xi.shape)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")

# Surface
ax.plot_surface(
    Xi, Yi, Zi,
    alpha=0.6,
    linewidth=0,
    antialiased=True
)

# Original data
ax.scatter(x, y, z, color="k", s=50)
ax.view_init(elev=25, azim=20)
ax.invert_xaxis()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Polynomial Surface Fit (degree={degree})")

plt.tight_layout()
plt.savefig("3d_polynomial_surface.png", dpi=300)
plt.close(fig)