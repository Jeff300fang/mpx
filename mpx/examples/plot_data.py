import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import FuncFormatter, MaxNLocator

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
plt.rcParams.update({
    "font.size": 16,          # base font size
})


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",   # used if available
        "Times",             # macOS fallback
        "Nimbus Roman",      # Linux fallback
    ],
    "mathtext.fontset": "stix",
})

# -----------------------------
# Helpers for "log-looking" ticks
# -----------------------------
def pow10_formatter(val, pos=None):
    # val is log10(value)
    if np.isfinite(val) and abs(val - round(val)) < 1e-6:
        return r"$10^{%d}$" % int(round(val))
    return ""


def set_loglike_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(pow10_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(pow10_formatter))
    ax.zaxis.set_major_formatter(FuncFormatter(pow10_formatter))


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
# Filter out missing z-values
# ------------------------------------------------------------
mask = ~np.isnan(data[:, 2])
x = data[mask, 0].astype(float)
y = data[mask, 1].astype(float)
z = data[mask, 2].astype(float)

# log axes require strictly positive values
if np.any(x <= 0) or np.any(y <= 0) or np.any(z <= 0):
    raise ValueError("Log plots require x, y, z to be strictly positive.")

# ------------------------------------------------------------
# Work in log10 coordinates FOR PLOTTING (robust in 3D)
# ------------------------------------------------------------
xL = np.log10(x)
yL = np.log10(y)
zL = np.log10(z)

# ------------------------------------------------------------
# 1) 3D scatter plot in log coordinates (no set_*scale('log'))
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25, azim=40)

ax.scatter(xL, yL, zL, s=70)

ax.invert_xaxis()
ax.set_xlabel("Number of Decision Variables")
ax.set_ylabel("Horizon")
ax.set_zlabel("Solve Time (ms)")
ax.set_title("3D Scatter (Log–Log–Log, Stable Rendering)")

set_loglike_ticks(ax)

plt.tight_layout()
plt.savefig("3d_scatter_loglike.png", dpi=300)
plt.close(fig)

print("Saved plot to 3d_scatter_loglike.png")

# ------------------------------------------------------------
# 2) Polynomial regression in log space:
#    zL = poly(xL, yL)
# ------------------------------------------------------------
X_log = np.column_stack([xL, yL])

degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=True)
X_log_poly = poly.fit_transform(X_log)

model = LinearRegression().fit(X_log_poly, zL)

# ------------------------------------------------------------
# 3) Evaluate model on a grid IN LOG SPACE (important!)
#    This avoids warped surfaces / folding artifacts
# ------------------------------------------------------------
XiL, YiL = np.meshgrid(
    np.linspace(xL.min(), xL.max(), 60),
    np.linspace(yL.min(), yL.max(), 60),
)

grid_log = np.column_stack([XiL.ravel(), YiL.ravel()])
ZiL = model.predict(poly.transform(grid_log)).reshape(XiL.shape)

# OPTIONAL SAFETY: If you see folds due to the polynomial itself, clip the
# predicted z range to something sane (comment out if you dislike clipping).
# ZiL = np.clip(ZiL, zL.min() - 0.25, zL.max() + 0.25)

# ------------------------------------------------------------
# 4) Plot surface in log coordinates (stable) + data points
# ------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25, azim=20)

ax.plot_surface(
    XiL, YiL, ZiL,
    alpha=0.6,
    linewidth=0,
    antialiased=True
)

ax.scatter(xL, yL, zL, color="k", s=50)

ax.invert_xaxis()
ax.set_xlabel("Number of Decision Variables")
ax.set_ylabel("Horizon")
ax.set_zlabel("Solve Time (ms)")
ax.set_title(
    "Solve Time vs Horizon vs Number of Decision Variables",
    pad=0   # try values in [4, 12]
)

set_loglike_ticks(ax)

plt.tight_layout()
plt.savefig("3d_polynomial_surface_loglike.png", dpi=300)
plt.close(fig)

print("Saved plot to 3d_polynomial_surface_loglike.png")
