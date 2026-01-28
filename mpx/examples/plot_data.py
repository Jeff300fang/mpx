import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------------
# Matplotlib styling (PDF vector-safe)
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
# Log-like tick helpers for 3D
# -----------------------------
def pow10_major_formatter(val, pos=None):
    if np.isfinite(val) and abs(val - round(val)) < 1e-6:
        return rf"$10^{{{int(round(val))}}}$"
    return ""

def set_loglike_ticks_3d(ax, minor=True):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))

    ax.xaxis.set_major_formatter(FuncFormatter(pow10_major_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(pow10_major_formatter))
    ax.zaxis.set_major_formatter(FuncFormatter(pow10_major_formatter))

    if not minor:
        return

    def minor_ticks(lim):
        lo, hi = lim
        kmin = int(np.floor(lo))
        kmax = int(np.ceil(hi))
        ticks = []
        for k in range(kmin, kmax + 1):
            for m in range(2, 10):
                t = k + np.log10(m)
                if lo <= t <= hi:
                    ticks.append(t)
        return ticks

    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks(sorted(ax.get_xlim()))))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks(sorted(ax.get_ylim()))))
    ax.zaxis.set_minor_locator(FixedLocator(minor_ticks(sorted(ax.get_zlim()))))

    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.zaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

# -----------------------------
# 3D spacing helpers
# -----------------------------
def style_3d_axes(ax, title, xlabel, ylabel, zlabel):
    ax.set_xlabel(xlabel, labelpad=22)
    ax.set_ylabel(ylabel, labelpad=18)
    ax.set_zlabel(zlabel, labelpad=18)

    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)
    ax.tick_params(axis="z", pad=6)

    ax.margins(x=0.08, y=0.08, z=0.08)

def save_pdf(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.5)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
data = np.array([
    [10, 10,   3.00], [10, 25,   3.87], [10, 50,   4.40], [10, 100,  5.00],
    [10, 500,  5.85], [10, 1000, 4.66], [10, 2000, 7.93], [10, 3000, 9.9],

    [25, 10,   4.07], [25, 25,   4.4], [25, 50,   6.29], [25, 100,  8.8],
    [25, 500,  9.3], [25, 1000, 10.79], [25, 2000, 11.7], [25, 3000, 15.8],

    [50, 10,   4.08], [50, 25,   5.44], [50, 50,   6], [50, 100,  7],
    [50, 500,  12.05], [50, 1000, 14.89], [50, 2000, 21.6], [50, 3000, 29.2],

    [75, 10,   4.88], [75, 25,   6.24], [75, 50,   7.05], [75, 100,  9.3],
    [75, 500,  18.07], [75, 1000, 30.89], [75, 2000, 48], [75, 3000, 72.98],

    [100, 10,   5.3], [100, 25,   8.23], [100, 50,   11.89],
    [100, 100,  16.5], [100, 500,  40.95], [100, 1000, 72.97], [100, 2000, 138],
])

x, y, z = data[:, 0], data[:, 1], data[:, 2]
xL, yL, zL = np.log10(x), np.log10(y), np.log10(z)

# ------------------------------------------------------------
# 1) Scatter plot (vector PDF)
# ------------------------------------------------------------
fig = plt.figure(figsize=(9.5, 4.5))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25, azim=40)

ax.scatter(xL, yL, zL, s=70)
ax.invert_xaxis()

style_3d_axes(
    ax,
    "3D Scatter (Log–Log–Log)",
    "Decision Variables",
    "Horizon",
    "Solve Time (ms)",
)

set_loglike_ticks_3d(ax)
save_pdf(fig, "3d_scatter_loglike.pdf")
plt.close(fig)

# ------------------------------------------------------------
# Polynomial regression in log space
# ------------------------------------------------------------
X_log = np.column_stack([xL, yL])
poly = PolynomialFeatures(degree=2, include_bias=True)
model = LinearRegression().fit(poly.fit_transform(X_log), zL)

XiL, YiL = np.meshgrid(
    np.linspace(xL.min(), xL.max(), 60),
    np.linspace(yL.min(), yL.max(), 60),
)

ZiL = model.predict(
    poly.transform(np.column_stack([XiL.ravel(), YiL.ravel()]))
).reshape(XiL.shape)

# ------------------------------------------------------------
# 2) Surface plot (vector PDF)
# ------------------------------------------------------------
fig = plt.figure(figsize=(9.0, 9.0))
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25, azim=20)

ax.plot_surface(XiL, YiL, ZiL, alpha=0.6, linewidth=0)
ax.scatter(xL, yL, zL, color="k", s=50)
ax.invert_xaxis()

style_3d_axes(
    ax,
    "Solve Time vs Horizon vs Decision Variables",
    "Decision Variables",
    "Horizon",
    "Solve Time (ms)",
)

set_loglike_ticks_3d(ax)
save_pdf(fig, "3d_polynomial_surface_loglike.pdf")
plt.close(fig)

print("Saved vector PDFs:")
print("  - 3d_scatter_loglike.pdf")
print("  - 3d_polynomial_surface_loglike.pdf")
