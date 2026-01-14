import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
horizon = np.array([10, 25, 50, 75, 100, 200, 500, 1000, 2000, 3000])

plt.rcParams.update({
    "font.size": 19,          # base font size
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

data = {
    "Our Method":                 [2.854, 3.36, 3.64, 4.94, 6.06, 8.5, 11.1, 17.2, 26.78, 38.1],
    "ADMM Float64":               [4.11, 5.03, 5.97, 7.72, 10.23, 14.99, 29.07, 51.07, 93.3, 140.5],
    "acados (HPIPM)":    [0.773, 1.46, 2.6, 5.19, 6.72, 12.08, 38.02, 127.8, 583, 1716],
    "acados (OSQP)":     [3.08, 4.57, 9.88, 13.31, 18.96, 45.6, 305, 1777, 12474, np.nan],
    "primal-dual iLQR AL":        [5.26, 6.14, 8.75, 11.326, 15.49, 20.1, 33.26, 60.9, np.nan, np.nan],
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(9, 6))

for label, values in data.items():
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    plt.plot(
        horizon[mask],
        y[mask],
        marker="o",
        linewidth=2,
        markersize=5,
        label=label,
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Horizon Length")
plt.ylabel("Solve Time (ms)")
plt.title("Solver Scaling vs Horizon Length (Log Scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# -----------------------------
# Save / Show
# -----------------------------
plt.savefig("solver_scaling_log.png", dpi=300, bbox_inches="tight")
plt.show()
