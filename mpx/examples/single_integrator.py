from __future__ import annotations
from jax import config
config.update("jax_enable_x64", True)
from dataclasses import dataclass
from typing import Any, Callable

import time
import jax
import jax.numpy as jnp

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper
from mpx.utils.mpc_utils import combine_constraints
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray


def make_control_box_constraints(u_min: jnp.ndarray, u_max: jnp.ndarray) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for control bounds:
      u - u_max <= 0
      u_min - u <= 0
    """
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints

def make_circle_constraints(xc, yc, r):
    def constraint(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([r**2 - (x[0] - xc)**2 - (x[1] - yc)**2])
    
    return constraint

def make_ellipsoid_constraints(
    xc: float,
    yc: float,
    a: float,
    b: float,
    *,
    theta: float = 0.0,
):
    """
    Inequality constraint g(x) <= 0 representing OUTSIDE an ellipsoid.

    Axis-aligned (theta=0):
        g(x) = 1 - ((x-xc)/a)^2 - ((y-yc)/b)^2

    If theta != 0, we rotate coordinates by theta (radians) before evaluating.
    """
    a2 = a * a
    b2 = b * b
    cth = jnp.cos(theta)
    sth = jnp.sin(theta)

    def constraint(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        dx = x[0] - xc
        dy = x[1] - yc

        # rotate into ellipsoid frame (if theta=0 this is identity)
        xrp =  cth * dx + sth * dy
        yrp = -sth * dx + cth * dy

        val = 1.0 - (xrp * xrp) / a2 - (yrp * yrp) / b2
        return jnp.array([val], dtype=x.dtype)

    return constraint

def dynamics(x, u, t, *, parameter):
    dt = parameter
    px, py = x
    vx, vy = u

    px_next = px + dt * vx
    py_next = py + dt * vy

    return jnp.array([px_next, py_next], dtype=x.dtype)

def cost(W, reference, x, u, t):
    wx, wy, wvx, wvy = W
    xref = reference[t]

    dx = x[0] - xref[0]
    dy = x[1] - xref[1]

    vx, vy = u

    return (
        wx * dx * dx
        + wy * dy * dy
        + wvx * vx * vx
        + wvy * vy * vy
    )

def make_constant_disturbance(
    n: int,
    alpha: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, nc),
    where E[t] = alpha * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)  # (n, nc)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance

# Define Cost
W = jnp.array([30.0, 30.0, 1.0, 1.0])

nx = 2
nu = 2
N = 110
dt = 0.1

cfg = MPCConfig(
    n=nx,
    nu=nu,
    N=N,
    W=W,
    u_ref=jnp.zeros((nu,)),
)

admm_cfg = ADMMConfig(
    eps_abs=1e-3,
    eps_rel=0,
    rho_max=1e2,
    max_iterations=2000,
)

sls_cfg = SLSConfig(
    max_sls_iterations=3,
    sls_primal_tol=1e-2,
    enable_fastsls=False,
    warm_start=False,
)

sqp_cfg = SQPConfig(
    max_sqp_iterations = 50,
    warm_start=True,
    line_search=True,
)

# Define Constraints
obstacles = jnp.array([])
u_min = jnp.array([-1.0, -1.0])
u_max = jnp.array([1.0, 1.0])
constraints_u = make_control_box_constraints(u_min, u_max)
circle_constraint = make_circle_constraints(0, 0, 0.3)

ellipsoid_constraint = make_ellipsoid_constraints(
    xc=0.0, yc=0.0,
    a=0.45, b=0.25,
    theta=jnp.deg2rad(90.0),
)
# constraints_all = combine_constraints(constraints_u, circle_constraint)
constraints_all = combine_constraints(constraints_u, ellipsoid_constraint)
# circle_constraint = make_circle_constraints(0.0, 0, 0.3)
# constraints_all = combine_constraints(constraints_u, ellipsoid_constraint)
nc = 2 * nu + 1

E_mag = 0.075
alpha_sim = E_mag * dt
disturbance = make_constant_disturbance(n=nx, alpha=alpha_sim)

x0 = jnp.array([-0.75, -0.75])
parameter = dt
x_goal = jnp.array([0.7, 1.0]) 
X_ref = jnp.tile(x_goal[None, :], (N + 1, 1))  # shape (N+1, nx)
def make_straight_line_reference(
    x0: jnp.ndarray,
    x_goal: jnp.ndarray,
    N: int,
    dt: float,
    v_des: float,
) -> jnp.ndarray:
    """
    Straight-line reference from x0 to x_goal at constant speed v_des (>=0).
    Returns X_ref with shape (N+1, 2). Once goal is reached, it stays there.
    """
    x0 = jnp.asarray(x0)
    x_goal = jnp.asarray(x_goal)

    d = x_goal - x0
    dist = jnp.linalg.norm(d)

    # Unit direction (safe for dist=0)
    dir_vec = jnp.where(dist > 1e-12, d / dist, jnp.zeros_like(d))

    # How far along the line we should be at each time step
    t = jnp.arange(N + 1, dtype=x0.dtype) * dt
    s = jnp.minimum(v_des * t, dist)  # clip so we don't pass the goal

    # Position reference
    X_ref = x0[None, :] + s[:, None] * dir_vec[None, :]
    return X_ref

v_des = 0.6  # <--- desired speed (position units / second)
X_ref = make_straight_line_reference(x0=x0, x_goal=x_goal, N=N, dt=dt, v_des=v_des)
reference = X_ref

# Nominal init
def project_strictly_outside_ellipsoid(
    X: jnp.ndarray,  # (T,2)
    *,
    xc: float,
    yc: float,
    a: float,
    b: float,
    theta: float = 0.0,
    eps: float = 1e-3,          # strict margin: s >= 1+eps
    keep_endpoints: bool = True # keep X[0], X[-1] unchanged if desired
) -> jnp.ndarray:
    """
    Projects points that violate s(x) >= 1+eps to the (1+eps)-level set of the ellipsoid,
    by scaling in the rotated ellipsoid frame (radial projection in that metric).
    """
    a2 = a * a
    b2 = b * b
    cth = jnp.cos(theta)
    sth = jnp.sin(theta)

    def proj_one(x):
        dx = x[0] - xc
        dy = x[1] - yc

        # rotate into ellipsoid frame
        xrp =  cth * dx + sth * dy
        yrp = -sth * dx + cth * dy

        s = (xrp * xrp) / a2 + (yrp * yrp) / b2  # want >= 1+eps

        # scale only if inside/too close
        s_safe = jnp.maximum(s, 1e-12)
        scale = jnp.sqrt((1.0 + eps) / s_safe)
        do_scale = s < (1.0 + eps)

        xrp2 = jnp.where(do_scale, xrp * scale, xrp)
        yrp2 = jnp.where(do_scale, yrp * scale, yrp)

        # rotate back
        dx2 =  cth * xrp2 - sth * yrp2
        dy2 =  sth * xrp2 + cth * yrp2
        return jnp.array([xc + dx2, yc + dy2], dtype=x.dtype)

    Xp = jax.vmap(proj_one)(X)

    if keep_endpoints and X.shape[0] >= 2:
        Xp = Xp.at[0].set(X[0])
        Xp = Xp.at[-1].set(X[-1])
    return Xp
theta = float(jnp.deg2rad(90.0))
X_in = project_strictly_outside_ellipsoid(
    X_ref,
    xc=0.0, yc=0.0,
    a=0.45, b=0.25,
    theta=theta,
    eps=1e-3,              # increase if you want more clearance
    keep_endpoints=False,  # set True if you insist on exact endpoints
)

controller = GenericMPCControllerWrapper(
    sls_cfg,
    sqp_cfg,
    admm_cfg,
    config=cfg,
    dynamics=dynamics,
    constraints=constraints_all,
    obstacles=obstacles,
    cost=cost,
    num_constraints=nc,
    disturbance=disturbance,
    limited_memory=False,
    shift=1,
    X_in=X_in.copy(),
    U_in=jnp.zeros((cfg.N, cfg.nu))
)
for i in range(1):
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(x0=x0, reference=reference, parameter=parameter)

def ell_val(x):
    dx = x[0] - 0.0
    dy = x[1] - 0.0
    # theta = 90deg
    cth, sth = np.cos(np.deg2rad(90.0)), np.sin(np.deg2rad(90.0))
    xrp =  cth * dx + sth * dy
    yrp = -sth * dx + cth * dy
    return 1.0 - (xrp*xrp)/(0.45**2) - (yrp*yrp)/(0.25**2)

vals = np.array([ell_val(np.asarray(X_pred[t])) for t in range(X_pred.shape[0])])
print("max constraint value (should be <=0):", vals.max())
print("min constraint value:", vals.min())
print("indices where active-ish:", np.where(vals > -1e-2)[0][:20])


# START VISUALIZATION
def plot_ref_and_shape(
    reference: jnp.ndarray,
    *,
    shape: str = "circle",   # "circle" or "ellipsoid"
    xc: float = 0.0,
    yc: float = 0.0,
    r: float = 0.3,
    a: float = 0.45,
    b: float = 0.25,
    theta: float = 0.0,      # radians
    x0: jnp.ndarray | None = None,
    X_pred: jnp.ndarray | None = None,
    title: str = "Reference + constraint",
):
    ref = np.asarray(reference)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(ref[:, 0], ref[:, 1], "--", linewidth=2, label="reference")

    if X_pred is not None:
        Xp = np.asarray(X_pred)
        ax.plot(Xp[:, 0], Xp[:, 1], "-", linewidth=2, label="X_pred")

    if x0 is not None:
        x0n = np.asarray(x0)
        ax.scatter([x0n[0]], [x0n[1]], s=60, marker="o", label="start")

    ax.scatter([ref[-1, 0]], [ref[-1, 1]], s=60, marker="*", label="goal")

    if shape == "circle":
        boundary = plt.Circle((xc, yc), r, fill=False, linewidth=2, label="circle boundary")
        infeas = plt.Circle((xc, yc), r, fill=True, alpha=0.15)
        ax.add_patch(boundary)
        ax.add_patch(infeas)
        shape_extent_x = (xc - r, xc + r)
        shape_extent_y = (yc - r, yc + r)

    elif shape == "ellipsoid":
        from matplotlib.patches import Ellipse
        boundary = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.rad2deg(theta),
                           fill=False, linewidth=2, label="ellipsoid boundary")
        infeas = Ellipse((xc, yc), width=2*a, height=2*b, angle=np.rad2deg(theta),
                         fill=True, alpha=0.15)
        ax.add_patch(boundary)
        ax.add_patch(infeas)
        # conservative bounds for view (ignores rotation; good enough for padding)
        shape_extent_x = (xc - a, xc + a)
        shape_extent_y = (yc - b, yc + b)

    else:
        raise ValueError(f"Unknown shape={shape}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend()

    xs = [ref[:, 0]]
    ys = [ref[:, 1]]
    if X_pred is not None:
        xs.append(np.asarray(X_pred)[:, 0])
        ys.append(np.asarray(X_pred)[:, 1])
    if x0 is not None:
        xs.append(np.asarray(x0)[0:1])
        ys.append(np.asarray(x0)[1:2])

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    pad = 0.25
    ax.set_xlim(min(x_all.min(), shape_extent_x[0]) - pad, max(x_all.max(), shape_extent_x[1]) + pad)
    ax.set_ylim(min(y_all.min(), shape_extent_y[0]) - pad, max(y_all.max(), shape_extent_y[1]) + pad)

    plt.show()
    return fig, ax
# --- call it after controller.run ---
USE_ELLIPSOID = True
# USE_ELLIPSOID = False
# plot_ref_and_shape(
#     reference,
#     shape="ellipsoid" if USE_ELLIPSOID else "circle",
#     xc=0.0, yc=0.0,
#     r=0.3,
#     a=0.45, b=0.25, theta=float(jnp.deg2rad(90.0)),
#     x0=x0,
#     X_pred=X_pred,
#     title="Reference + outside-shape constraint",
# )
def plot_three(reference, X_in, X_pred, *, title="Trajectories + constraint"):
    ref = np.asarray(reference)
    Xin = np.asarray(X_in)
    Xp  = np.asarray(X_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ref[:, 0], ref[:, 1], "--", linewidth=2, label="reference")
    ax.plot(Xin[:, 0], Xin[:, 1], "--", linewidth=2, label="X_in (projected)")
    ax.plot(Xp[:, 0],  Xp[:, 1],  "-",  linewidth=2, label="X_pred (MPC)")

    from matplotlib.patches import Ellipse
    boundary = Ellipse((0.0, 0.0), width=2*0.45, height=2*0.25,
                       angle=np.rad2deg(np.deg2rad(90.0)),
                       fill=False, linewidth=2, label="ellipsoid boundary")
    infeas = Ellipse((0.0, 0.0), width=2*0.45, height=2*0.25,
                     angle=np.rad2deg(np.deg2rad(90.0)),
                     fill=True, alpha=0.15)
    ax.add_patch(boundary)
    ax.add_patch(infeas)

    ax.scatter([x0[0]], [x0[1]], s=60, marker="o", label="start")
    ax.scatter([ref[-1, 0]], [ref[-1, 1]], s=60, marker="*", label="goal")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return fig, ax
plot_three(reference, X_in, X_pred, title="reference vs X_in vs X_pred + ellipsoid")