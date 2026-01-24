"""
binary_search_max_u.py

Binary-search MAX_U so that "near constraint %" is within [23%, 27%].

Assumptions:
- near_constraint% is (weakly) monotone DECREASING in MAX_U (larger bounds -> fewer saturations).
- Uses your same model/controller structure; we rebuild the controller each eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple
import time

import jax
import jax.numpy as jnp
from jax import config

# Keep this consistent with your experiment
config.update("jax_enable_x64", False)

from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper


# -----------------------------
# Angle wrapping
# -----------------------------
def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


# -----------------------------
# N-link planar pendulum dynamics (Lagrangian via AD)
# State: x = [q; qd], input: u = tau
# -----------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class NLinkParams:
    l: jnp.ndarray
    m: jnp.ndarray
    I: jnp.ndarray
    lc: jnp.ndarray
    g: float = 9.81
    b: float = 0.0

    def tree_flatten(self):
        children = (self.l, self.m, self.I, self.lc, jnp.asarray(self.g), jnp.asarray(self.b))
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        l, m, I, lc, g, b = children
        return cls(l=l, m=m, I=I, lc=lc, g=g, b=b)


def _com_positions(q: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    theta = jnp.cumsum(q)
    ex, ey = jnp.cos(theta), jnp.sin(theta)

    dx_full = p.l * ex
    dy_full = p.l * ey

    xj = jnp.concatenate([jnp.zeros((1,), dtype=q.dtype), jnp.cumsum(dx_full)])
    yj = jnp.concatenate([jnp.zeros((1,), dtype=q.dtype), jnp.cumsum(dy_full)])

    xi = xj[:-1] + p.lc * ex
    yi = yj[:-1] + p.lc * ey
    return jnp.stack([xi, yi], axis=-1)  # (n,2)


def _kinetic_energy(q: jnp.ndarray, qd: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    Jr = jax.jacobian(lambda qq: _com_positions(qq, p))(q)  # (n,2,n)
    v = jnp.einsum("i j k, k -> i j", Jr, qd)               # (n,2)
    omega = jnp.cumsum(qd)                                  # (n,)

    T_trans = 0.5 * jnp.sum(p.m * jnp.sum(v * v, axis=-1))
    T_rot = 0.5 * jnp.sum(p.I * (omega * omega))
    return T_trans + T_rot


def _potential_energy(q: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    y = _com_positions(q, p)[:, 1]
    return jnp.sum(p.m * p.g * y)


@jax.jit
def nlink_qdd(q: jnp.ndarray, qd: jnp.ndarray, tau: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    T = lambda qq, qdq: _kinetic_energy(qq, qdq, p)
    V = lambda qq: _potential_energy(qq, p)

    M = jax.jacobian(lambda qdq: jax.grad(lambda qdq2: T(q, qdq2))(qdq))(qd)
    dpdq_dq = jax.jacobian(lambda qq: jax.grad(lambda qdq: T(qq, qdq))(qd))(q)
    coriolis_like = dpdq_dq @ qd

    dT_dq = jax.grad(lambda qq: T(qq, qd))(q)
    dV_dq = jax.grad(V)(q)

    b = jnp.asarray(p.b)
    damping = b * qd

    rhs = tau - damping - coriolis_like + dT_dq - dV_dq
    return jnp.linalg.solve(M, rhs)


@jax.jit
def pendulum_step(x: jnp.ndarray, u: jnp.ndarray, dt: float, p: NLinkParams) -> jnp.ndarray:
    n = u.shape[0]
    q = x[:n]
    qd = x[n:]
    qdd = nlink_qdd(q, qd, u, p)
    xdot = jnp.concatenate([qd, qdd], axis=0)
    return x + dt * xdot


def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dt, p = parameter
    return pendulum_step(x, u, dt, p)


# -----------------------------
# Cost and Constraints matching your mpc() expectations
# -----------------------------
def cost(W: jnp.ndarray, reference: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    wq, wqd, wu = W
    n = u.shape[0]
    xref = reference[t]
    q = x[:n]
    qd = x[n:]
    q_ref = xref[:n]
    qd_ref = xref[n:]
    dq = wrap_to_pi(q - q_ref)
    dqd = qd - qd_ref
    return wq * jnp.sum(dq * dq) + wqd * jnp.sum(dqd * dqd) + wu * jnp.sum(u * u)


def make_torque_box_constraints(u_min: jnp.ndarray, u_max: jnp.ndarray):
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints


def make_zero_disturbance(n: int, nc: int):
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        return jnp.zeros((T, n, nc), dtype=X_prefix.dtype)
    return disturbance


def make_random_periodic_pushes(
    nu: int,
    dt: float,
    period_sec: float = 0.6,
    push_duration_sec: float = 0.08,
    amp_max: float = 2.0,
    per_joint: bool = True,
) -> Callable[[int, jax.Array], jnp.ndarray]:
    period_steps = max(1, int(period_sec / dt))
    dur_steps = max(1, int(push_duration_sec / dt))

    def tau_disturb(k: int, key: jax.Array) -> jnp.ndarray:
        phase = k % period_steps
        on = jnp.where(phase < dur_steps, 1.0, 0.0)

        key, sub = jax.random.split(key)
        if per_joint:
            push = amp_max * (2.0 * jax.random.uniform(sub, (nu,)) - 1.0)
        else:
            s = amp_max * (2.0 * jax.random.uniform(sub, ()) - 1.0)
            push = jnp.ones((nu,)) * s
        return on * push

    return tau_disturb


# -----------------------------
# Config
# -----------------------------
@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray


# -----------------------------
# One evaluation: run rollout and return near-constraint fraction
# -----------------------------
def evaluate_max_u(
    MAX_U: float,
    *,
    nlinks: int = 2,
    N: int = 25,
    dt: float | None = None,
    alpha: float = 0.9,
    seed: int = 0,
    push_period_sec: float = 0.005,
    push_duration_sec: float = 0.1,
    max_amp: float | None = None,
    warmup: bool = True,
) -> Tuple[float, dict]:
    n = 2 * nlinks
    nu = nlinks

    if dt is None:
        dt = min(0.5 / N, 0.005)

    W = jnp.array([50.0, 5.0, 0.1], dtype=jnp.float32)
    cfg = MPCConfig(n=n, nu=nu, N=N, W=W, u_ref=jnp.zeros((nu,), dtype=jnp.float32))

    sls_cfg = SLSConfig(enable_fastsls=False)
    admm_cfg = ADMMConfig(eps_abs=1e-2, eps_rel=1e-2, rho_max=1e15, max_iterations=400)

    # Physical params (consistent with your later “total L, total M” version)
    L = 1.0
    M = 1.0
    l = (L / nlinks) * jnp.ones((nlinks,), dtype=jnp.float32)
    m = (M / nlinks) * jnp.ones((nlinks,), dtype=jnp.float32)
    lc = 0.5 * l
    I = (1.0 / 12.0) * m * l**2

    p = NLinkParams(l=l, m=m, I=I, lc=lc, g=9.81, b=0.05)
    parameter = (dt, p)

    # Upright reference
    q_ref = jnp.zeros((nlinks,), dtype=jnp.float32).at[0].set(jnp.pi / 2)
    qd_ref = jnp.zeros((nlinks,), dtype=jnp.float32)
    x_ref = jnp.concatenate([q_ref, qd_ref], axis=0)
    reference = jnp.tile(x_ref[None, :], (N + 1, 1))

    # Torque bounds
    u_max = float(MAX_U) * jnp.ones((nu,), dtype=jnp.float32)
    u_min = -u_max
    constraints_torque = make_torque_box_constraints(u_min, u_max)
    nc = 2 * nu

    disturbance = make_zero_disturbance(n=n, nc=nc)

    controller = GenericMPCControllerWrapper(
        sls_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_torque,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
    )

    # Initial condition
    x = jnp.concatenate([q_ref, qd_ref], axis=0)

    # Rollout length
    T_steps = min(int(1 / dt), 200)

    # Disturbance amplitude (match your heuristic unless explicitly provided)
    if max_amp is None:
        max_amp = float(0.525 * (N ** 0.488))

    tau_disturb = make_random_periodic_pushes(
        nu=nu,
        dt=dt,
        period_sec=push_period_sec,
        push_duration_sec=push_duration_sec,
        amp_max=max_amp,
        per_joint=True,
    )

    # Warmup compilation
    if warmup:
        _ = controller.run(x0=x, reference=reference, parameter=parameter)

    key = jax.random.PRNGKey(seed)

    near_constraint = 0
    total_time = 0.0
    min_time = float("inf")

    for k in range(T_steps):
        start = time.perf_counter()
        u0, X_pred, U_pred, V_pred = controller.run(x0=x, reference=reference, parameter=parameter)
        u0.block_until_ready()
        end = time.perf_counter()

        dt_run = end - start
        total_time += dt_run
        min_time = min(min_time, dt_run)

        key, sub = jax.random.split(key)
        tau_d = tau_disturb(k, sub)

        if k > 10:
            x = pendulum_step(x, u0 + tau_d, dt, p)
        else:
            x = pendulum_step(x, u0, dt, p)

        if jnp.any(jnp.abs(u0) >= alpha * u_max):
            near_constraint += 1

    frac = float(near_constraint) / float(T_steps)
    info = {
        "T_steps": T_steps,
        "near_constraint": int(near_constraint),
        "near_frac": frac,
        "mean_runtime_ms": (total_time / T_steps) * 1000.0,
        "min_runtime_ms": min_time * 1000.0,
        "dt": float(dt),
        "max_amp": float(max_amp),
        "alpha": float(alpha),
    }
    return frac, info


# -----------------------------
# Binary search wrapper
# -----------------------------
def find_max_u_for_near_frac(
    target_lo: float = 0.23,
    target_hi: float = 0.27,
    *,
    nlinks: int = 2,
    N: int = 25,
    dt: float | None = None,
    alpha: float = 0.9,
    seed: int = 0,
    max_iters: int = 20,
    tol_u: float = 1e-3,
    init_lo_u: float = 0.05,
    init_hi_u: float = 2.0,
) -> Tuple[float, float, dict]:
    """
    Returns (best_MAX_U, best_frac, best_info).
    """

    def in_band(frac: float) -> bool:
        return (target_lo <= frac) and (frac <= target_hi)

    # Bracket: want
    #   at low_u: frac >= target_hi (too many near-constraint events)
    #   at high_u: frac <= target_lo (too few near-constraint events)
    low_u = init_lo_u
    high_u = init_hi_u

    frac_low, info_low = evaluate_max_u(low_u, nlinks=nlinks, N=N, dt=dt, alpha=alpha, seed=seed, warmup=True)
    frac_high, info_high = evaluate_max_u(high_u, nlinks=nlinks, N=N, dt=dt, alpha=alpha, seed=seed, warmup=True)

    # Expand bracket if needed
    # Expand downwards if low_u already yields too small frac
    expand_guard = 0
    while frac_low < target_hi and expand_guard < 12:
        low_u *= 0.5
        frac_low, info_low = evaluate_max_u(low_u, nlinks=nlinks, N=N, dt=dt, alpha=alpha, seed=seed, warmup=True)
        expand_guard += 1

    expand_guard = 0
    while frac_high > target_lo and expand_guard < 12:
        high_u *= 2.0
        frac_high, info_high = evaluate_max_u(high_u, nlinks=nlinks, N=N, dt=dt, alpha=alpha, seed=seed, warmup=True)
        expand_guard += 1

    print("\n--- Bracket ---")
    print(f"low_u={low_u:.6f}  frac={100*frac_low:.2f}%  (want >= {100*target_hi:.2f}% to be 'too tight')")
    print(f"high_u={high_u:.6f} frac={100*frac_high:.2f}% (want <= {100*target_lo:.2f}% to be 'too loose')")

    # If we still don't have a proper bracket, bail with best effort
    if not (frac_low >= target_hi and frac_high <= target_lo):
        # pick whichever endpoint is closer to the band
        def band_distance(f):
            if f < target_lo:
                return target_lo - f
            if f > target_hi:
                return f - target_hi
            return 0.0

        d_low = band_distance(frac_low)
        d_high = band_distance(frac_high)
        if d_low <= d_high:
            print("WARNING: Could not bracket monotone crossing. Returning best-effort low_u endpoint.")
            return low_u, frac_low, info_low
        else:
            print("WARNING: Could not bracket monotone crossing. Returning best-effort high_u endpoint.")
            return high_u, frac_high, info_high

    # Binary search: maintain invariant
    #   low_u => frac >= target_hi
    #   high_u => frac <= target_lo
    best_u = None
    best_frac = None
    best_info = None

    for it in range(max_iters):
        mid_u = 0.5 * (low_u + high_u)
        frac_mid, info_mid = evaluate_max_u(mid_u, nlinks=nlinks, N=N, dt=dt, alpha=alpha, seed=seed, warmup=True)

        # Track best (distance to band)
        def band_distance(f):
            if f < target_lo:
                return target_lo - f
            if f > target_hi:
                return f - target_hi
            return 0.0

        if best_u is None or band_distance(frac_mid) < band_distance(best_frac):
            best_u, best_frac, best_info = mid_u, frac_mid, info_mid

        print(
            f"[it {it:02d}] MAX_U={mid_u:.6f}  near={100*frac_mid:.2f}%  "
            f"(mean {info_mid['mean_runtime_ms']:.2f} ms, min {info_mid['min_runtime_ms']:.2f} ms)"
        )

        if in_band(frac_mid):
            print("Hit target band.")
            return mid_u, frac_mid, info_mid

        # Monotone decision:
        # if frac_mid is too high => bounds too tight => increase MAX_U (move low up)
        if frac_mid > target_hi:
            low_u = mid_u
        # if frac_mid too low => bounds too loose => decrease MAX_U (move high down)
        elif frac_mid < target_lo:
            high_u = mid_u
        else:
            # inside band already handled; keep for completeness
            return mid_u, frac_mid, info_mid

        if abs(high_u - low_u) < tol_u:
            print("Reached tol_u without hitting band exactly; returning best-so-far.")
            break

    return best_u, best_frac, best_info


def main():
    target_lo = 0.23
    target_hi = 0.27

    best_u, best_frac, info = find_max_u_for_near_frac(
        target_lo=target_lo,
        target_hi=target_hi,
        nlinks=2,
        N=25,
        dt=None,       # keeps your dt=min(0.5/N, 0.005)
        alpha=0.9,
        seed=0,
        max_iters=20,
        tol_u=1e-3,
        init_lo_u=0.726,
        init_hi_u=0.74,
    )

    print("\n=== Result ===")
    print(f"BEST MAX_U: {best_u:.6f}")
    print(f"near-constraint: {100*best_frac:.2f}% (target band {100*target_lo:.2f}% - {100*target_hi:.2f}%)")
    print("info:", info)


if __name__ == "__main__":
    main()
