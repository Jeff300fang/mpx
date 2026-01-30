from __future__ import annotations
from jax import config
config.update("jax_enable_x64", True)
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp

from typing import Callable

import jax
import jax.numpy as jnp


from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from mpx.utils.al_mpc_wrapper import PDILQRConfig, PrimalDualILQRMPCWrapper

# -----------------------------
# Angle wrapping
# -----------------------------
def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles elementwise to (-pi, pi]."""
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
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        l, m, I, lc, g, b = children
        return cls(l=l, m=m, I=I, lc=lc, g=g, b=b)


def _com_positions(q: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    """COM positions of each link in world frame (x,y), relative joint angles q."""
    theta = jnp.cumsum(q)  # absolute angles
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
    # y increases upward; gravity acts toward -y, so V = m g y
    y = _com_positions(q, p)[:, 1]
    return jnp.sum(p.m * p.g * y)


@jax.jit
def nlink_qdd(q: jnp.ndarray, qd: jnp.ndarray, tau: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    T = lambda qq, qdq: _kinetic_energy(qq, qdq, p)
    V = lambda qq: _potential_energy(qq, p)

    # Mass matrix: M = ∂/∂qd (∂T/∂qd)
    M = jax.jacobian(lambda qdq: jax.grad(lambda qdq2: T(q, qdq2))(qdq))(qd)

    # coriolis-like term: (∂/∂q (∂T/∂qd)) qd
    dpdq_dq = jax.jacobian(lambda qq: jax.grad(lambda qdq: T(qq, qdq))(qd))(q)
    coriolis_like = dpdq_dq @ qd

    dT_dq = jax.grad(lambda qq: T(qq, qd))(q)
    dV_dq = jax.grad(V)(q)

    b = jnp.asarray(p.b)
    damping = (b * qd) if b.shape != () else (b * qd)

    rhs = tau - damping - coriolis_like + dT_dq - dV_dq
    return jnp.linalg.solve(M, rhs)


@jax.jit
def pendulum_step(x: jnp.ndarray, u: jnp.ndarray, dt: float, p: NLinkParams) -> jnp.ndarray:
    nlinks = u.shape[0]
    q = x[:nlinks]
    qd = x[nlinks:]
    qdd = nlink_qdd(q, qd, u, p)
    xdot = jnp.concatenate([qd, qdd], axis=0)
    return x + dt * xdot


def dynamics_with_parameter(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    """Your original signature (kept for parity)."""
    dt, p = parameter
    return pendulum_step(x, u, dt, p)



def tracking_cost_factory(W: jnp.ndarray, reference: jnp.ndarray, nu: int) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Returns cost(x,u,t) matching your original stage cost:
      wq * ||wrap(q - q_ref)||^2 + wqd * ||qd - qd_ref||^2 + wu * ||u||^2
    reference: (N+1, n)
    """
    W = jnp.asarray(W)

    def cost(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        wq, wqd, wu = W
        xref = reference[t]
        q = x[:nu]
        qd = x[nu:]
        q_ref = xref[:nu]
        qd_ref = xref[nu:]
        dq = wrap_to_pi(q - q_ref)
        dqd = qd - qd_ref
        return (wq * jnp.sum(dq * dq) +
                wqd * jnp.sum(dqd * dqd) +
                wu * jnp.sum(u * u))

    return cost


def make_torque_box_inequality(u_min: jnp.ndarray, u_max: jnp.ndarray) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for torque bounds:
      u - u_max <= 0
      u_min - u <= 0
    Returns shape (2*nu,)
    """
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def ineq(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return ineq


def make_end_effector_height_inequality(y_min: float, p: NLinkParams) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Example extra inequality: y_min - y_ee <= 0  (i.e. y_ee >= y_min).
    Not used by default; included for completeness.
    """
    def ineq(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        nu = u.shape[0]
        q = x[:nu]
        pos = _com_positions(q, p)   # (nlinks, 2); last link COM, not true EE
        y_last = pos[-1, 1]
        return jnp.array([y_min - y_last], dtype=x.dtype)

    return ineq


def make_first_joint_angle_inequality(q0_min: float, q0_max: float) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Enforce: q0_min <= q[0] <= q0_max
      q0 - q0_max <= 0
      q0_min - q0 <= 0
    """
    def ineq(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        q0 = x[0]
        return jnp.array([q0 - q0_max, q0_min - q0], dtype=x.dtype)

    return ineq


def combine_inequalities(*ineq_fns: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Concatenate multiple inequality constraints into one vector.
    """
    def ineq(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        parts = [fn(x, u, t) for fn in ineq_fns]
        if len(parts) == 0:
            return jnp.empty((0,), dtype=x.dtype)
        return jnp.concatenate(parts, axis=0)

    return ineq


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
            push = amp_max * (2.0 * jax.random.uniform(sub, (nu,), dtype=jnp.float64) - 1.0)
        else:
            s = amp_max * (2.0 * jax.random.uniform(sub, (), dtype=jnp.float64) - 1.0)
            push = jnp.ones((nu,), dtype=jnp.float64) * s

        return on * push

    return tau_disturb

@dataclass(frozen=True)
class MPCBenchmarkConfig:
    nlinks: int = 10
    N: int = 200
    dt_cap: float = 0.005
    T_steps_cap: int = 500

    # weights: (q, qd, u)
    W: tuple[float, float, float] = (50.0, 5.0, 0.1)

    # torque bounds
    u_max: float = 3.0

    # disturbance
    dist_on_after_k: int = 10
    push_period_sec: float = 0.005
    push_duration_sec: float = 0.1


def main():
    cfg = MPCBenchmarkConfig()

    nlinks = cfg.nlinks
    n = 2 * nlinks
    nu = nlinks

    N = 2000
    dt = min(0.5 / N, cfg.dt_cap)

    W = jnp.array(cfg.W, dtype=jnp.float64)

    # Pendulum parameters
    p = NLinkParams(
        l=jnp.ones((nlinks,), dtype=jnp.float64),
        m=jnp.ones((nlinks,), dtype=jnp.float64),
        I=(1.0 / 12.0) * jnp.ones((nlinks,), dtype=jnp.float64),
        lc=0.5 * jnp.ones((nlinks,), dtype=jnp.float64),
        g=9.81,
        b=0.05,
    )

    # Upright reference:
    # all links point along +y => absolute angles theta_i = pi/2.
    # with relative angles q: q_ref = [pi/2, 0, 0, ...], qd_ref = 0.
    q_ref = jnp.zeros((nlinks,), dtype=jnp.float64).at[0].set(jnp.pi / 2)
    qd_ref = jnp.zeros((nlinks,), dtype=jnp.float64)
    x_ref = jnp.concatenate([q_ref, qd_ref], axis=0)

    reference = jnp.tile(x_ref[None, :], (N + 1, 1))

    # Torque inequality constraints
    u_max = cfg.u_max * jnp.ones((nu,), dtype=jnp.float64)
    u_min = -u_max
    ineq = make_torque_box_inequality(u_min, u_max)

    # PD-iLQR MPC controller
    pd_cfg = PDILQRConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float64),

        # You can tune these to match your baseline behavior
        max_iterations=10000,
        max_al_iterations=100,
        penalty_init=0.1,
        penalty_update_rate=5.0,
        c_sq_threshold=1e-3,
        shift=1,
    )

    # cost factory closes over W and nu; wrapper will pass reference each call
    def cost_factory(reference_in: jnp.ndarray):
        return tracking_cost_factory(W=W, reference=reference_in, nu=nu)

    controller = PrimalDualILQRMPCWrapper(
        cfg=pd_cfg,
        dt=dt,
        p=p,
        cost_fn_factory=cost_factory,
        inequality_constraint=ineq,
        init_with_rollout=True,  # recommended for high nlinks
    )

    # Initial condition: near upright
    x = x_ref

    # Disturbance generator (your same logic)
    max_amp = 0.525 * (N ** 0.488)
    print("MAX disturbance amp:", float(max_amp))

    tau_disturb = make_random_periodic_pushes(
        nu=nu,
        dt=dt,
        period_sec=cfg.push_period_sec,
        push_duration_sec=cfg.push_duration_sec,
        amp_max=float(max_amp),
        per_joint=True,
    )
    key = jax.random.PRNGKey(0)

    # Timing/benchmark bookkeeping
    T_steps = min(int(1 / dt), cfg.T_steps_cap)
    total_time = 0.0
    min_time = float("inf")
    near_constraint = 0
    alpha = 0.9

    xs = []
    us = []
    it_ilqrs = []
    it_als = []
    oks = []

    # Compilation warmup
    u0, X_pred, U_pred, V_pred, it_ilqr, it_al, ok = controller.run(x0=x, reference=reference)
    u0.block_until_ready()

    for k in range(T_steps):
        print(f"sim iteration {k}")
        start = time.perf_counter()

        u0, X_pred, U_pred, V_pred, it_ilqr, it_al, ok = controller.run(x0=x, reference=reference)
        u0.block_until_ready()

        end = time.perf_counter()
        dt_run = end - start
        print(dt_run)
        total_time += dt_run
        min_time = min(min_time, dt_run)

        key, sub = jax.random.split(key)
        tau_d = tau_disturb(k, sub)

        if k > cfg.dist_on_after_k:
            x = pendulum_step(x, u0 + tau_d, dt, p)
        else:
            x = pendulum_step(x, u0, dt, p)

        if jnp.any(jnp.abs(u0) >= alpha * u_max):
            near_constraint += 1

        xs.append(x)
        us.append(u0)
        it_ilqrs.append(it_ilqr)
        it_als.append(it_al)
        oks.append(ok)

    xs = jnp.stack(xs, axis=0)
    us = jnp.stack(us, axis=0)
    it_ilqrs = jnp.asarray(it_ilqrs)
    it_als = jnp.asarray(it_als)
    oks = jnp.asarray(oks)

    print("Final state x =", xs[-1])
    print("Final input u =", us[-1])
    print("Mean |u| =", jnp.mean(jnp.linalg.norm(us, axis=1)))
    print("Max |u| =", jnp.max(jnp.abs(us)))
    print("Mean |q| =", jnp.mean(jnp.linalg.norm(xs[:, :nlinks], axis=1)))
    print("Mean run time (ms):", (total_time / T_steps) * 1000.0)
    print("Min run time (ms):", min_time * 1000.0)
    print("Near constraint:", int(near_constraint))
    print("Percentage:", float(near_constraint) / float(T_steps))

    print("Mean it_ilqr:", float(jnp.mean(it_ilqrs)))
    print("Mean it_al:", float(jnp.mean(it_als)))
    print("Solve ok rate:", float(jnp.mean(oks.astype(jnp.float64))))

    # Optional: dump raw arrays
    # print(us)
    # print(jnp.argmax(us))
    # print(us.shape)


if __name__ == "__main__":
    main()
