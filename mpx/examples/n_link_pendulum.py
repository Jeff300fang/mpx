"""
nlink_pendulum_mpc_experiment.py

End-to-end experiment: use your mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers.mpc
to control an n-link planar pendulum (stacked state x = [q; qd]) with torque box constraints.

Assumptions (matches the mpc() you pasted):
  - dynamics(x, u, t, *, parameter=...) returns x_{t+1} (discrete-time)
  - cost(W, reference, x, u, t) returns scalar stage cost
  - constraints(x, u, t) returns g(x,u,t) with g <= 0 (inequality)
  - disturbance(X_prefix) returns E used by get_controller(..., E, eta)
    We provide a conservative "zeros" shape (T, n, nc). If your get_controller expects a
    different E shape, adjust make_zero_disturbance() accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Any

import jax
import jax.numpy as jnp
import time

from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper

# -----------------------------
# N-link planar pendulum dynamics (Lagrangian via AD)
# State: x = [q; qd], input: u = tau
# -----------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class NLinkParams:
    l:  jnp.ndarray
    m:  jnp.ndarray
    I:  jnp.ndarray
    lc: jnp.ndarray
    g:  float = 9.81
    b:  float = 0.0

    def tree_flatten(self):
        # children must be arrays / pytrees of arrays
        children = (self.l, self.m, self.I, self.lc, jnp.asarray(self.g), jnp.asarray(self.b))
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        l, m, I, lc, g, b = children
        # convert scalar arrays back to Python floats if you want, but not necessary
        return cls(l=l, m=m, I=I, lc=lc, g=g, b=b)


def _com_positions(q: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    """COM positions of each link in world frame (x,y), relative joint angles q."""
    theta = jnp.cumsum(q)  # absolute angles
    ex, ey = jnp.cos(theta), jnp.sin(theta)

    dx_full = p.l * ex
    dy_full = p.l * ey

    xj = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dx_full)])
    yj = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dy_full)])

    xi = xj[:-1] + p.lc * ex
    yi = yj[:-1] + p.lc * ey
    return jnp.stack([xi, yi], axis=-1)  # (n,2)


def _kinetic_energy(q: jnp.ndarray, qd: jnp.ndarray, p: NLinkParams) -> jnp.ndarray:
    Jr = jax.jacobian(lambda qq: _com_positions(qq, p))(q)  # (n,2,n)
    v = jnp.einsum("i j k, k -> i j", Jr, qd)              # (n,2)
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
    n = u.shape[0]
    q = x[:n]
    qd = x[n:]
    qdd = nlink_qdd(q, qd, u, p)
    xdot = jnp.concatenate([qd, qdd], axis=0)
    return x + dt * xdot


def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter) -> jnp.ndarray:
    """Discrete-time dynamics required by your model_evaluator_helper residual."""
    dt, p = parameter
    return pendulum_step(x, u, dt, p)


# -----------------------------
# Cost and Constraints matching your mpc() expectations
# -----------------------------
def cost(W: jnp.ndarray, reference: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Stage cost: quadratic tracking to reference[t].
    W = [wq, wqd, wu]
    """
    wq, wqd, wu = W
    n = u.shape[0]
    xref = reference[t]
    q = x[:n]
    qd = x[n:]
    q_ref = xref[:n]
    qd_ref = xref[n:]
    return (
        wq * jnp.sum((q - q_ref) ** 2)
        + wqd * jnp.sum((qd - qd_ref) ** 2)
        + wu * jnp.sum(u**2)
    )


def make_torque_box_constraints(u_min: jnp.ndarray, u_max: jnp.ndarray):
    """
    Inequality constraints g(x,u,t) <= 0 for torque bounds:
      u - u_max <= 0
      u_min - u <= 0
    """
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints


def make_zero_disturbance(n: int, nc: int):
    """
    Conservative default: returns E with shape (T, n, nc).
    If fast_sls_utils.get_controller expects a different shape, change this function.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        # X_prefix: (T, n)
        T = X_prefix.shape[0]
        return jnp.zeros((T, n, nc), dtype=X_prefix.dtype)
    return disturbance


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
# Main experiment
# -----------------------------
def main():
    # Problem setup
    nlinks = 50
    n = 2 * nlinks
    nu = nlinks

    N = 100
    dt = 0.02

    # Weights: (q, qd, u)
    W = jnp.array([50.0, 5.0, 0.1], dtype=jnp.float32)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float32),
    )

    # Pendulum parameters
    p = NLinkParams(
        l=jnp.ones((nlinks,), dtype=jnp.float32),
        m=jnp.ones((nlinks,), dtype=jnp.float32),
        I=(1.0 / 12.0) * jnp.ones((nlinks,), dtype=jnp.float32),
        lc=0.5 * jnp.ones((nlinks,), dtype=jnp.float32),
        g=9.81,
        b=0.05,
    )
    parameter = (dt, p)

    # Reference: regulate to zero state
    reference = jnp.zeros((N + 1, n), dtype=jnp.float32)

    # Torque bounds
    u_max = 5.0 * jnp.ones((nu,), dtype=jnp.float32)
    u_min = -u_max
    constraints = make_torque_box_constraints(u_min, u_max)
    nc = 2 * nu

    disturbance = make_zero_disturbance(n=n, nc=nc)

    # Controller
    controller = GenericMPCControllerWrapper(
        config=cfg,
        dynamics=dynamics,
        constraints=constraints,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
    )

    # Closed-loop rollout
    key = jax.random.PRNGKey(0)
    q0  = 0.1 * jax.random.normal(key, (nlinks,), dtype=jnp.float32)
    qd0 = jnp.zeros((nlinks,), dtype=jnp.float32)
    x = jnp.concatenate([q0, qd0], axis=0)

    T_steps = 20
    xs = []
    us = []

    total_time = 0
    min_time = jnp.inf
    # Compilation warmup
    u0, X_pred, U_pred, V_pred = controller.run(x0=x, reference=reference, parameter=parameter)
    for k in range(T_steps):
        start = time.perf_counter()
        u0, X_pred, U_pred, V_pred = controller.run(x0=x, reference=reference, parameter=parameter)
        end = time.perf_counter()
        total_time = total_time + (end - start)
        min_time = min(total_time, min_time)
        x = pendulum_step(x, u0, dt, p)
        xs.append(x)
        us.append(u0)

    xs = jnp.stack(xs, axis=0)
    us = jnp.stack(us, axis=0)

    print("Final state x =", xs[-1])
    print("Final input u =", us[-1])
    print("Mean |u| =", jnp.mean(jnp.linalg.norm(us, axis=1)))
    print("Mean |q| =", jnp.mean(jnp.linalg.norm(xs[:, :nlinks], axis=1)))
    print("Total mean ms run:", total_time / T_steps * 1000 )
    print("Min_time:", min_time)


if __name__ == "__main__":
    main()
