from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


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
