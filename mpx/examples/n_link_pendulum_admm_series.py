"""
nlink_pendulum_mpc_experiment.py

End-to-end experiment: use your mpx.utils.generic_mpc_wrapper.GenericMPCControllerWrapper
to control an n-link planar pendulum (state x = [q; qd]) with torque box constraints.

This version targets an UPRIGHT equilibrium and incorporates angle wrapping in the cost.

Assumptions (matches your mpc() usage):
  - dynamics(x, u, t, *, parameter=...) returns x_{t+1} (discrete-time)
  - cost(W, reference, x, u, t) returns scalar stage cost
  - constraints(x, u, t) returns g(x,u,t) with g <= 0 (inequality)
  - disturbance(X_prefix) returns E used by get_controller(..., E, eta)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import time
import jax
import jax.numpy as jnp

from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.utils.generic_mpc_wrapper import GenericMPCControllerWrapper


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
    n = u.shape[0]
    q = x[:n]
    qd = x[n:]
    qdd = nlink_qdd(q, qd, u, p)
    xdot = jnp.concatenate([qd, qdd], axis=0)
    return x + dt * xdot

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    """Discrete-time dynamics required by your model evaluator."""
    dt, p = parameter
    return pendulum_step(x, u, dt, p)


# -----------------------------
# Cost and Constraints matching your mpc() expectations
# -----------------------------
def cost(W: jnp.ndarray, reference: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray: 
    """ Stage cost: quadratic tracking to reference[t], WITH angle wrapping on q.
    W = [wq, wqd, wu] """
    wq, wqd, wu = W
    n = u.shape[0]
    xref = reference[t]
    q = x[:n]
    qd = x[n:]
    q_ref = xref[:n]
    qd_ref = xref[n:]
    dq = wrap_to_pi(q - q_ref)
    dqd = qd - qd_ref 
    return ( wq * jnp.sum(dq * dq) + wqd * jnp.sum(dqd * dqd) + wu * jnp.sum(u * u) )



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
    If your get_controller expects a different E shape, change this function.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        # X_prefix: (T, n)
        T = X_prefix.shape[0]
        return jnp.zeros((T, n, nc), dtype=X_prefix.dtype)

    return disturbance


def make_end_effector_height_constraint(y_min, p: NLinkParams):
    def constraints(x, u, t):
        n = u.shape[0]
        q = x[:n]
        pos = _com_positions(q, p)  # or compute EE directly
        y_ee = pos[-1, 1]
        return jnp.array([y_min - y_ee])
    return constraints

def make_random_periodic_pushes(
    nu: int,
    dt: float,
    period_sec: float = 0.6,         # how often a push window starts
    push_duration_sec: float = 0.08, # how long the push lasts
    amp_max: float = 2.0,            # max torque magnitude per joint
    per_joint: bool = True,          # True: each joint gets its own random push
) -> Callable[[int, jax.Array], jnp.ndarray]:
    period_steps = max(1, int(period_sec / dt))
    dur_steps = max(1, int(push_duration_sec / dt))

    def tau_disturb(k: int, key: jax.Array) -> jnp.ndarray:
        phase = k % period_steps
        on = jnp.where(phase < dur_steps, 1.0, 0.0) # 1.0 during push window else 0.0

        key, sub = jax.random.split(key)
        if per_joint:
            push = amp_max * (2.0 * jax.random.uniform(sub, (nu,), dtype=jnp.float32) - 1.0)
        else:
            s = amp_max * (2.0 * jax.random.uniform(sub, (), dtype=jnp.float32) - 1.0)
            push = jnp.ones((nu,), dtype=jnp.float32) * s

        return on * push

    return tau_disturb

def make_first_joint_angle_constraint(q0_min: float, q0_max: float):
    """
    Enforce: q0_min <= q[0] <= q0_max
    """
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        q0 = x[0]
        return jnp.array([
            q0 - q0_max,   # q0 <= q0_max
            q0_min - q0,   # q0 >= q0_min
        ])
    return constraints

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
# # -----------------------------
# def main():
#     # Problem setup
#     nlinks = 10
#     n = 2 * nlinks
#     nu = nlinks

#     N = 10
#     dt = min(0.5 / N, 0.005)

#     # Weights: (q, qd, u)
#     W = jnp.array([50.0, 5.0, 0.1], dtype=jnp.float32)

#     cfg = MPCConfig(
#         n=n,
#         nu=nu,
#         N=N,
#         W=W,
#         u_ref=jnp.zeros((nu,), dtype=jnp.float32),
#     )

#     admm_cfg = ADMMConfig(
#         eps_abs=1e-2,
#         eps_rel=1e-2,
#         condense_block_size=5,
#         rho_max=1e10,
#         max_iterations=400
#     )

#     # Pendulum parameters
#     p = NLinkParams(
#         l=jnp.ones((nlinks,), dtype=jnp.float32),
#         m=jnp.ones((nlinks,), dtype=jnp.float32),
#         I=(1.0 / 12.0) * jnp.ones((nlinks,), dtype=jnp.float32),
#         lc=0.5 * jnp.ones((nlinks,), dtype=jnp.float32),
#         g=9.81,
#         b=0.05,
#     )
#     parameter = (dt, p)

#     # -----------------------------
#     # Upright reference
#     # -----------------------------
#     # "Upright" here means all links point along +y, i.e. absolute angles theta_i = +pi/2.
#     # With relative angles q, a simple equilibrium is:
#     #   q_ref = [pi/2, 0, 0, ..., 0], qd_ref = 0.
#     q_ref = jnp.zeros((nlinks,), dtype=jnp.float32).at[0].set(jnp.pi / 2)
#     qd_ref = jnp.zeros((nlinks,), dtype=jnp.float32)
#     x_ref = jnp.concatenate([q_ref, qd_ref], axis=0)

#     # Constant reference over the horizon
#     reference = jnp.tile(x_ref[None, :], (N + 1, 1))

#     # Torque bounds
#     u_max = (3.0) * jnp.ones((nu,), dtype=jnp.float32)
#     u_min = -u_max
#     constraints_torque = make_torque_box_constraints(u_min, u_max)
#     nc = 2 * nu

#     disturbance = make_zero_disturbance(n=n, nc=nc)

#     # Controller
#     controller = GenericMPCControllerWrapper(
#         admm_cfg,
#         config=cfg,
#         dynamics=dynamics,
#         constraints=constraints_torque,
#         cost=cost,
#         num_constraints=nc,
#         disturbance=disturbance,
#         limited_memory=False,
#         shift=1,
#     )

#     # -----------------------------
#     # Initial condition: near upright (local balancing)
#     # -----------------------------
#     q0 = q_ref   # small perturbation around upright
#     qd0 = jnp.zeros((nlinks,), dtype=jnp.float32)
#     x = jnp.concatenate([q0, qd0], axis=0)
#     # jax.debug.print("{}", x)

#     # Closed-loop rollout
#     T_steps = min(int(1/dt), 1000)
#     # T_steps = 100 
#     xs = []
#     us = []

#     total_time = 0.0
#     min_time = float("inf")
#     max_amp = 0.525 * N ** 0.488
#     # max_amp = 5
#     print("MAX:", max_amp)
#     # Compilation warmup
#     _ = controller.run(x0=x, reference=reference, parameter=parameter)

#     tau_disturb = make_random_periodic_pushes(
#         nu=nu,
#         dt=dt,
#         period_sec=0.005,
#         push_duration_sec=0.1,
#         amp_max=max_amp,
#         per_joint=True,
#     )
#     key = jax.random.PRNGKey(0)
#     near_constraint = 0
#     alpha = 0.9
#     for k in range(T_steps):
#         print(f"sim iteration {k}")
#         start = time.perf_counter()
#         u0, X_pred, U_pred, V_pred = controller.run(x0=x, reference=reference, parameter=parameter)
#         u0.block_until_ready()
#         end = time.perf_counter()

#         dt_run = end - start
#         total_time += dt_run
#         min_time = min(min_time, dt_run)

#         key, sub = jax.random.split(key)
#         tau_d = tau_disturb(k, sub)
#         if k > 10:
#             x = pendulum_step(x, u0 + tau_d, dt, p)
#         else:
#             x = pendulum_step(x, u0, dt, p)
#         # jax.debug.print(" u = {}", u0)
#         # jax.debug.print("x = {}", x)
#         if jnp.any(jnp.abs(u0) >= alpha * u_max):
#             near_constraint += 1
#         xs.append(x)
#         us.append(u0)

#     xs = jnp.stack(xs, axis=0)
#     us = jnp.stack(us, axis=0)

#     print("Final state x =", xs[-1])
#     print("Final input u =", us[-1])
#     print("Mean |u| =", jnp.mean(jnp.linalg.norm(us, axis=1)))
#     print("Max |u| =", jnp.max(jnp.abs(us)))
#     print("Mean |q| =", jnp.mean(jnp.linalg.norm(xs[:, :nlinks], axis=1)))
#     print("Mean run time (ms):", (total_time / T_steps) * 1000.0)
#     print("Min run time (ms):", min_time * 1000.0)
#     print("Near constraint: ", near_constraint)
#     print("Percentage:", near_constraint / T_steps)

def run_single_experiment(seed: int):
    # -----------------------------
    # Problem setup (UNCHANGED)
    # -----------------------------
    nlinks = 10
    n = 2 * nlinks
    nu = nlinks

    N = 3000
    dt = min(0.5 / N, 0.005)

    W = jnp.array([50.0, 5.0, 0.1], dtype=jnp.float32)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float32),
    )

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        condense_block_size=5,
        rho_max=1e10,
        max_iterations=400,
    )

    p = NLinkParams(
        l=jnp.ones((nlinks,), dtype=jnp.float32),
        m=jnp.ones((nlinks,), dtype=jnp.float32),
        I=(1.0 / 12.0) * jnp.ones((nlinks,), dtype=jnp.float32),
        lc=0.5 * jnp.ones((nlinks,), dtype=jnp.float32),
        g=9.81,
        b=0.05,
    )
    parameter = (dt, p)

    # Upright reference
    q_ref = jnp.zeros((nlinks,), dtype=jnp.float32).at[0].set(jnp.pi / 2)
    qd_ref = jnp.zeros((nlinks,), dtype=jnp.float32)
    x_ref = jnp.concatenate([q_ref, qd_ref], axis=0)
    reference = jnp.tile(x_ref[None, :], (N + 1, 1))

    u_max = 3.0 * jnp.ones((nu,), dtype=jnp.float32)
    u_min = -u_max
    constraints = make_torque_box_constraints(u_min, u_max)
    nc = 2 * nu

    disturbance = make_zero_disturbance(n=n, nc=nc)

    controller = GenericMPCControllerWrapper(
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        limited_memory=False,
        shift=1,
    )

    # Initial state
    x = jnp.concatenate([q_ref, jnp.zeros((nlinks,), dtype=jnp.float32)], axis=0)

    # Compile once per run (important)
    _ = controller.run(x0=x, reference=reference, parameter=parameter)

    tau_disturb = make_random_periodic_pushes(
        nu=nu,
        dt=dt,
        period_sec=0.005,
        push_duration_sec=0.1,
        amp_max=0.525 * N ** 0.488,
        per_joint=True,
    )

    key = jax.random.PRNGKey(seed)

    T_steps = min(int(1 / dt), 1000)

    solve_times = []

    for k in range(T_steps):
        start = time.perf_counter()
        u0, *_ = controller.run(x0=x, reference=reference, parameter=parameter)
        u0.block_until_ready()
        end = time.perf_counter()

        solve_times.append(end - start)

        key, sub = jax.random.split(key)
        tau_d = tau_disturb(k, sub)
        x = pendulum_step(x, u0 + tau_d if k > 10 else u0, dt, p)

    solve_times = jnp.array(solve_times)

    return {
        "mean_ms": float(jnp.mean(solve_times) * 1e3),
        "min_ms": float(jnp.min(solve_times) * 1e3),
        "max_ms": float(jnp.max(solve_times) * 1e3),
        "std_ms": float(jnp.std(solve_times) * 1e3),
    }

def main():
    num_runs = 10
    stats = []

    for i in range(num_runs):
        print(f"\n=== Run {i+1}/{num_runs} ===")
        out = run_single_experiment(seed=i)
        print(out)
        stats.append(out)

    # Aggregate
    mean_times = jnp.array([s["mean_ms"] for s in stats])
    min_times  = jnp.array([s["min_ms"]  for s in stats])
    max_times  = jnp.array([s["max_ms"]  for s in stats])

    print("\n===== AGGREGATE RESULTS =====")
    print(f"Mean solve time (ms): {mean_times.mean():.3f}")
    print(f"Std  solve time (ms): {mean_times.std():.3f}")
    print(f"Best mean (ms):       {mean_times.min():.3f}")
    print(f"Worst mean (ms):      {mean_times.max():.3f}")

if __name__ == "__main__":
    main()
