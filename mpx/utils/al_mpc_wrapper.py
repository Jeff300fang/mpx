from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import time

# -----------------------------
# Import your solver
# -----------------------------
# IMPORTANT:
# Update this import path to wherever constrained_primal_dual_ilqr lives in your repo.
from mpx.primal_dual_ilqr.primal_dual_ilqr.constrained_optimizers import constrained_primal_dual_ilqr
from mpx.examples.pendulum_dynamics import pendulum_step


@dataclass(frozen=True)
class PDILQRConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray

    # solver knobs
    max_iterations: int = 100
    max_al_iterations: int = 5
    slope_threshold: float = 1e-4
    var_threshold: float = 0.0
    c_sq_threshold: float = 1e-4
    make_psd: bool = True
    psd_delta: float = 1e-6
    armijo_factor: float = 1e-4
    alpha_0: float = 1.0
    alpha_mult: float = 0.5
    alpha_min: float = 5e-5
    complementary_slackness_threshold: float = 1e-2
    penalty_init: float = 1.0
    penalty_update_rate: float = 10.0

    # MPC behavior
    shift: int = 1  # match your ADMM wrapper usage


def _shift_traj_last(X: jnp.ndarray, U: jnp.ndarray, V: jnp.ndarray, shift: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Shift trajectories forward by `shift` and repeat last element to keep shapes fixed.
      X: (N+1,n), U: (N,nu), V: (N+1,n)
    """
    if shift <= 0:
        return X, U, V
    X = jnp.concatenate([X[shift:], jnp.repeat(X[-1:], shift, axis=0)], axis=0)
    U = jnp.concatenate([U[shift:], jnp.repeat(U[-1:], shift, axis=0)], axis=0)
    V = jnp.concatenate([V[shift:], jnp.repeat(V[-1:], shift, axis=0)], axis=0)
    return X, U, V


def _rollout_guess(x0: jnp.ndarray, U: jnp.ndarray, dt: float, p) -> jnp.ndarray:
    """
    Deterministic rollout of dynamics to initialize X guess.
    """
    def step(x, u):
        x_next = pendulum_step(x, u, dt, p)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, U)
    X = jnp.concatenate([x0[None, :], xs], axis=0)
    return X


class PrimalDualILQRMPCWrapper:
    """
    Usage:
      u0, X_pred, U_pred, V_pred, it_ilqr, it_al, ok = controller.run(x0, reference)
    """
    def __init__(
        self,
        cfg: PDILQRConfig,
        dt: float,
        p,
        cost_fn_factory: Callable[[jnp.ndarray], Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]],
        inequality_constraint: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        init_with_rollout: bool = True,
    ):
        self.cfg = cfg
        self.dt = float(dt)
        self.p = p
        self.inequality_constraint = inequality_constraint
        self.cost_fn_factory = cost_fn_factory
        self.init_with_rollout = init_with_rollout

        n, nu, N = cfg.n, cfg.nu, cfg.N
        self.X = jnp.zeros((N + 1, n), dtype=jnp.float32)
        self.U = jnp.zeros((N, nu), dtype=jnp.float32)
        self.V = jnp.zeros((N + 1, n), dtype=jnp.float32)

        self._solve = self._build_solve()

    def _build_solve(self):
        cfg = self.cfg
        dt = self.dt
        p = self.p
        ineq = self.inequality_constraint

        def dynamics_local(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
            return pendulum_step(x, u, dt, p)

        def solve_once(x0: jnp.ndarray, X_in: jnp.ndarray, U_in: jnp.ndarray, V_in: jnp.ndarray, reference: jnp.ndarray):
            cost_local = self.cost_fn_factory(reference)

            X, U, V, it_ilqr, it_al, ok = constrained_primal_dual_ilqr(
                cost_local,
                dynamics_local,
                x0,
                X_in,
                U_in,
                V_in,
                equality_constraint=lambda x, u, t: jnp.empty((0,), dtype=x.dtype),
                inequality_constraint=ineq,
                max_iterations=cfg.max_iterations,
                max_al_iterations=cfg.max_al_iterations,
                slope_threshold=cfg.slope_threshold,
                var_threshold=cfg.var_threshold,
                c_sq_threshold=cfg.c_sq_threshold,
                make_psd=cfg.make_psd,
                psd_delta=cfg.psd_delta,
                armijo_factor=cfg.armijo_factor,
                alpha_0=cfg.alpha_0,
                alpha_mult=cfg.alpha_mult,
                alpha_min=cfg.alpha_min,
                complementary_slackness_threshold=cfg.complementary_slackness_threshold,
                penalty_init=cfg.penalty_init,
                penalty_update_rate=cfg.penalty_update_rate,
            )
            return X, U, V, it_ilqr, it_al, ok

        return jax.jit(solve_once)

    def run(self, x0: jnp.ndarray, reference: jnp.ndarray):
        # MPC warm-start housekeeping
        self.X = self.X.at[0].set(x0)
        self.U = self.U  # unchanged
        self.V = self.V  # unchanged

        # Optional: improve first-guess X with rollout (often stabilizes iLQR on big nlinks)
        # if self.init_with_rollout:
        #     self.X = _rollout_guess(x0, self.U, self.dt, self.p)
        X, U, V, it_ilqr, it_al, ok = self._solve(x0, self.X, self.U, self.V, reference)

        # Shift guesses for next MPC step
        self.X, self.U, self.V = _shift_traj_last(X, U, V, self.cfg.shift)

        u0 = U[0]
        return u0, X, U, V, it_ilqr, it_al, ok
