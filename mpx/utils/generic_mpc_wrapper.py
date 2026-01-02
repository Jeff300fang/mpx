
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers


class GenericMPCControllerWrapper:
    def __init__(
        self,
        sls_config,
        admm_config,
        config,
        dynamics,
        constraints,
        cost,
        num_constraints: int,
        disturbance,
        limited_memory: bool = False,
        shift: int = 1,
    ):
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.sls_config = sls_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift

        # Warm starts
        self.U0 = jnp.tile(jnp.zeros((config.nu,)), (config.N, 1))         # (T, nu)
        self.X0 = jnp.tile(jnp.zeros((config.n,)),  (config.N + 1, 1))     # (T+1, n)
        self.V0 = jnp.zeros((config.N + 1, config.n))                      # (T+1, n)

        # ADMM states for inequality constraints
        self.w = jnp.zeros((config.N + 1, num_constraints))                # (T+1, nc)
        self.y = jnp.zeros((config.N + 1, num_constraints))                # (T+1, nc)
        self.rho = jnp.asarray(0.1, dtype=self.w.dtype)

        # Store callables
        self.dynamics = dynamics
        self.constraints = constraints
        self.cost = cost
        self.disturbance = disturbance

        # mpc signature in your pasted file:
        # mpc(cost, dynamics, hessian_approx, limited_memory, constraints, disturbance,
        #     reference, parameter, W, x0, X_in, U_in, V_in, w, y, rho)
        work = partial(
            optimizers.mpc,
            self.sls_config,
            self.admm_config,
            cost,
            dynamics,
            None,               # hessian_approx
            limited_memory,
            constraints,
            disturbance,
        )
        self._solve = jax.jit(work)

        @jax.jit
        def update_and_extract(U, X, V, x0):
            def safe():
                s = self.shift
                new_U0 = jnp.concatenate([U[s:], jnp.tile(U[-1:], (s, 1))], axis=0)
                new_X0 = jnp.concatenate([X[s:], jnp.tile(X[-1:], (s, 1))], axis=0)
                new_V0 = jnp.concatenate([V[s:], jnp.tile(V[-1:], (s, 1))], axis=0)
                return new_U0, new_X0, new_V0

            def unsafe():
                new_U0 = jnp.tile(self.config.u_ref, (self.config.N, 1))
                new_X0 = jnp.tile(x0, (self.config.N + 1, 1))
                new_V0 = jnp.zeros((self.config.N + 1, self.config.n))
                return new_U0, new_X0, new_V0

            return jax.lax.cond(jnp.isnan(U[0, 0]), unsafe, safe)

        self._update_and_extract = update_and_extract

    def run(self, x0: jnp.ndarray, reference: jnp.ndarray, parameter: Any):
        X, U, V, w, y, rho = self._solve(
            reference,
            parameter,
            self.config.W,
            x0,
            self.X0,
            self.U0,
            self.V0,
            self.w,
            self.y,
            self.rho,
        )

        # Warm-start ADMM-ish states
        # TODO: Make this an option to warm start / not warm
        s = self.shift
        self.w = jnp.zeros_like(w)
        self.y = jnp.zeros_like(y)
        # Car
        # self.w = jnp.concatenate([w[self.shift:], jnp.tile(w[-1:], (self.shift, 1))], axis=0)
        # self.y = jnp.concatenate([y[self.shift:], jnp.tile(y[-1:], (self.shift, 1))], axis=0)

        # rho management (matches your earlier wrapper pattern)
        rho = jnp.asarray(rho, dtype=self.rho.dtype)
        self.rho = jnp.maximum(jnp.minimum(rho, 1e3) * 0.9, 0.1) # car
        # self.rho = jnp.maximum(jnp.minimum(rho, 0.1) * 0.9, 0.1) # pendulum 
        self.y = rho / self.rho * self.y

        self.U0, self.X0, self.V0 = self._update_and_extract(U, X, V, x0)

        return U[0], X, U, V