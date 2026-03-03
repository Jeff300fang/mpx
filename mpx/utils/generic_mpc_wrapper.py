
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers as optimizers
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig
from mpx.linearization_sls.src.rhs_eval import build_auto_rhs_analytic


def pack_dynamics_as_single_input(dynamics, nx: int, nu: int, *, parameter, t_dim: int = 1, t_as_scalar: bool = True):
    dyn = partial(dynamics, parameter=parameter)  # binds keyword-only arg

    D = nx + nu + t_dim

    def f_flat(z: jnp.ndarray) -> jnp.ndarray:
        x = z[:nx]
        u = z[nx:nx+nu]
        t_slice = z[nx+nu:nx+nu+t_dim]
        t = t_slice[0] if (t_dim == 1 and t_as_scalar) else t_slice
        return dyn(x, u, t)

    return f_flat, D

class GenericMPCControllerWrapper:
    def __init__(
        self,
        sls_config,
        sqp_config,
        admm_config,
        config,
        dynamics,
        constraints,
        obstacles,
        cost,
        num_constraints: int,
        disturbance,
        X_in,
        U_in,
        limited_memory: bool = False,
        shift: int = 1,
    ):
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.sls_config = sls_config
        self.sqp_config = sqp_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift
        self.obstacles = obstacles
        num_obstacles = self.obstacles.shape[0]
        self.h_ct_ws = jnp.zeros((config.N + 1, num_constraints - num_obstacles))
        self.beta_ws = jnp.ones((config.N + 1, config.N + 1, num_constraints - num_obstacles)) * 1e-10
        self.mu_ws = jnp.zeros((config.N + 1, num_constraints))
        self.Phi_x_ws = jnp.zeros((config.N + 1, config.N + 1, config.n, config.n))
        self.Phi_u_ws = jnp.zeros((config.N, config.N + 1, config.nu, config.n))

        # Warm starts
        # self.U0 = jnp.tile(jnp.zeros((config.nu,)), (config.N, 1))         # (T, nu)
        self.U0 = U_in
        # self.X0 = jnp.tile(jnp.zeros((config.n,)),  (config.N + 1, 1))     # (T+1, n)
        self.X0 = X_in    # (T+1, n)
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
        D = config.n + config.nu
        V = D + 1
        f_flat, D_flat = pack_dynamics_as_single_input(
            dynamics,
            nx=config.n,
            nu=config.nu,
            parameter=config.dt,     # whatever your "parameter" is
            t_dim=1,
            t_as_scalar=True
        )
        _, rhs_tm_fn = build_auto_rhs_analytic(f_flat, D=D_flat, V=V)
        # TODO: Make this an argument
        splts_cfg = (4, 4, 4, 4, 1)
        work = partial(
            optimizers.mpc,
            self.sls_config,
            self.sqp_config,
            self.admm_config,
            cost,
            dynamics,
            None,               # hessian_approx
            limited_memory,
            constraints,
            disturbance,
            rhs_tm_fn,
            splts_cfg
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
        X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, betaN, muN = self._solve(
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
            self.obstacles,
            self.h_ct_ws, self.beta_ws, self.mu_ws, self.Phi_x_ws, self.Phi_u_ws
        )
        self.h_ct_ws = jnp.concatenate(
            [backoffs[self.shift:], jnp.tile(backoffs[-1:], (self.shift, 1))],
            axis=0
        )
        self.beta_ws = jnp.concatenate(
            [betaN[self.shift:], jnp.tile(betaN[-1:], (self.shift, 1))],
            axis=0
        )
        self.mu_ws = jnp.concatenate(
            [muN[self.shift:], jnp.tile(muN[-1:], (self.shift, 1))],
            axis=0
        )

        # Warm-start ADMM-ish states
        # TODO: Make this an option to warm start / not warm
        # self.w = jnp.zeros_like(w)
        # self.y = jnp.zeros_like(y)
        # Car
        self.w = jnp.concatenate([w[self.shift:], jnp.tile(w[-1:], (self.shift, 1))], axis=0)
        self.y = jnp.concatenate([y[self.shift:], jnp.tile(y[-1:], (self.shift, 1))], axis=0)

        # rho management (matches your earlier wrapper pattern)
        rho = jnp.asarray(rho, dtype=self.rho.dtype)
        # self.rho = jnp.maximum(jnp.minimum(rho, 1e3) * 0.9, 0.01) # car
        # self.rho = jnp.maximum(jnp.minimum(rho, 0.1) * 0.9, 0.01) # pendulum 
        rho = jnp.array(10.0)
        self.y = rho / self.rho * self.y

        self.U0, self.X0, self.V0 = self._update_and_extract(U, X, V, x0)
        self.Phi_u_ws = Phi_u
        self.Phi_x_ws = Phi_x
        return U[0], X, U, V, backoffs, Phi_x, Phi_u