from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import fast_sls_solve_gpu, SLSConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation


# -----------------------------
# Utilities
# -----------------------------
def build_double_integrator_2d_mats(dt: float, dtype=jnp.float32):
    A = jnp.array(
        [
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    B = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [dt,  0.0],
            [0.0, dt ],
        ],
        dtype=dtype,
    )
    return A, B


def make_box_constraints(T: int, px_max, py_max, vx_max, vy_max, ux_max, uy_max, dtype=jnp.float32):
    nx, nu = 4, 2
    nc = 12

    C_stage = jnp.zeros((nc, nx), dtype=dtype)
    D_stage = jnp.zeros((nc, nu), dtype=dtype)
    f_stage = jnp.zeros((nc,), dtype=dtype)

    # px, py
    C_stage = C_stage.at[0, 0].set( 1.0); f_stage = f_stage.at[0].set(px_max)
    C_stage = C_stage.at[1, 0].set(-1.0); f_stage = f_stage.at[1].set(px_max)
    C_stage = C_stage.at[2, 1].set( 1.0); f_stage = f_stage.at[2].set(py_max)
    C_stage = C_stage.at[3, 1].set(-1.0); f_stage = f_stage.at[3].set(py_max)

    # vx, vy
    C_stage = C_stage.at[4, 2].set( 1.0); f_stage = f_stage.at[4].set(vx_max)
    C_stage = C_stage.at[5, 2].set(-1.0); f_stage = f_stage.at[5].set(vx_max)
    C_stage = C_stage.at[6, 3].set( 1.0); f_stage = f_stage.at[6].set(vy_max)
    C_stage = C_stage.at[7, 3].set(-1.0); f_stage = f_stage.at[7].set(vy_max)

    # ax, ay
    D_stage = D_stage.at[8, 0].set( 1.0); f_stage = f_stage.at[8].set(ux_max)
    D_stage = D_stage.at[9, 0].set(-1.0); f_stage = f_stage.at[9].set(ux_max)
    D_stage = D_stage.at[10, 1].set( 1.0); f_stage = f_stage.at[10].set(uy_max)
    D_stage = D_stage.at[11, 1].set(-1.0); f_stage = f_stage.at[11].set(uy_max)

    C = jnp.broadcast_to(C_stage, (T + 1, nc, nx)).astype(dtype)
    D = jnp.broadcast_to(D_stage, (T + 1, nc, nu)).astype(dtype)
    f = jnp.broadcast_to(f_stage, (T + 1, nc)).astype(dtype)
    return C, D, f


def make_quadratic_cost(T: int, x_ref: jnp.ndarray, dtype=jnp.float32):
    nx, nu = 4, 2
    w_p = 100.0
    w_v = 1.0
    w_u = 0.1

    Qk = jnp.diag(jnp.array([w_p, w_p, w_v, w_v], dtype=dtype))
    Rk = w_u * jnp.eye(nu, dtype=dtype)

    Q = jnp.broadcast_to(Qk, (T + 1, nx, nx)).astype(dtype)
    R = jnp.broadcast_to(Rk, (T,     nu, nu)).astype(dtype)

    x_ref = x_ref.astype(dtype)
    q = jnp.einsum("kij,j->ki", -Q, x_ref)
    r = jnp.zeros((T, nu), dtype=dtype)

    M = jnp.zeros((T, nx, nu), dtype=dtype)
    return Q, q, R, r, M


def build_ltv_dynamics(T: int, dt: float, dtype=jnp.float32):
    A0, B0 = build_double_integrator_2d_mats(dt, dtype=dtype)
    A = jnp.broadcast_to(A0, (T, 4, 4)).astype(dtype)
    B = jnp.broadcast_to(B0, (T, 4, 2)).astype(dtype)
    c = jnp.zeros((T + 1, 4), dtype=dtype)
    return A, B, c


def build_disturbance_scale(T: int, sigma: float, dtype=jnp.float32):
    nx = 4
    E0 = sigma * jnp.eye(nx, dtype=dtype)
    E = jnp.broadcast_to(E0, (T, nx, nx)).astype(dtype)
    return E


# -----------------------------
# Dynamics stepping for MPC sim
# -----------------------------
def step_double_integrator(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    px, py, vx, vy = x
    ax, ay = u
    return jnp.array(
        [px + vx * dt,
         py + vy * dt,
         vx + ax * dt,
         vy + ay * dt],
        dtype=x.dtype,
    )


# -----------------------------
# MPC loop (records plans + tubes)
# -----------------------------
def run_mpc_loop_with_plans(
    *,
    x0: jnp.ndarray,
    x_ref: jnp.ndarray,
    T: int,
    dt: float,
    cfg: ADMMConfig,
    sls_config: SLSConfig,
    A: jnp.ndarray,
    B: jnp.ndarray,
    c: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray,
    f: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    M: jnp.ndarray,
    E: jnp.ndarray,
    n_steps: int,
    dtype=jnp.float32,
):
    nx, nu = 4, 2
    nc = C.shape[1]

    x_hist = [x0]
    u_hist = []
    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    for t in range(n_steps):
        x_cur = x_hist[-1]

        q = jnp.einsum("kij,j->ki", -Q, x_ref.astype(dtype))

        # NOTE: if your formulation needs explicit x0 constraints, replace this accordingly
        c_mpc = c.at[0].set(x_cur)

        w = jnp.zeros((T + 1, nc), dtype=dtype)
        y = jnp.zeros((T + 1, nc), dtype=dtype)
        rho = jnp.array(1.0, dtype=dtype)

        xN, uN, vN, w, y, rho, convergedN, converged_admm, h_ct = fast_sls_solve_gpu(
            cfg,
            Q, q,
            R, jnp.zeros((T, nu), dtype=dtype),
            M,
            A, B, c_mpc,
            C, D, f,
            w, y, rho,
            sls_config,
            E,
        )

        # Tube bounds (xy only). Assumes h_ct has at least 2 dims and aligned with xN over time.
        # If h_ct is [T+1, nx], this is correct. If it's [T+1, nc], you must map it appropriately.
        h_xy = h_ct[:, :2]
        # jax.debug.print("h_xy: {}", h_xy)
        lower = xN[:, :2] - h_xy
        upper = xN[:, :2] + h_xy

        plans_xy.append(xN[:, 0:2])
        lowers_xy.append(lower)
        uppers_xy.append(upper)

        u0 = uN[0]
        x_next = step_double_integrator(x_cur, u0, dt)

        u_hist.append(u0)
        x_hist.append(x_next)

        if not bool(jnp.all(jnp.isfinite(x_next))):
            print(f"Non-finite state at MPC step {t}; stopping.")
            break

    x_hist = jnp.stack(x_hist, axis=0)                    # [N+1, 4]
    u_hist = jnp.stack(u_hist, axis=0)                    # [N, 2]
    plans_xy = jnp.stack(plans_xy, axis=0)                # [N, T+1, 2]
    lowers_xy = jnp.stack(lowers_xy, axis=0)              # [N, T+1, 2]
    uppers_xy = jnp.stack(uppers_xy, axis=0)              # [N, T+1, 2]
    return x_hist, u_hist, plans_xy, lowers_xy, uppers_xy


# -----------------------------
# Video rendering (MP4) with green tube shading
# -----------------------------
def make_mpc_plan_video_mp4(
    x_hist: jnp.ndarray,
    plans_xy: jnp.ndarray,
    lowers_xy: jnp.ndarray,
    uppers_xy: jnp.ndarray,
    px_max: float,
    py_max: float,
    out_path: str = "mpc_plans.mp4",
    fps: int = 10,
):
    """
    Each frame t shows:
      - executed trajectory up to t
      - planned trajectory at time t
      - a green shaded tube region between lower/upper around the planned trajectory
      - XY constraint rectangle
    """
    x_hist_np = np.array(jax.device_get(x_hist))          # [N+1, 4]
    plans_np = np.array(jax.device_get(plans_xy))         # [N, T+1, 2]
    lowers_np = np.array(jax.device_get(lowers_xy))       # [N, T+1, 2]
    uppers_np = np.array(jax.device_get(uppers_xy))       # [N, T+1, 2]
    n_steps = plans_np.shape[0]

    # Limits
    all_px = np.concatenate([x_hist_np[:, 0], plans_np[:, :, 0].ravel(), lowers_np[:, :, 0].ravel(), uppers_np[:, :, 0].ravel()])
    all_py = np.concatenate([x_hist_np[:, 1], plans_np[:, :, 1].ravel(), lowers_np[:, :, 1].ravel(), uppers_np[:, :, 1].ravel()])
    x_min = min(all_px.min(), -px_max) - 0.5
    x_max = max(all_px.max(),  px_max) + 0.5
    y_min = min(all_py.min(), -py_max) - 0.5
    y_max = max(all_py.max(),  py_max) + 0.5

    fig, ax = plt.subplots()

    # Constraint box
    rect = Rectangle((-px_max, -py_max), 2.0 * px_max, 2.0 * py_max, fill=False, linewidth=2, label="XY box constraint")
    ax.add_patch(rect)

    executed_line, = ax.plot([], [], linewidth=2, label="Executed (closed-loop)")
    planned_line,  = ax.plot([], [], linewidth=2, linestyle="--", label="Planned (open-loop)")
    cur_pt = ax.scatter([], [], marker="o", label="Current state")
    end_pt = ax.scatter([], [], marker="x", label="End of plan")

    # Tube polygon (green shaded). We create one PolyCollection via fill() and then update its vertices.
    tube_poly = ax.fill([], [], alpha=0.25, label="Tube (lower/upper)")[0]

    title = ax.set_title("MPC Plans + Tube Over Time")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    ax.grid(True)
    ax.legend(loc="best")

    def init():
        executed_line.set_data([], [])
        planned_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        end_pt.set_offsets(np.zeros((0, 2)))
        tube_poly.set_xy(np.zeros((0, 2)))
        title.set_text("MPC Plans + Tube Over Time")
        return executed_line, planned_line, cur_pt, end_pt, tube_poly, title

    def update(t):
        # executed
        ex_px = x_hist_np[: t + 1, 0]
        ex_py = x_hist_np[: t + 1, 1]
        executed_line.set_data(ex_px, ex_py)

        # planned
        pl_px = plans_np[t, :, 0]
        pl_py = plans_np[t, :, 1]
        planned_line.set_data(pl_px, pl_py)

        # tube bounds
        lo_px = lowers_np[t, :, 0]
        lo_py = lowers_np[t, :, 1]
        up_px = uppers_np[t, :, 0]
        up_py = uppers_np[t, :, 1]

        # Build a closed polygon: go along upper in order, then along lower in reverse order
        poly_x = np.concatenate([up_px, lo_px[::-1]])
        poly_y = np.concatenate([up_py, lo_py[::-1]])
        tube_poly.set_xy(np.stack([poly_x, poly_y], axis=1))

        # markers
        cur_pt.set_offsets(np.array([[x_hist_np[t, 0], x_hist_np[t, 1]]]))
        end_pt.set_offsets(np.array([[pl_px[-1], pl_py[-1]]]))

        title.set_text(f"MPC step {t}/{n_steps-1}")
        return executed_line, planned_line, cur_pt, end_pt, tube_poly, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        blit=True,
        interval=int(1000 / fps),
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=200)
    plt.close(fig)
    print(f"Saved video to: {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    dt = 0.1
    T = 100
    dtype = jnp.float32

    x_ref = jnp.array([4.0, 1.0, 0.0, 0.0], dtype=dtype)

    px_max = 5.0
    py_max = 1.2
    vx_max = 5.0
    vy_max = 5.0
    ux_max = 5.0
    uy_max = 5.0

    A, B, c = build_ltv_dynamics(T, dt, dtype=dtype)
    C, D, f = make_box_constraints(T, px_max, py_max, vx_max, vy_max, ux_max, uy_max, dtype=dtype)
    Q, q, R, r, M = make_quadratic_cost(T, x_ref=x_ref, dtype=dtype)
    E = build_disturbance_scale(T, sigma=0.05, dtype=dtype)

    sls_config = SLSConfig(
        max_sls_iterations=10,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
    )

    cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        rho_max=1e5,
    )

    x0 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
    n_steps = 30

    x_hist, u_hist, plans_xy, lowers_xy, uppers_xy = run_mpc_loop_with_plans(
        x0=x0,
        x_ref=x_ref,
        T=T,
        dt=dt,
        cfg=cfg,
        sls_config=sls_config,
        A=A,
        B=B,
        c=c,
        C=C,
        D=D,
        f=f,
        Q=Q,
        R=R,
        M=M,
        E=E,
        n_steps=n_steps,
        dtype=dtype,
    )

    make_mpc_plan_video_mp4(
        x_hist=x_hist,
        plans_xy=plans_xy,
        lowers_xy=lowers_xy,
        uppers_xy=uppers_xy,
        px_max=px_max,
        py_max=py_max,
        out_path="mpc_plans.mp4",
        fps=10,
    )

    print("x_final:", x_hist[-1])
    print("max |u|:", float(jnp.max(jnp.abs(u_hist))))


if __name__ == "__main__":
    main()
