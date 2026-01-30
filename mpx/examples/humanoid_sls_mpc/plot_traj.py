#!/usr/bin/env python3
"""
replay_h1_plans_tubes.py

Replay (no MPC solving) of saved H1 logs showing:
  - executed xy (solve-to-solve)
  - planned xy per solve
  - tube rectangles computed by YOUR get_trajectory_tubes(Phi_x)
  - obstacle circles

Usage:
  python replay_h1_plans_tubes.py \
    --npz /path/to/humanoid_sls_mpc/h1_mpc_logs_.npz \
    --out replay.mp4 \
    --fps 20 \
    --box-stride 2
"""

import argparse
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# <-- your method
from mpx.utils.fast_sls_visual import get_trajectory_tubes


def save_replay(
    xs_xy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    plans_xy: np.ndarray,
    lowers_xy: np.ndarray,
    uppers_xy: np.ndarray,
    filename: str = "replay.mp4",
    dt: float = 0.02,
    fps: int | None = None,
    box_stride: int = 1,
    margin: float = 0.5,
    title_str: str = "H1 MPC Replay: Plans + Tubes",
):
    xs_xy = np.asarray(xs_xy)
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    plans_xy = np.asarray(plans_xy)
    lowers_xy = np.asarray(lowers_xy)
    uppers_xy = np.asarray(uppers_xy)

    n_steps = plans_xy.shape[0]
    if n_steps == 0:
        raise ValueError("plans_xy is empty; nothing to replay.")
    if fps is None:
        fps = max(1, int(round(1.0 / dt)))
    interval_ms = int(round(1000.0 / fps))

    xs_len = xs_xy.shape[0]

    all_px = np.concatenate([
        xs_xy[:, 0].ravel(),
        plans_xy[:, :, 0].ravel(),
        lowers_xy[:, :, 0].ravel(),
        uppers_xy[:, :, 0].ravel(),
        centers[:, 0].ravel() if centers.size else np.array([], dtype=float),
    ])
    all_py = np.concatenate([
        xs_xy[:, 1].ravel(),
        plans_xy[:, :, 1].ravel(),
        lowers_xy[:, :, 1].ravel(),
        uppers_xy[:, :, 1].ravel(),
        centers[:, 1].ravel() if centers.size else np.array([], dtype=float),
    ])

    xmin, xmax = float(all_px.min() - margin), float(all_px.max() + margin)
    ymin, ymax = float(all_py.min() - margin), float(all_py.max() + margin)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title_str)
    ax.grid(True)

    # obstacles
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle((float(c[0]), float(c[1])), float(r), alpha=0.35))

    executed_line, = ax.plot([], [], lw=2, alpha=0.9, label="Executed (solve-to-solve)")
    planned_line,  = ax.plot([], [], lw=2, ls="--", alpha=0.95, label="Planned (open-loop)")
    cur_pt = ax.scatter([], [], marker="o", s=45, label="Current")
    end_pt = ax.scatter([], [], marker="x", s=55, label="Plan end")

    tube_boxes = PatchCollection([], alpha=0.20, match_original=False, label="Tube boxes")
    ax.add_collection(tube_boxes)

    ax.legend(loc="lower left", framealpha=0.9)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def init():
        executed_line.set_data([], [])
        planned_line.set_data([], [])
        cur_pt.set_offsets(np.zeros((0, 2)))
        end_pt.set_offsets(np.zeros((0, 2)))
        tube_boxes.set_paths([])
        hud.set_text("")
        return executed_line, planned_line, cur_pt, end_pt, tube_boxes, hud

    def update(t: int):
        # executed up to solve t
        t_show = min(t, xs_len - 1)
        executed_line.set_data(xs_xy[: t_show + 1, 0], xs_xy[: t_show + 1, 1])

        # plan at solve t
        pl = plans_xy[t]
        planned_line.set_data(pl[:, 0], pl[:, 1])

        # tube boxes at solve t
        lo = lowers_xy[t]
        up = uppers_xy[t]

        rects = []
        stride = max(int(box_stride), 1)
        for k in range(0, lo.shape[0], stride):
            w = float(up[k, 0] - lo[k, 0])
            h = float(up[k, 1] - lo[k, 1])
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            rects.append(Rectangle((float(lo[k, 0]), float(lo[k, 1])), w, h))
        tube_boxes.set_paths(rects)

        # markers
        cur_pt.set_offsets(np.array([[xs_xy[t_show, 0], xs_xy[t_show, 1]]]))
        end_pt.set_offsets(np.array([[pl[-1, 0], pl[-1, 1]]]))

        hud.set_text(f"solve {t+1}/{n_steps}")
        return executed_line, planned_line, cur_pt, end_pt, tube_boxes, hud

    ani = animation.FuncAnimation(
        fig, update, frames=n_steps, init_func=init, blit=True, interval=interval_ms
    )

    ext = filename.lower().split(".")[-1]
    if ext == "mp4":
        if animation.FFMpegWriter.isAvailable():
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(filename, writer=writer, dpi=200)
        else:
            raise RuntimeError("Requested .mp4 but ffmpeg is not available. Install ffmpeg or save .gif.")
    elif ext == "gif":
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
    else:
        raise ValueError(f"Unsupported extension .{ext}. Use .mp4 or .gif.")

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to h1_mpc_logs_.npz")
    ap.add_argument("--out", default="replay.mp4", help="Output file (.mp4 or .gif)")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--dt", type=float, default=None,
                    help="Override dt per frame; default uses 1/mpc_frequency in npz")
    ap.add_argument("--box-stride", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.75)
    args = ap.parse_args()

    dat = np.load(args.npz, allow_pickle=True)
    X = dat["X"]          # (n_solves, N+1, nx)
    Phi_x = dat["Phi_x"]  # (n_solves, ...)

    obstacles = np.asarray(dat["obstacles"], dtype=np.float64) if "obstacles" in dat else None
    if obstacles is None:
        centers = np.zeros((0, 2), dtype=np.float64)
        radii = np.zeros((0,), dtype=np.float64)
    else:
        centers = obstacles[:, :2]
        radii = obstacles[:, 2]

    n_solves, N1, nx = X.shape

    # executed: current (solve-to-solve) state each solve
    xs_xy = X[:, 0, :2]      # (n_solves, 2)
    plans_xy = X[:, :, :2]   # (n_solves, N+1, 2)

    # dt for animation
    if args.dt is not None:
        dt_frame = float(args.dt)
    else:
        if "mpc_frequency" in dat:
            dt_frame = 1.0 / float(dat["mpc_frequency"])
        elif "dt" in dat:
            dt_frame = float(dat["dt"])
        else:
            dt_frame = 0.02

    # compute tubes using your method
    tubes_xy = np.zeros((n_solves, N1, 2), dtype=np.float64)
    for t in range(n_solves):
        tube = get_trajectory_tubes(Phi_x[t])  # your function
        tube = np.asarray(tube)
        if tube.ndim != 2 or tube.shape[0] != N1:
            raise ValueError(f"get_trajectory_tubes returned {tube.shape} at solve {t}, expected (N+1, d).")
        if tube.shape[1] < 2:
            raise ValueError(f"Tube dim is {tube.shape[1]} (<2). Need x/y bounds.")
        tubes_xy[t] = tube[:, :2]

    lowers_xy = plans_xy - tubes_xy
    uppers_xy = plans_xy + tubes_xy

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_replay(
        xs_xy=xs_xy,
        centers=centers,
        radii=radii,
        plans_xy=plans_xy,
        lowers_xy=lowers_xy,
        uppers_xy=uppers_xy,
        filename=args.out,
        dt=dt_frame,
        fps=args.fps,
        box_stride=args.box_stride,
        margin=args.margin,
        title_str="H1 SLS-MPC Replay: Plans + Tubes",
    )

    print(f"[ok] wrote {args.out}")
    print(f"      n_solves={n_solves}, horizon={N1-1}, nx={nx}, dt_frame={dt_frame:.4f}")
    print(f"      Phi_x[t] -> get_trajectory_tubes -> tubes_xy.shape={tubes_xy.shape}")


if __name__ == "__main__":
    main()
