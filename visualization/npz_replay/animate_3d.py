# visualization/npz_replay/animate_3d.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass(slots=True)
class AnimConfig:
    stride: int = 5
    interval_ms: int = 33
    tail: int = 200
    show_fov_ray: bool = True
    ray_len: float = 5.0
    equal_aspect: bool = True
    save_mp4: str | None = None
    show: bool = True


def load_npz(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _set_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def animate_3d(npz_path: str | Path, cfg: AnimConfig = AnimConfig()):
    """
    3D animation of UAV and target trajectories in NED.
    Uses uav_p, tgt_p, uav_q for optional camera forward ray.
    """
    data = load_npz(npz_path)
    uav_p_all = np.asarray(data["uav_p"])
    tgt_p_all = np.asarray(data["tgt_p"])
    uav_q_all = np.asarray(data.get("uav_q", None))

    idx = np.arange(0, uav_p_all.shape[0], cfg.stride)
    uav_p = uav_p_all[idx]
    tgt_p = tgt_p_all[idx]
    uav_q = uav_q_all[idx] if uav_q_all is not None else None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    all_p = np.vstack([uav_p_all, tgt_p_all])
    ax.set_xlim(np.min(all_p[:, 0]), np.max(all_p[:, 0]))
    ax.set_ylim(np.min(all_p[:, 1]), np.max(all_p[:, 1]))
    ax.set_zlim(np.min(all_p[:, 2]), np.max(all_p[:, 2]))

    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_zlabel("Down [m]")
    ax.set_title("3D Animation (NED)")

    (uav_line,) = ax.plot([], [], [], lw=2, label="UAV trail")
    (tgt_line,) = ax.plot([], [], [], lw=2, label="Target trail")
    (uav_pt,) = ax.plot([], [], [], marker="o", markersize=6, linestyle="None", label="UAV")
    (tgt_pt,) = ax.plot([], [], [], marker="o", markersize=6, linestyle="None", label="Target")

    (ray_line,) = ax.plot([], [], [], lw=2, label="Camera forward ray") if cfg.show_fov_ray else (None,)

    ax.legend(loc="best")

    if cfg.equal_aspect:
        _set_equal_3d(ax)

    def cam_forward_world(q_eb: np.ndarray) -> np.ndarray:
        from utils.math3d import quat_to_R

        r = quat_to_R(q_eb)
        return r @ np.array([1.0, 0.0, 0.0], dtype=float)

    def init():
        uav_line.set_data([], [])
        uav_line.set_3d_properties([])
        tgt_line.set_data([], [])
        tgt_line.set_3d_properties([])
        uav_pt.set_data([], [])
        uav_pt.set_3d_properties([])
        tgt_pt.set_data([], [])
        tgt_pt.set_3d_properties([])
        if ray_line is not None:
            ray_line.set_data([], [])
            ray_line.set_3d_properties([])
        return tuple(x for x in [uav_line, tgt_line, uav_pt, tgt_pt, ray_line] if x is not None)

    def update(frame: int):
        i = frame
        i0 = max(0, i - cfg.tail)

        uav_line.set_data(uav_p[i0:i, 0], uav_p[i0:i, 1])
        uav_line.set_3d_properties(uav_p[i0:i, 2])

        tgt_line.set_data(tgt_p[i0:i, 0], tgt_p[i0:i, 1])
        tgt_line.set_3d_properties(tgt_p[i0:i, 2])

        uav_pt.set_data([uav_p[i, 0]], [uav_p[i, 1]])
        uav_pt.set_3d_properties([uav_p[i, 2]])

        tgt_pt.set_data([tgt_p[i, 0]], [tgt_p[i, 1]])
        tgt_pt.set_3d_properties([tgt_p[i, 2]])

        if ray_line is not None and uav_q is not None:
            fwd = cam_forward_world(uav_q[i])
            p0 = uav_p[i]
            p1 = p0 + cfg.ray_len * fwd
            ray_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
            ray_line.set_3d_properties([p0[2], p1[2]])

        return tuple(x for x in [uav_line, tgt_line, uav_pt, tgt_pt, ray_line] if x is not None)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(uav_p),
        init_func=init,
        interval=cfg.interval_ms,
        blit=False,
    )

    if cfg.save_mp4:
        out = Path(cfg.save_mp4)
        out.parent.mkdir(parents=True, exist_ok=True)
        ani.save(out.as_posix(), dpi=150)

    if cfg.show:
        plt.show()

    return ani

