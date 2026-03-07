# visualization/npz_replay/plot_traj.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class PlotConfig:
    equal_aspect: bool = True
    show: bool = True
    save_path: str | None = None


def load_npz(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _set_equal_3d(ax):
    """Make 3D axes have equal scale (approx)."""
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


def plot_trajectory(npz_path: str | Path, cfg: PlotConfig = PlotConfig()):
    """
    Plot 3D trajectories in NED world frame:
      x: North, y: East, z: Down
    """
    data = load_npz(npz_path)
    uav_p = np.asarray(data["uav_p"])
    tgt_p = np.asarray(data["tgt_p"])
    dist = np.asarray(data.get("dist", None)) if "dist" in data else None
    t = np.asarray(data.get("t", None)) if "t" in data else None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(uav_p[:, 0], uav_p[:, 1], uav_p[:, 2], label="UAV")
    ax.plot(tgt_p[:, 0], tgt_p[:, 1], tgt_p[:, 2], label="Target")

    ax.scatter([uav_p[0, 0]], [uav_p[0, 1]], [uav_p[0, 2]], marker="o", label="UAV start")
    ax.scatter([tgt_p[0, 0]], [tgt_p[0, 1]], [tgt_p[0, 2]], marker="o", label="Target start")
    ax.scatter([uav_p[-1, 0]], [uav_p[-1, 1]], [uav_p[-1, 2]], marker="x", label="UAV end")
    ax.scatter([tgt_p[-1, 0]], [tgt_p[-1, 1]], [tgt_p[-1, 2]], marker="x", label="Target end")

    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_zlabel("Down [m]")
    ax.set_title("Trajectories (NED)")
    ax.legend(loc="best")

    if cfg.equal_aspect:
        _set_equal_3d(ax)

    if dist is not None and t is not None:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(t, dist)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("||p_t - p_u|| [m]")
        ax2.set_title("Relative Distance")

    if cfg.save_path:
        out = Path(cfg.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
        if dist is not None and t is not None:
            fig2.savefig(out.with_name(out.stem + "_dist.png"), dpi=150, bbox_inches="tight")

    if cfg.show:
        plt.show()

    return fig

