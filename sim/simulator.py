from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from models.state import UAVState, TargetState, CameraMeasurement
from sim.scheduler import MultiRateScheduler
from utils.math3d import quat_to_R

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:  # pragma: no cover
    plt = None
    Poly3DCollection = None


@dataclass(slots=True)
class TerminationConfig:
    t_final: float = 20.0
    hit_radius: float = 0.5  # meters


class Simulator:
    """
    Real-time visualizer driven by scheduler timing.

    Main purpose:
    - Receive current UAV/Target states from the main simulation loop
    - Visualize UAV via vis_uav(...)
    - Visualize target via vis_target(...)
    - Only refresh when scheduler.should_visualization(step) is true
    """

    def __init__(
        self,
        scheduler: MultiRateScheduler,
        enable: bool = True,
        realtime: bool = True,
        enable_fov: bool = True,
        cam_width: int = 640,
        cam_height: int = 480,
        cam_fx: float = 320.0,
        cam_fy: float = 320.0,
        cam_cx: float = 320.0,
        cam_cy: float = 240.0,
        trail_len: int = 1000,
        auto_axis: bool = False,
        ned_axes: bool = True,
        map_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        map_size: tuple[float, float, float] = (160.0, 120.0, 80.0),
        colors: dict | None = None,
        uav_visual_scale: float = 1.0,
        target_marker_size: float = 6.0,
        fov_target_marker_size: float = 7.0,
    ):
        self.sch = scheduler
        self.realtime = bool(realtime)
        backend_is_noninteractive = False
        if plt is not None:
            backend = str(plt.get_backend()).lower()
            backend_is_noninteractive = "agg" in backend
            if enable and backend_is_noninteractive:
                print(f"[Simulator] visualization disabled: non-interactive matplotlib backend '{backend}'")
        self.enable = bool(enable and (plt is not None) and (Poly3DCollection is not None) and (not backend_is_noninteractive))
        self.enable_fov = bool(enable_fov)
        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)
        self.cam_fx = float(cam_fx)
        self.cam_fy = float(cam_fy)
        self.cam_cx = float(cam_cx)
        self.cam_cy = float(cam_cy)
        self.trail_len = max(10, int(trail_len))
        self.auto_axis = bool(auto_axis)
        self.ned_axes = bool(ned_axes)
        self.map_center = np.asarray(map_center, dtype=float).reshape(3)
        self.map_size = np.asarray(map_size, dtype=float).reshape(3)
        self.map_size = np.maximum(self.map_size, 1.0)
        self.colors = self._build_colors(colors or {})
        self.uav_visual_scale = float(uav_visual_scale)
        self.target_marker_size = float(target_marker_size)
        self.fov_target_marker_size = float(fov_target_marker_size)

        self._u_hist: list[np.ndarray] = []
        self._t_hist: list[np.ndarray] = []
        self._last_vis_wall_t: float | None = None

        self.fig = None
        self._layout_gs = None
        self.ax = None
        self._u_line = None
        self._t_line = None
        self._t_dot = None
        self._bbox_lines = []
        self._u_arm1 = None
        self._u_arm2 = None
        self._u_forward = None
        self._u_forward_h1 = None
        self._u_forward_h2 = None
        self._u_rotor_disks = []
        self.ax_fov = None
        self._fov_target_dot = None
        self._fov_status_text = None

        if self.enable:
            plt.ion()
            self._init_canvas()

    @staticmethod
    def _parse_color(value, default):
        if value is None:
            return default
        if isinstance(value, str):
            return value
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size not in (3, 4):
            raise ValueError(f"color must have 3 or 4 components, got shape {arr.shape}")
        if np.max(arr) > 1.0:
            arr = arr / 255.0
        return tuple(float(x) for x in arr)

    def _build_colors(self, cfg: dict) -> dict:
        defaults = {
            "uav_traj": (158.0 / 255.0, 22.0 / 255.0, 157.0 / 255.0, 0.55),
            "uav_body_left": (22.0 / 255.0, 5.0 / 255.0, 139.0 / 255.0),
            "uav_body_right": (98.0 / 255.0, 0.0, 170.0 / 255.0),
            "uav_arrow": (204.0 / 255.0, 74.0 / 255.0, 116.0 / 255.0),
            "target_traj": (252.0 / 255.0, 180.0 / 255.0, 49.0 / 255.0),
            "target_dot": (235.0 / 255.0, 120.0 / 255.0, 82.0 / 255.0),
            "fov_target": (181.0 / 255.0, 35.0 / 255.0, 4.0 / 255.0),
            "fov_status_no_target": (181.0 / 255.0, 35.0 / 255.0, 4.0 / 255.0),
            "fov_status_has_target": "green",
        }
        return {k: self._parse_color(cfg.get(k), v) for k, v in defaults.items()}

    def _init_canvas(self) -> None:
        if self.enable_fov:
            self.fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            self._layout_gs = self.fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
            self.ax = self.fig.add_subplot(self._layout_gs[0, 0], projection="3d")
        else:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m] (NED)")
        self.ax.set_title("Real-time UAV/Target Visualization (NED)")

        uav_traj_color = self.colors["uav_traj"]
        traj_alpha = 1.0 if len(uav_traj_color) < 4 else uav_traj_color[3]
        traj_rgb = uav_traj_color[:3]
        (self._u_line,) = self.ax.plot([], [], [], lw=0.9, alpha=traj_alpha, color=traj_rgb, label="UAV Traj")
        target_traj_kwargs = {"lw": 1.5, "label": "Target Traj"}
        if self.colors["target_traj"] is not None:
            target_traj_kwargs["color"] = self.colors["target_traj"]
        (self._t_line,) = self.ax.plot([], [], [], **target_traj_kwargs)
        (self._t_dot,) = self.ax.plot(
            [],
            [],
            [],
            marker="o",
            ms=self.target_marker_size,
            color=self.colors["target_dot"],
            linestyle="None",
        )
        (self._u_arm1,) = self.ax.plot([], [], [], color=self.colors["uav_body_left"], lw=2.0)
        (self._u_arm2,) = self.ax.plot([], [], [], color=self.colors["uav_body_right"], lw=2.0)
        (self._u_forward,) = self.ax.plot([], [], [], color=self.colors["uav_arrow"], lw=1.4)
        (self._u_forward_h1,) = self.ax.plot([], [], [], color=self.colors["uav_arrow"], lw=1.2)
        (self._u_forward_h2,) = self.ax.plot([], [], [], color=self.colors["uav_arrow"], lw=1.2)
        self._u_rotor_disks = []
        rotor_colors = [
            self.colors["uav_body_right"],
            self.colors["uav_body_left"],
            self.colors["uav_body_left"],
            self.colors["uav_body_right"],
        ]
        for rotor_color in rotor_colors:
            disk = Poly3DCollection(
                [],
                facecolors=rotor_color,
                edgecolors=rotor_color,
                linewidths=0.6,
                alpha=0.95,
            )
            self.ax.add_collection3d(disk)
            self._u_rotor_disks.append(disk)
        self.ax.legend(loc="best")
        self._apply_fixed_bounds()
        self._init_bbox_lines()
        self._update_bbox_lines_from_axes()
        if self.enable_fov:
            self._init_fov_canvas()
        # Ensure window is created immediately so subsequent updates are visible in real time.
        plt.show(block=False)
        plt.pause(0.001)

    def _init_fov_canvas(self) -> None:
        self.ax_fov = self.fig.add_subplot(self._layout_gs[0, 1])
        self.ax_fov.set_title("Camera FOV (First-person)")
        self.ax_fov.set_xlabel("horizontal angle [deg]")
        self.ax_fov.set_ylabel("vertical angle [deg]")

        half_hfov_deg = float(np.degrees(np.arctan((0.5 * self.cam_width) / max(self.cam_fx, 1e-9))))
        half_vfov_deg = float(np.degrees(np.arctan((0.5 * self.cam_height) / max(self.cam_fy, 1e-9))))

        self.ax_fov.set_xlim(-half_hfov_deg, half_hfov_deg)
        self.ax_fov.set_ylim(half_vfov_deg, -half_vfov_deg)
        self.ax_fov.set_aspect("equal", adjustable="box")
        self.ax_fov.margins(x=0.04, y=0.04)
        self.ax_fov.grid(True, color="0.82", linewidth=0.8)

        border_x = [-half_hfov_deg, half_hfov_deg, half_hfov_deg, -half_hfov_deg, -half_hfov_deg]
        border_y = [-half_vfov_deg, -half_vfov_deg, half_vfov_deg, half_vfov_deg, -half_vfov_deg]
        self.ax_fov.plot(border_x, border_y, color="black", lw=1.4)

        # Red target marker
        (self._fov_target_dot,) = self.ax_fov.plot(
            [],
            [],
            marker="o",
            markersize=self.fov_target_marker_size,
            color=self.colors["fov_target"],
            linestyle="None",
        )
        self._fov_target_dot.set_visible(False)
        self._fov_status_text = self.ax_fov.text(
            0.02,
            0.98,
            "no target",
            transform=self.ax_fov.transAxes,
            ha="left",
            va="top",
            color=self.colors["fov_status_no_target"],
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.85, boxstyle="round,pad=0.25"),
        )

    def _apply_fixed_bounds(self) -> None:
        hx, hy, hz = 0.5 * self.map_size
        cx, cy, cz = self.map_center
        self.ax.set_xlim(cx - hx, cx + hx)
        self.ax.set_ylim(cy - hy, cy + hy)
        if self.ned_axes:
            self.ax.set_zlim(cz + hz, cz - hz)  # invert for NED "Down positive"
        else:
            self.ax.set_zlim(cz - hz, cz + hz)
        self.ax.set_box_aspect((float(self.map_size[0]), float(self.map_size[1]), float(self.map_size[2])))

    def _init_bbox_lines(self) -> None:
        # 12 edges of a box (wireframe only)
        self._bbox_lines = []
        for _ in range(12):
            (line,) = self.ax.plot([], [], [], color="0.35", lw=1.0, alpha=0.9)
            self._bbox_lines.append(line)

    def _update_bbox_lines_from_axes(self) -> None:
        if not self._bbox_lines:
            return

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        z0, z1 = self.ax.get_zlim()
        zmin, zmax = min(z0, z1), max(z0, z1)

        p000 = (xmin, ymin, zmin)
        p001 = (xmin, ymin, zmax)
        p010 = (xmin, ymax, zmin)
        p011 = (xmin, ymax, zmax)
        p100 = (xmax, ymin, zmin)
        p101 = (xmax, ymin, zmax)
        p110 = (xmax, ymax, zmin)
        p111 = (xmax, ymax, zmax)

        edges = [
            (p000, p100), (p010, p110), (p001, p101), (p011, p111),  # x-direction
            (p000, p010), (p100, p110), (p001, p011), (p101, p111),  # y-direction
            (p000, p001), (p100, p101), (p010, p011), (p110, p111),  # z-direction
        ]

        for line, (a, b) in zip(self._bbox_lines, edges):
            line.set_data([a[0], b[0]], [a[1], b[1]])
            line.set_3d_properties([a[2], b[2]])

    def _trim_hist(self) -> None:
        if len(self._u_hist) > self.trail_len:
            self._u_hist = self._u_hist[-self.trail_len :]
        if len(self._t_hist) > self.trail_len:
            self._t_hist = self._t_hist[-self.trail_len :]

    def _update_bounds(self) -> None:
        pts = []
        if self._u_hist:
            pts.append(np.vstack(self._u_hist))
        if self._t_hist:
            pts.append(np.vstack(self._t_hist))
        if not pts:
            return

        all_xyz = np.vstack(pts)
        mins = all_xyz.min(axis=0)
        maxs = all_xyz.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pads = 0.05 * spans

        self.ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
        self.ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
        if self.ned_axes:
            self.ax.set_zlim(maxs[2] + pads[2], mins[2] - pads[2])
        else:
            self.ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
        self.ax.set_box_aspect((float(spans[0] + 2.0 * pads[0]), float(spans[1] + 2.0 * pads[1]), float(spans[2] + 2.0 * pads[2])))

    @staticmethod
    def _set_line3d(line, p0: np.ndarray, p1: np.ndarray) -> None:
        line.set_data([float(p0[0]), float(p1[0])], [float(p0[1]), float(p1[1])])
        line.set_3d_properties([float(p0[2]), float(p1[2])])

    def vis_uav(self, uav: UAVState) -> None:
        if not self.enable:
            return
        self._u_hist.append(np.asarray(uav.p_e, dtype=float).copy())
        self._trim_hist()

        up = np.vstack(self._u_hist)
        self._u_line.set_data(up[:, 0], up[:, 1])
        self._u_line.set_3d_properties(up[:, 2])

        # Draw UAV body with actual attitude (q_eb: body->world, FRD body frame).
        p = np.asarray(uav.p_e, dtype=float).reshape(3)
        R = quat_to_R(uav.q_eb)

        arm_len = 0.9 * self.uav_visual_scale
        diag = arm_len / np.sqrt(2.0)
        arrow_len = 2.24 * arm_len
        arrow_head = 0.35 * arm_len

        # Blue X-body (two arms in the body x-y plane).
        b_a1 = np.array([diag, diag, 0.0], dtype=float)
        b_a2 = np.array([-diag, -diag, 0.0], dtype=float)
        b_b1 = np.array([diag, -diag, 0.0], dtype=float)
        b_b2 = np.array([-diag, diag, 0.0], dtype=float)

        e_a1 = p + R @ b_a1
        e_a2 = p + R @ b_a2
        e_b1 = p + R @ b_b1
        e_b2 = p + R @ b_b2

        self._set_line3d(self._u_arm1, e_a1, e_a2)
        self._set_line3d(self._u_arm2, e_b1, e_b2)

        # Four blue rotor disks (filled circles in UAV body plane).
        rotors = np.vstack([e_a1, e_a2, e_b1, e_b2])
        rotor_r = 0.48 * arm_len
        e_x = R @ np.array([1.0, 0.0, 0.0], dtype=float)
        e_y = R @ np.array([0.0, 1.0, 0.0], dtype=float)
        th = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        cth = np.cos(th)
        sth = np.sin(th)
        for disk, c in zip(self._u_rotor_disks, rotors):
            verts = [c + rotor_r * (cth_i * e_x + sth_i * e_y) for cth_i, sth_i in zip(cth, sth)]
            disk.set_verts([verts])

        # Red forward arrow along +x_b, rotated into world.
        b_f = np.array([1.0, 0.0, 0.0], dtype=float)
        b_r = np.array([0.0, 1.0, 0.0], dtype=float)
        e_f = R @ b_f
        e_r = R @ b_r

        tip = p + arrow_len * e_f
        self._set_line3d(self._u_forward, p, tip)

        h_base = tip - arrow_head * e_f
        h1 = h_base + 0.5 * arrow_head * e_r
        h2 = h_base - 0.5 * arrow_head * e_r
        self._set_line3d(self._u_forward_h1, tip, h1)
        self._set_line3d(self._u_forward_h2, tip, h2)

    def vis_target(self, tgt: TargetState) -> None:
        if not self.enable:
            return
        self._t_hist.append(np.asarray(tgt.p_e, dtype=float).copy())
        self._trim_hist()

        tp = np.vstack(self._t_hist)
        self._t_line.set_data(tp[:, 0], tp[:, 1])
        self._t_line.set_3d_properties(tp[:, 2])

        self._t_dot.set_data([tp[-1, 0]], [tp[-1, 1]])
        self._t_dot.set_3d_properties([tp[-1, 2]])

    def vis_fov(self, cam: CameraMeasurement | None, has_target: bool | None = None) -> None:
        if (not self.enable) or (not self.enable_fov) or (self._fov_target_dot is None):
            return
        if self._fov_status_text is not None:
            status = bool(has_target) if has_target is not None else bool(cam is not None and cam.valid)
            self._fov_status_text.set_text("has target" if status else "no target")
            self._fov_status_text.set_color(self.colors["fov_status_has_target"] if status else self.colors["fov_status_no_target"])
        if cam is None or (cam.p_norm is None):
            self._fov_target_dot.set_visible(False)
            return

        x_deg = float(np.degrees(np.arctan(cam.p_norm[0])))
        y_deg = float(np.degrees(np.arctan(cam.p_norm[1])))
        self._fov_target_dot.set_data([x_deg], [y_deg])
        self._fov_target_dot.set_visible(True)

    def update(
        self,
        step: int,
        uav: UAVState,
        tgt: TargetState,
        cam: CameraMeasurement | None = None,
        has_target: bool | None = None,
    ) -> None:
        if not self.enable:
            return
        if not self.sch.should_visualization(step):
            return

        if self.realtime:
            # Pace visualization to wall-clock time so users can observe the process.
            vis_interval = self.sch.dt * self.sch.n_visualization
            now = time.perf_counter()
            if self._last_vis_wall_t is not None:
                wait_s = vis_interval - (now - self._last_vis_wall_t)
                if wait_s > 0.0:
                    time.sleep(wait_s)
            self._last_vis_wall_t = time.perf_counter()

        self.vis_uav(uav)
        self.vis_target(tgt)
        self.vis_fov(cam, has_target=has_target)
        if self.auto_axis:
            self._update_bounds()
        else:
            self._apply_fixed_bounds()
        self._update_bbox_lines_from_axes()
        self.ax.set_title(f"Real-time UAV/Target Visualization | t={uav.t:.2f}s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.enable_fov and (self.ax_fov is not None):
            self.ax_fov.set_title(f"Camera FOV (First-person) | t={uav.t:.2f}s")
        plt.pause(0.001)

    def close(self, block: bool = True) -> None:
        if not self.enable:
            return
        if block:
            plt.ioff()
            plt.show()
        else:
            plt.close(self.fig)
