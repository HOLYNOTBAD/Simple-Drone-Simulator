# control/vpn_acc_controller.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from control.controller_base import ControllerBase
from models.state import ControlCommand, Observation
from utils.math3d import clamp_norm, quat_to_R, quat_to_euler_ZYX

if TYPE_CHECKING:
    from visualization.monitor import Monitor


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _vex(s: np.ndarray) -> np.ndarray:
    return np.array([s[2, 1], s[0, 2], s[1, 0]], dtype=float)


@dataclass(slots=True)
class VpnAccControllerParams:
    mass: float = 0.5
    g: float = 9.81

    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0

    bbox_width_ref_px: float = 120.0
    bbox_filter_memory: float = 0.7
    fallback_scale: float = 0.45
    s_min: float = 0.25
    s_max: float = 0.7

    k1: float = 1.0
    v_d0: float = 20.0
    alpha: float = 0.6

    k2: float = 0.6
    nav_gain: float = 4.0
    tau: float = 0.15

    k3: float = 0.6
    sigma_lo: float = 0.3
    sigma_hi: float = 0.5

    accel_max: float = 12.0
    yaw_rate_far_gain: float = 1.2
    yaw_rate_near_gain: float = 0.6
    yaw_d: float = np.pi / 2.0

    att_kp: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 2.0], dtype=float))
    omega_max: float = 6.0
    thrust_min: float = 0.0
    thrust_max: float = 30.0
    eps: float = 1e-6


class VpnAccController(ControllerBase):
    """
    Velocity-pursuit + proportional-navigation acceleration controller.

    This file keeps two interfaces:

    1. `compute_from_legacy(...)`
       A near-direct port of the original `VpnAccelerationController(...)`
       routine. It returns `[ax, ay, az, 0, 0, 0, yaw_rate]`.

    2. `compute(obs)`
       Framework-facing adapter that converts the acceleration demand into the
       project's `ControlCommand(thrust, omega_cmd_b)` format.

    Notes:
    - The original guidance law needs target image center and bbox width.
    - Current `Observation` only provides `p_norm`, so `compute(obs)` uses the
      current normalized feature and an internal scale fallback when bbox width
      is unavailable.
    - If you later extend `Observation` with bbox width, pass it to
      `compute_accel_command(...)` directly for full behavior.
    """

    def __init__(self, p: VpnAccControllerParams):
        self.p = p
        self.monitor: Monitor | None = None
        self._monitor_step = 0
        self._bw_f = float(self.p.bbox_width_ref_px)
        self._n_delta_vel_f = np.zeros(3, dtype=float)
        self._last_p_norm: np.ndarray | None = None
        self._last_t: float | None = None
        self._R_b_c = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        self._n_cc = np.array([0.0, 0.0, 1.0], dtype=float)

    def reset(self) -> None:
        self._monitor_step = 0
        self._bw_f = float(self.p.bbox_width_ref_px)
        self._n_delta_vel_f[:] = 0.0
        self._last_p_norm = None
        self._last_t = None

    def set_monitor(self, monitor: "Monitor | None") -> None:
        self.monitor = monitor

    def _push_monitor(self, name: str, color: str, data: float, group: str, t: float, step: int) -> None:
        if self.monitor is None:
            return
        self.monitor.push(name=name, color=color, data=float(data), t=t, group=group, step=step)

    @staticmethod
    def _smooth_gate(x: float) -> float:
        x_clip = float(np.clip(x, 0.0, 1.0))
        return x_clip * x_clip * (3.0 - 2.0 * x_clip)

    def _sigma(self, v: np.ndarray, lo: float, hi: float) -> float:
        n = float(np.linalg.norm(v))
        if hi <= lo + self.p.eps:
            return 1.0 if n > lo else 0.0
        return self._smooth_gate((n - lo) / (hi - lo))

    def _sat(self, v: np.ndarray, max_norm: float) -> np.ndarray:
        return clamp_norm(np.asarray(v, dtype=float).reshape(3), max_norm)

    def _pixels_to_los_earth(self, R_e_b: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
        n_co = np.array(
            [
                float(uv_px[0]) - self.p.cx,
                float(uv_px[1]) - self.p.cy,
                self.p.fx,
            ],
            dtype=np.float64,
        )
        n_co = _normalize(n_co, self.p.eps)
        n_bo = self._R_b_c @ n_co
        return R_e_b @ n_bo

    def _scale_from_width(self, bbox_width_px: float | None) -> tuple[float, float, float]:
        if bbox_width_px is None:
            bw = float(self.p.fallback_scale * self.p.bbox_width_ref_px)
        else:
            bw = float(max(bbox_width_px, self.p.eps))

        a = float(np.clip(self.p.bbox_filter_memory, 0.0, 1.0))
        self._bw_f = a * self._bw_f + (1.0 - a) * bw

        s_raw = self._bw_f / max(float(self.p.bbox_width_ref_px), self.p.eps)
        s = float(np.clip(s_raw, self.p.s_min, self.p.s_max))
        gamma = float(np.clip((self.p.s_max - s) / max(self.p.s_max - self.p.s_min, self.p.eps), 0.0, 1.0))
        return bw, s, gamma

    def _guidance_core(
        self,
        *,
        R_e_b: np.ndarray,
        mav_vel_e: np.ndarray,
        uv_px: np.ndarray,
        bbox_width_px: float | None,
        uv_px_last: np.ndarray | None,
        dt: float,
        t: float,
        step: int,
    ) -> np.ndarray:
        n_eo = self._pixels_to_los_earth(R_e_b, uv_px)
        n_delta = np.zeros(3, dtype=float)
        n_delta_vel = np.zeros(3, dtype=float)

        if uv_px_last is not None and dt > 1e-3:
            n_eol = self._pixels_to_los_earth(R_e_b, uv_px_last)
            n_delta = n_eo - n_eol
            if dt > 0.0:
                n_delta_vel = n_delta / dt

        _, s, gamma = self._scale_from_width(bbox_width_px)

        v_d = self.p.v_d0 * (1.0 - self.p.alpha * s)
        a_v = self.p.k1 * (v_d * n_eo - mav_vel_e)

        V = float(np.linalg.norm(mav_vel_e))
        alpha_f = float(np.exp(-max(dt, 0.0) / max(self.p.tau, self.p.eps)))
        self._n_delta_vel_f = alpha_f * self._n_delta_vel_f + (1.0 - alpha_f) * n_delta_vel
        a_vpn = self.p.k2 * gamma * self.p.nav_gain * V * self._n_delta_vel_f

        n_bc = self._R_b_c @ self._n_cc
        n_ec = R_e_b @ n_bc
        n_c = _normalize(n_ec, self.p.eps)
        e_n = n_eo - float(np.dot(n_eo, n_c)) * n_c
        a_los = -self.p.k3 * s * V * self._sigma(e_n, self.p.sigma_lo, self.p.sigma_hi) * e_n

        a_d = self._sat(a_v + a_vpn + a_los, self.p.accel_max)

        yaw_err = (self.p.cx - float(uv_px[0])) / max(self.p.cx, self.p.eps)
        yaw_rate_far = self.p.yaw_rate_far_gain * yaw_err
        yaw_rate_near = self.p.yaw_rate_near_gain * yaw_err
        yaw_rate = (1.0 - gamma) * yaw_rate_near + gamma * yaw_rate_far

        self._push_monitor("vpn_s", "tab:blue", s, "vpn_scale", t, step)
        self._push_monitor("vpn_gamma", "tab:orange", gamma, "vpn_scale", t, step)
        self._push_monitor("vpn_ax", "tab:red", a_d[0], "vpn_accel", t, step)
        self._push_monitor("vpn_ay", "tab:green", a_d[1], "vpn_accel", t, step)
        self._push_monitor("vpn_az", "tab:purple", a_d[2], "vpn_accel", t, step)
        self._push_monitor("vpn_yaw_rate", "black", yaw_rate, "vpn_yaw", t, step)

        return np.array([a_d[0], a_d[1], a_d[2], 0.0, 0.0, 0.0, yaw_rate], dtype=float)

    def compute_from_legacy(
        self,
        pos_info: dict,
        pos_i: np.ndarray,
        pos_i_last: np.ndarray | None = None,
        dt: float = 0.0,
        controller_reset: bool = False,
        yaw_d: float | None = None,
    ) -> np.ndarray:
        """
        Direct adapter for legacy code.

        Parameters mirror the original function:
        - `pos_info["mav_R"]`: body->earth rotation matrix
        - `pos_info["mav_vel"]`: UAV velocity in earth frame
        - `pos_i`: `[u, v, bbox_width, ...]`
        - `pos_i_last`: previous image measurement in the same layout
        """
        if controller_reset:
            self.reset()
        if yaw_d is not None:
            self.p.yaw_d = float(yaw_d)

        R_e_b = np.asarray(pos_info["mav_R"], dtype=float).reshape(3, 3)
        mav_vel_e = np.asarray(pos_info["mav_vel"], dtype=float).reshape(3)

        pos_i_arr = np.asarray(pos_i, dtype=float).reshape(-1)
        if pos_i_arr.shape[0] < 3:
            raise ValueError("pos_i must contain at least [u, v, bbox_width]")

        uv_px = pos_i_arr[:2]
        bbox_width_px = float(pos_i_arr[2])
        uv_px_last = None if pos_i_last is None else np.asarray(pos_i_last, dtype=float).reshape(-1)[:2]

        return self._guidance_core(
            R_e_b=R_e_b,
            mav_vel_e=mav_vel_e,
            uv_px=uv_px,
            bbox_width_px=bbox_width_px,
            uv_px_last=uv_px_last,
            dt=float(dt),
            t=0.0,
            step=self._monitor_step,
        )

    def compute_accel_command(
        self,
        obs: Observation,
        *,
        bbox_width_px: float | None = None,
        p_norm_last: np.ndarray | None = None,
        dt: float = 0.0,
    ) -> np.ndarray:
        if (not obs.has_target) or (obs.p_norm is None):
            return np.zeros(7, dtype=float)

        R_e_b = quat_to_R(obs.q_eb)
        mav_vel_e = np.asarray(obs.v_e, dtype=float).reshape(3)
        uv_px = np.array(
            [
                self.p.cx + self.p.fx * float(obs.p_norm[0]),
                self.p.cy + self.p.fy * float(obs.p_norm[1]),
            ],
            dtype=float,
        )

        uv_px_last = None
        if p_norm_last is not None:
            p_norm_last_arr = np.asarray(p_norm_last, dtype=float).reshape(2)
            uv_px_last = np.array(
                [
                    self.p.cx + self.p.fx * float(p_norm_last_arr[0]),
                    self.p.cy + self.p.fy * float(p_norm_last_arr[1]),
                ],
                dtype=float,
            )

        return self._guidance_core(
            R_e_b=R_e_b,
            mav_vel_e=mav_vel_e,
            uv_px=uv_px,
            bbox_width_px=bbox_width_px,
            uv_px_last=uv_px_last,
            dt=float(dt),
            t=float(obs.t),
            step=self._monitor_step,
        )

    def _accel_to_control_command(self, obs: Observation, accel_cmd: np.ndarray, yaw_rate_cmd: float) -> ControlCommand:
        R_e_b = quat_to_R(obs.q_eb)
        yaw, _, _ = quat_to_euler_ZYX(obs.q_eb)
        yaw_sp = float(self.p.yaw_d if self.p.yaw_d is not None else yaw)

        thrust_vec_e = self.p.mass * (np.asarray(accel_cmd, dtype=float).reshape(3) - np.array([0.0, 0.0, self.p.g], dtype=float))
        thrust = float(np.linalg.norm(thrust_vec_e))

        if thrust < self.p.eps:
            return ControlCommand(
                t=obs.t,
                thrust=float(np.clip(self.p.mass * self.p.g, self.p.thrust_min, self.p.thrust_max)),
                omega_cmd_b=np.array([0.0, 0.0, np.clip(yaw_rate_cmd, -self.p.omega_max, self.p.omega_max)], dtype=float),
            )

        zb_des_e = -thrust_vec_e / thrust
        xc_ref_e = np.array([np.cos(yaw_sp), np.sin(yaw_sp), 0.0], dtype=float)
        yb_des_e = np.cross(zb_des_e, xc_ref_e)
        if np.linalg.norm(yb_des_e) < self.p.eps:
            yb_des_e = np.array([0.0, 1.0, 0.0], dtype=float)
        yb_des_e = _normalize(yb_des_e, self.p.eps)
        xb_des_e = _normalize(np.cross(yb_des_e, zb_des_e), self.p.eps)
        R_sp = np.column_stack([xb_des_e, yb_des_e, zb_des_e])

        e_r = _vex(R_sp.T @ R_e_b - R_e_b.T @ R_sp)
        omega_cmd_b = -np.asarray(self.p.att_kp, dtype=float).reshape(3) * e_r
        omega_cmd_b[2] = float(np.clip(yaw_rate_cmd, -self.p.omega_max, self.p.omega_max))
        omega_cmd_b = clamp_norm(omega_cmd_b, self.p.omega_max)

        thrust = float(np.clip(thrust, self.p.thrust_min, self.p.thrust_max))
        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=omega_cmd_b)

    def compute_with_bbox(self, obs: Observation, bbox_width_px: float | None = None) -> ControlCommand:
        monitor_step = self._monitor_step
        self._monitor_step += 1

        if (not obs.has_target) or (obs.p_norm is None):
            hover = float(np.clip(self.p.mass * self.p.g, self.p.thrust_min, self.p.thrust_max))
            return ControlCommand(t=obs.t, thrust=hover, omega_cmd_b=np.zeros(3, dtype=float))

        dt = 0.0 if self._last_t is None else max(float(obs.t - self._last_t), 0.0)
        accel_and_yaw = self.compute_accel_command(
            obs,
            bbox_width_px=bbox_width_px,
            p_norm_last=self._last_p_norm,
            dt=dt,
        )

        self._last_p_norm = np.asarray(obs.p_norm, dtype=float).reshape(2)
        self._last_t = float(obs.t)

        self._push_monitor("vpn_dt", "tab:gray", dt, "vpn_debug", obs.t, monitor_step)
        return self._accel_to_control_command(obs, accel_and_yaw[:3], float(accel_and_yaw[6]))

    def compute(self, obs: Observation) -> ControlCommand:
        return self.compute_with_bbox(obs, bbox_width_px=None)