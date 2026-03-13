# control/vpn_acc_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from control.basic_control.setpoints import AccelerationSetpoint
from models.state import Observation
from utils.math3d import clamp_norm, quat_to_R

if TYPE_CHECKING:
    from observe.obj_tracker import Bbox
    from visualization.monitor import Monitor


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


@dataclass(slots=True)
class VpnAccControllerParams:
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
    eps: float = 1e-6


class VpnAccController:
    """
    Velocity-pursuit + proportional-navigation acceleration controller.

     This controller directly outputs `AccelerationSetpoint` for BasicController.

    Notes:
    - Guidance uses tracker-provided `Bbox` directly (u, v, bw, bh).
    """

    def __init__(self, p: VpnAccControllerParams):
        self.p = p
        self.monitor: Monitor | None = None
        self._bw_f = float(self.p.bbox_width_ref_px)
        self._n_delta_vel_f = np.zeros(3, dtype=float)
        self._last_uv_px: np.ndarray | None = None
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
        self._bw_f = float(self.p.bbox_width_ref_px)
        self._n_delta_vel_f[:] = 0.0
        self._last_uv_px = None
        self._last_t = None

    def set_monitor(self, monitor: "Monitor | None") -> None:
        self.monitor = monitor

    @staticmethod
    def _smooth_gate(x: float) -> float:
        x_clip = float(np.clip(x, 0.0, 1.0))
        return x_clip * x_clip * (3.0 - 2.0 * x_clip)

    def _sigma(self, v: np.ndarray, lo: float, hi: float) -> float:
        n = float(np.linalg.norm(v))
        if hi <= lo + self.p.eps:
            return 1.0 if n > lo else 0.0
        return self._smooth_gate((n - lo) / (hi - lo))

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

    def _scale_from_width(self, bbox_width_px: float | None) -> tuple[float, float]:
        if bbox_width_px is None:
            bw = float(self.p.fallback_scale * self.p.bbox_width_ref_px)
        else:
            bw = float(max(bbox_width_px, self.p.eps))

        a = float(np.clip(self.p.bbox_filter_memory, 0.0, 1.0))
        self._bw_f = a * self._bw_f + (1.0 - a) * bw

        s_raw = self._bw_f / max(float(self.p.bbox_width_ref_px), self.p.eps)
        s = float(np.clip(s_raw, self.p.s_min, self.p.s_max))
        gamma = float(np.clip((self.p.s_max - s) / max(self.p.s_max - self.p.s_min, self.p.eps), 0.0, 1.0))
        return s, gamma

    def _guidance_core(
        self,
        *,
        R_e_b: np.ndarray,
        mav_vel_e: np.ndarray,
        uv_px: np.ndarray,
        bbox_width_px: float | None,
        uv_px_last: np.ndarray | None,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        n_eo = self._pixels_to_los_earth(R_e_b, uv_px)
        n_delta_vel = np.zeros(3, dtype=float)

        if uv_px_last is not None and dt > 1e-3:
            n_eol = self._pixels_to_los_earth(R_e_b, uv_px_last)
            n_delta_vel = (n_eo - n_eol) / dt

        s, gamma = self._scale_from_width(bbox_width_px)


        # 追踪项
        v_d = self.p.v_d0 * (1.0 - self.p.alpha * s)
        a_v = self.p.k1 * (v_d * n_eo - mav_vel_e)

        # vpn导引项
        V = float(np.linalg.norm(mav_vel_e))
        alpha_f = float(np.exp(-max(dt, 0.0) / max(self.p.tau, self.p.eps)))
        self._n_delta_vel_f = alpha_f * self._n_delta_vel_f + (1.0 - alpha_f) * n_delta_vel
        a_vpn = self.p.k2 * gamma * self.p.nav_gain * V * self._n_delta_vel_f

        # LOS居中项
        n_bc = self._R_b_c @ self._n_cc
        n_ec = R_e_b @ n_bc
        n_c = _normalize(n_ec, self.p.eps)
        e_n = n_eo - float(np.dot(n_eo, n_c)) * n_c
        a_los = -self.p.k3 * s * V * self._sigma(e_n, self.p.sigma_lo, self.p.sigma_hi) * e_n

        # 期望加速度
        a_d = clamp_norm(np.asarray(a_v + a_vpn + a_los, dtype=float).reshape(3), self.p.accel_max)

        yaw_err = (self.p.cx - float(uv_px[0])) / max(self.p.cx, self.p.eps)
        yaw_rate_far = self.p.yaw_rate_far_gain * yaw_err
        yaw_rate_near = self.p.yaw_rate_near_gain * yaw_err
        yaw_rate = (1.0 - gamma) * yaw_rate_near + gamma * yaw_rate_far
        # 画个图确认一下正方向
        return a_d, float(-yaw_rate)

    def compute_accel_command(
        self,
        obs: Observation,
        *,
        bbox: Bbox | None = None,
        uv_px_last: np.ndarray | None = None,
        dt: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        if not obs.has_target:
            return np.zeros(3, dtype=float), 0.0

        R_e_b = quat_to_R(obs.q_eb)
        mav_vel_e = np.asarray(obs.v_e, dtype=float).reshape(3)
        if bbox is None:
            return np.zeros(3, dtype=float), 0.0
        uv_px = np.array([float(bbox.u), float(bbox.v)], dtype=float)
        bbox_width_px = float(bbox.bw)

        uv_px_prev = None
        if uv_px_last is not None:
            uv_px_prev = np.asarray(uv_px_last, dtype=float).reshape(2)

        return self._guidance_core(
            R_e_b=R_e_b,
            mav_vel_e=mav_vel_e,
            uv_px=uv_px,
            bbox_width_px=bbox_width_px,
            uv_px_last=uv_px_prev,
            dt=float(dt),
        )

    def compute_with_bbox(self, obs: Observation, bbox: Bbox | None = None) -> AccelerationSetpoint:
        if not obs.has_target:
            return AccelerationSetpoint(a_sp_e=np.zeros(3, dtype=float), yaw_sp=None, yawdot_sp=0.0)

        dt = 0.0 if self._last_t is None else max(float(obs.t - self._last_t), 0.0)
        accel_cmd, yaw_rate_cmd = self.compute_accel_command(
            obs,
            bbox=bbox,
            uv_px_last=self._last_uv_px,
            dt=dt,
        )

        self._last_uv_px = None if bbox is None else np.array([float(bbox.u), float(bbox.v)], dtype=float)
        self._last_t = float(obs.t)
        return AccelerationSetpoint(
            a_sp_e=accel_cmd,
            yaw_sp=None,
            yawdot_sp=float(yaw_rate_cmd),
        )