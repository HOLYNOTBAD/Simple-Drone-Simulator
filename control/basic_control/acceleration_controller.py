from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.state import UAVState
from utils.math3d import R_to_quat, quat_to_euler_ZYX
from .setpoints import AccelerationSetpoint, AttThrustSetpoint


@dataclass(slots=True)
class AccelerationControllerParams:
    mass: float = 1.0
    g: float = 9.81


class AccelerationController:
    def __init__(self, p: AccelerationControllerParams):
        self.p = p
        self._yaw_int: float | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._yaw_int = None
        self._last_t = None

    @staticmethod
    def _wrap_pi(x: float) -> float:
        return float((x + np.pi) % (2.0 * np.pi) - np.pi)

    def _resolve_yaw_setpoint(self, uav: UAVState, sp: AccelerationSetpoint) -> float:
        yaw_now, _, _ = quat_to_euler_ZYX(uav.q_eb)

        if sp.yaw_sp is not None:
            yaw_cmd = float(sp.yaw_sp)
            self._yaw_int = yaw_cmd
            self._last_t = float(uav.t)
            return yaw_cmd

        if sp.yawdot_sp is not None:
            if self._yaw_int is None:
                self._yaw_int = float(yaw_now)
                self._last_t = float(uav.t)
                return self._yaw_int

            dt = max(float(uav.t) - float(self._last_t if self._last_t is not None else uav.t), 0.0)
            self._yaw_int = self._wrap_pi(float(self._yaw_int) + float(sp.yawdot_sp) * dt)
            self._last_t = float(uav.t)
            return float(self._yaw_int)

        self._yaw_int = float(yaw_now)
        self._last_t = float(uav.t)
        return float(yaw_now)

    def step(self, uav: UAVState, sp: AccelerationSetpoint) -> AttThrustSetpoint:
        thrust_vec_e = self.p.mass * (
            np.asarray(sp.a_sp_e, dtype=float)
            - np.array([0.0, 0.0, self.p.g], dtype=float)
        )
        thrust_sp = float(np.linalg.norm(thrust_vec_e))

        zb_des_e = -thrust_vec_e / max(thrust_sp, 1e-9)
        yaw = self._resolve_yaw_setpoint(uav, sp)
        xc_ref_e = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        yb_des_e = np.cross(zb_des_e, xc_ref_e)
        if np.linalg.norm(yb_des_e) < 1e-9:
            yb_des_e = np.array([0.0, 1.0, 0.0], dtype=float)
        yb_des_e /= np.linalg.norm(yb_des_e)
        xb_des_e = np.cross(yb_des_e, zb_des_e)
        r_e_b = np.column_stack([xb_des_e, yb_des_e, zb_des_e])
        q_sp = R_to_quat(r_e_b)
        return AttThrustSetpoint(q_sp_eb=q_sp, thrust_sp=thrust_sp)