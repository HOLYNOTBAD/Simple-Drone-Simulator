from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.state import UAVState
from utils.math3d import R_to_quat
from .setpoints import AttThrustSetpoint, VelocitySetpoint


@dataclass(slots=True)
class VelocityControllerParams:
    mass: float = 1.0
    g: float = 9.81
    kp: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))


class VelocityController:
    def __init__(self, p: VelocityControllerParams):
        self.p = p

    def reset(self) -> None:
        pass

    def step(self, uav: UAVState, sp: VelocitySetpoint) -> AttThrustSetpoint:
        a_sp = self.p.kp * (sp.v_sp_e - uav.v_e)
        thrust_vec_e = self.p.mass * (a_sp - np.array([0.0, 0.0, self.p.g], dtype=float))
        thrust_sp = float(np.linalg.norm(thrust_vec_e))

        zb_des_e = -thrust_vec_e / max(thrust_sp, 1e-9)
        if sp.yaw_sp is None:
            xc_ref_e = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            yaw = float(sp.yaw_sp)
            xc_ref_e = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        yb_des_e = np.cross(zb_des_e, xc_ref_e)
        if np.linalg.norm(yb_des_e) < 1e-9:
            yb_des_e = np.array([0.0, 1.0, 0.0], dtype=float)
        yb_des_e /= np.linalg.norm(yb_des_e)
        xb_des_e = np.cross(yb_des_e, zb_des_e)
        r_e_b = np.column_stack([xb_des_e, yb_des_e, zb_des_e])
        q_sp = R_to_quat(r_e_b)
        return AttThrustSetpoint(q_sp_eb=q_sp, thrust_sp=thrust_sp)
