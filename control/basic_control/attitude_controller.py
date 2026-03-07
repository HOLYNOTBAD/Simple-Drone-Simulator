from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.state import UAVState
from utils.math3d import quat_to_R
from .setpoints import AttThrustSetpoint, RateThrustSetpoint


def _vex(s: np.ndarray) -> np.ndarray:
    return np.array([s[2, 1], s[0, 2], s[1, 0]], dtype=float)


@dataclass(slots=True)
class AttitudeControllerParams:
    k_r: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 2.0], dtype=float))
    omega_max: float = 6.0


class AttitudeController:
    def __init__(self, p: AttitudeControllerParams):
        self.p = p

    def reset(self) -> None:
        pass

    def step(self, uav: UAVState, sp: AttThrustSetpoint) -> RateThrustSetpoint:
        r = quat_to_R(uav.q_eb)
        r_sp = quat_to_R(sp.q_sp_eb)
        e_r = _vex(r_sp.T @ r - r.T @ r_sp)
        omega_sp = -self.p.k_r * e_r
        omega_sp = np.clip(omega_sp, -self.p.omega_max, self.p.omega_max)
        return RateThrustSetpoint(omega_sp_b=omega_sp, thrust_sp=sp.thrust_sp)

