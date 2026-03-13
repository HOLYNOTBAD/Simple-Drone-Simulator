from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.state import UAVState
from .setpoints import AccelerationSetpoint, VelocitySetpoint


@dataclass(slots=True)
class VelocityControllerParams:
    kp: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))


class VelocityController:
    def __init__(self, p: VelocityControllerParams):
        self.p = p

    def reset(self) -> None:
        pass

    def step(self, uav: UAVState, sp: VelocitySetpoint) -> AccelerationSetpoint:
        a_sp = self.p.kp * (sp.v_sp_e - uav.v_e)
        return AccelerationSetpoint(a_sp_e=a_sp, yaw_sp=sp.yaw_sp)
