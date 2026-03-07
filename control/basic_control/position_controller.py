from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.state import UAVState
from .setpoints import PositionSetpoint, VelocitySetpoint
from .utils import clamp_norm


@dataclass(slots=True)
class PositionControllerParams:
    kp: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))
    v_max: float = 8.0


class PositionController:
    def __init__(self, p: PositionControllerParams):
        self.p = p

    def reset(self) -> None:
        pass

    def step(self, uav: UAVState, sp: PositionSetpoint) -> VelocitySetpoint:
        v_sp = self.p.kp * (sp.p_sp_e - uav.p_e)
        v_sp = clamp_norm(v_sp, self.p.v_max)
        return VelocitySetpoint(v_sp_e=v_sp, yaw_sp=sp.yaw_sp)

