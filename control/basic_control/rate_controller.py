from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.state import ForceSetpoint, UAVState
from .setpoints import RateThrustSetpoint
from .utils import anti_windup_clip


@dataclass(slots=True)
class RateControllerParams:
    kp: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.6], dtype=float))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.02], dtype=float))
    i_limit: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 1.0], dtype=float))
    torque_limit: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 5.0], dtype=float))
    thrust_min: float = 0.0
    thrust_max: float = 40.0


class RateController:
    def __init__(self, p: RateControllerParams):
        self.p = p
        self._i_term = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self._i_term[:] = 0.0

    def step(self, uav: UAVState, sp: RateThrustSetpoint, dt: float) -> ForceSetpoint:
        err = np.asarray(sp.omega_sp_b, dtype=float) - np.asarray(uav.w_b, dtype=float)
        self._i_term += err * dt
        self._i_term = anti_windup_clip(self._i_term, self.p.i_limit)

        tau_sp_b = self.p.kp * err + self.p.ki * self._i_term
        tau_sp_b = np.clip(tau_sp_b, -self.p.torque_limit, self.p.torque_limit)
        thrust_sp = float(np.clip(sp.thrust_sp, self.p.thrust_min, self.p.thrust_max))

        return ForceSetpoint(t=uav.t, thrust_sp=thrust_sp, tau_sp_b=tau_sp_b)

