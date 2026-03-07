# models/target.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from models.state import TargetState


@dataclass(slots=True)
class TargetParams:
    """Simple target model parameters (L1)."""
    accel_e: np.ndarray | None = None  # constant acceleration in NED (optional)


class TargetPointMass:
    """
    L1 target model: constant velocity (or constant acceleration if provided).
    """

    def __init__(self, params: TargetParams):
        self.p = params
        if self.p.accel_e is not None:
            self.p.accel_e = np.asarray(self.p.accel_e, dtype=float).reshape(3)

    def step(self, x: TargetState, dt: float) -> TargetState:
        if self.p.accel_e is None:
            v_next = x.v_e
        else:
            v_next = x.v_e + self.p.accel_e * dt
        p_next = x.p_e + v_next * dt
        return TargetState(t=x.t + dt, p_e=p_next, v_e=v_next)