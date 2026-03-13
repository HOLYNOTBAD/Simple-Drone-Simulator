from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _vec(x, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _quat(q, name: str) -> np.ndarray:
    a = np.asarray(q, dtype=float).reshape(-1)
    if a.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


@dataclass(slots=True)
class PositionSetpoint:
    p_sp_e: np.ndarray
    yaw_sp: Optional[float] = None

    def __post_init__(self) -> None:
        self.p_sp_e = _vec(self.p_sp_e, 3, "PositionSetpoint.p_sp_e")


@dataclass(slots=True)
class VelocitySetpoint:
    v_sp_e: np.ndarray
    yaw_sp: Optional[float] = None

    def __post_init__(self) -> None:
        self.v_sp_e = _vec(self.v_sp_e, 3, "VelocitySetpoint.v_sp_e")


@dataclass(slots=True)
class AttThrustSetpoint:
    q_sp_eb: np.ndarray
    thrust_sp: float

    def __post_init__(self) -> None:
        self.q_sp_eb = _quat(self.q_sp_eb, "AttThrustSetpoint.q_sp_eb")
        self.thrust_sp = float(self.thrust_sp)


@dataclass(slots=True)
class RateThrustSetpoint:
    omega_sp_b: np.ndarray
    thrust_sp: float

    def __post_init__(self) -> None:
        self.omega_sp_b = _vec(self.omega_sp_b, 3, "RateThrustSetpoint.omega_sp_b")
        self.thrust_sp = float(self.thrust_sp)

