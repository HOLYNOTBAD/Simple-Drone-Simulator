# models/state.py
"""
Data contracts (protocol layer) for the whole simulation framework.

Key design goals:
- Modules exchange ONLY these dataclasses (stable interfaces).
- Coordinate frames are explicitly stated to avoid ambiguity.
- Shapes are validated early to catch bugs.

Conventions (recommended):
- World frame: {e} (inertial / earth-fixed)
- Body frame:  {b} (UAV body)
- Camera frame:{c} (camera optical frame, pinhole model)
- Image coords:{i} (pixel or normalized image plane)

Quaternion convention:
- q = [w, x, y, z]  (scalar-first)
- Represents rotation from body {b} to world {e}:  R_e_b(q)
  (i.e., vector in body transformed to world by v_e = R_e_b * v_b)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


Array = np.ndarray


def _as_vec(x, n: int, name: str) -> Array:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    return a


def _as_quat(q, name: str = "q") -> Array:
    a = np.asarray(q, dtype=float).reshape(-1)
    if a.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {a.shape}")
    # do not force-normalize silently; normalize explicitly in math utilities if desired
    return a


def _is_finite(a: Array, name: str) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")


@dataclass(slots=True)
class UAVState:
    """
    UAV rigid-body state (ground truth in simulation).

    t: time [s]
    p_e: position in world frame {e}, meters
    v_e: velocity in world frame {e}, m/s
    q_eb: quaternion (body -> world), [w, x, y, z]
    w_b: angular velocity in body frame {b}, rad/s
    """
    t: float
    p_e: Array
    v_e: Array
    q_eb: Array
    w_b: Array

    def __post_init__(self) -> None:
        self.p_e = _as_vec(self.p_e, 3, "UAVState.p_e")
        self.v_e = _as_vec(self.v_e, 3, "UAVState.v_e")
        self.q_eb = _as_quat(self.q_eb, "UAVState.q_eb")
        self.w_b = _as_vec(self.w_b, 3, "UAVState.w_b")
        _is_finite(self.p_e, "UAVState.p_e")
        _is_finite(self.v_e, "UAVState.v_e")
        _is_finite(self.q_eb, "UAVState.q_eb")
        _is_finite(self.w_b, "UAVState.w_b")


@dataclass(slots=True)
class TargetState:
    """
    Target point-mass state (ground truth in simulation).

    p_e: position in world frame {e}, meters
    v_e: velocity in world frame {e}, m/s
    """
    t: float
    p_e: Array
    v_e: Array

    def __post_init__(self) -> None:
        self.p_e = _as_vec(self.p_e, 3, "TargetState.p_e")
        self.v_e = _as_vec(self.v_e, 3, "TargetState.v_e")
        _is_finite(self.p_e, "TargetState.p_e")
        _is_finite(self.v_e, "TargetState.v_e")


@dataclass(slots=True)
class CameraMeasurement:
    """
    Camera measurement at a (possibly delayed) timestamp.

    p_cam: target relative position in camera frame {c}, meters (optional)
    bearing_c: unit line-of-sight vector in camera frame {c} (optional)
    p_norm: normalized image coordinate (x/z, y/z) in image plane (dimensionless)
            i.e., [u/fx, v/fy] if principal point removed and fx=fy=foc in pixels
    uv_px:  pixel coordinate [u, v] in pixels (optional, for logging/debug)
    range_m: line-of-sight distance from camera to target, meters (optional)
    valid:  whether the target is within FOV / successfully detected
    """
    t_meas: float
    p_cam: Optional[Array] = None         # shape (3,)
    bearing_c: Optional[Array] = None     # shape (3,)
    p_norm: Optional[Array] = None        # shape (2,)
    uv_px: Optional[Array] = None         # shape (2,)
    range_m: Optional[float] = None
    valid: bool = True

    def __post_init__(self) -> None:
        if self.p_cam is not None:
            self.p_cam = _as_vec(self.p_cam, 3, "CameraMeasurement.p_cam")
            _is_finite(self.p_cam, "CameraMeasurement.p_cam")
        if self.bearing_c is not None:
            self.bearing_c = _as_vec(self.bearing_c, 3, "CameraMeasurement.bearing_c")
            _is_finite(self.bearing_c, "CameraMeasurement.bearing_c")
        if self.p_norm is not None:
            self.p_norm = _as_vec(self.p_norm, 2, "CameraMeasurement.p_norm")
            _is_finite(self.p_norm, "CameraMeasurement.p_norm")
        if self.uv_px is not None:
            self.uv_px = _as_vec(self.uv_px, 2, "CameraMeasurement.uv_px")
            _is_finite(self.uv_px, "CameraMeasurement.uv_px")
        if self.range_m is not None:
            self.range_m = float(self.range_m)
            if not np.isfinite(self.range_m):
                raise ValueError("CameraMeasurement.range_m must be finite")


@dataclass(slots=True)
class Observation:
    """
    What the controller 'sees' at control update times.

    In L1:
      - constructed from perfect camera projection + perfect self-state
    In L2/L3:
      - constructed from sensor fusion/estimation; may be delayed/noisy.

    Minimal recommended fields for IBVS-style interception:
    - p_norm: normalized image feature (2D)
    - bearing_c: unit line-of-sight vector in camera frame (optional)
    - q_eb, w_b: UAV attitude & body angular rate (available from IMU in real systems)
    - v_e: UAV velocity in world (can be estimated; optional but often useful)
    - p_r: interceptor relative position in world, p_uav - p_tgt
    - v_r: interceptor relative velocity in world, v_uav - v_tgt
    - has_target: whether a valid target measurement is present
    """
    t: float
    p_norm: Optional[Array]               # shape (2,) or None if no target
    q_eb: Array                           # shape (4,)
    w_b: Array                            # shape (3,)
    v_e: Array                            # shape (3,)
    p_r: Optional[Array] = None           # shape (3,)
    v_r: Optional[Array] = None           # shape (3,)
    bearing_c: Optional[Array] = None     # shape (3,)
    range_m: Optional[float] = None
    has_target: bool = True

    def __post_init__(self) -> None:
        if self.p_norm is not None:
            self.p_norm = _as_vec(self.p_norm, 2, "Observation.p_norm")
            _is_finite(self.p_norm, "Observation.p_norm")
        if self.p_r is not None:
            self.p_r = _as_vec(self.p_r, 3, "Observation.p_r")
            _is_finite(self.p_r, "Observation.p_r")
        if self.v_r is not None:
            self.v_r = _as_vec(self.v_r, 3, "Observation.v_r")
            _is_finite(self.v_r, "Observation.v_r")
        if self.bearing_c is not None:
            self.bearing_c = _as_vec(self.bearing_c, 3, "Observation.bearing_c")
            _is_finite(self.bearing_c, "Observation.bearing_c")
        self.q_eb = _as_quat(self.q_eb, "Observation.q_eb")
        self.w_b = _as_vec(self.w_b, 3, "Observation.w_b")
        self.v_e = _as_vec(self.v_e, 3, "Observation.v_e")
        _is_finite(self.q_eb, "Observation.q_eb")
        _is_finite(self.w_b, "Observation.w_b")
        _is_finite(self.v_e, "Observation.v_e")
        if self.range_m is not None:
            self.range_m = float(self.range_m)
            if not np.isfinite(self.range_m):
                raise ValueError("Observation.range_m must be finite")


@dataclass(slots=True)
class ControlCommand:
    """
    Controller output command.

    You requested: thrust + body-rate command (omega_cmd).
    This matches many multicopter low-level interfaces.

    thrust: total thrust magnitude [N] (or normalized in [0,1] if you prefer)
            Choose one convention project-wide and stick to it.
    omega_cmd_b: desired body angular velocity [rad/s] in body frame {b}
    """
    t: float
    thrust: float
    omega_cmd_b: Array

    # Optional: keep a "mode" field to support hover/intercept etc.
    mode: Literal["rate"] = "rate"

    def __post_init__(self) -> None:
        self.omega_cmd_b = _as_vec(self.omega_cmd_b, 3, "ControlCommand.omega_cmd_b")
        _is_finite(self.omega_cmd_b, "ControlCommand.omega_cmd_b")
        if not np.isfinite(self.thrust):
            raise ValueError("ControlCommand.thrust must be finite")


@dataclass(slots=True)
class ForceSetpoint:
    """
    Low-level body wrench target.

    thrust_sp: total thrust magnitude [N]
    tau_sp_b: desired body torque [N*m] in body frame {b}
    """
    t: float
    thrust_sp: float
    tau_sp_b: Array

    def __post_init__(self) -> None:
        self.tau_sp_b = _as_vec(self.tau_sp_b, 3, "ForceSetpoint.tau_sp_b")
        _is_finite(self.tau_sp_b, "ForceSetpoint.tau_sp_b")
        if not np.isfinite(self.thrust_sp):
            raise ValueError("ForceSetpoint.thrust_sp must be finite")


@dataclass(slots=True)
class MotorCommand:
    """
    Executable actuator command for the motor model.
    """
    t: float
    motor_current_cmd: Array

    def __post_init__(self) -> None:
        self.motor_current_cmd = np.asarray(self.motor_current_cmd, dtype=float).reshape(-1)
        _is_finite(self.motor_current_cmd, "MotorCommand.motor_current_cmd")


# --- Optional helper container for logging/debugging (not required) ---

@dataclass(slots=True)
class SimSnapshot:
    """
    One record at a given simulation time for logging/debugging.
    """
    t: float
    uav: UAVState
    target: TargetState
    cam: Optional[CameraMeasurement]
    obs: Optional[Observation]
    cmd: Optional[ControlCommand]
