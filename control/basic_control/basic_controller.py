from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from models.rigid_body import RigidBodyParams
from models.state import ControlCommand, ForceSetpoint, MotorCommand, Observation, UAVState

from .attitude_controller import AttitudeController, AttitudeControllerParams
from .position_controller import PositionController, PositionControllerParams
from .rate_controller import RateController, RateControllerParams
from .setpoints import AttThrustSetpoint, PositionSetpoint, RateThrustSetpoint, VelocitySetpoint
from .velocity_controller import VelocityController, VelocityControllerParams


@dataclass(slots=True)
class AllocationParams:
    current_min: float = 0.0
    current_max: float = 40.0
    k_current: float = 40.0


class CTRL_MODE(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    ATT_THRUST = "att_thrust"
    RATE_THRUST = "rate_thrust"


class _ControlAllocator:
    def __init__(self, p: AllocationParams, rb: RigidBodyParams):
        self.p = p
        self.rb = rb
        self._rotor_pos_b, self._rotor_dirs = self._build_rotor_layout()
        self._alloc = self._build_allocation_matrix()
        self._alloc_pinv = np.linalg.pinv(self._alloc)
        self._f_per_rotor_max = float(self.rb.k_eta) * float(self.rb.rotor_speed_max) ** 2

    def _build_rotor_layout(self) -> tuple[np.ndarray, np.ndarray]:
        n = int(self.rb.num_rotors)
        if self.rb.rotor_pos is None:
            d = float(self.rb.arm_length) / np.sqrt(2.0)
            pos = np.array([[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]], dtype=float)
        else:
            pos = np.array([self.rb.rotor_pos[k] for k in sorted(self.rb.rotor_pos.keys())], dtype=float)
        dirs = (
            np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
            if self.rb.rotor_directions is None
            else np.asarray(self.rb.rotor_directions, dtype=float).reshape(-1)
        )
        if pos.shape != (n, 3):
            raise ValueError(f"rotor_pos shape must be ({n}, 3), got {pos.shape}")
        if dirs.shape != (n,):
            raise ValueError(f"rotor_directions shape must be ({n},), got {dirs.shape}")
        return pos, dirs

    def _build_allocation_matrix(self) -> np.ndarray:
        n = self._rotor_pos_b.shape[0]
        a = np.zeros((4, n), dtype=float)
        kappa = float(self.rb.k_m) / max(float(self.rb.k_eta), 1e-12)
        a[0, :] = 1.0
        a[1, :] = -self._rotor_pos_b[:, 1]
        a[2, :] = self._rotor_pos_b[:, 0]
        a[3, :] = self._rotor_dirs * kappa
        return a

    def step(self, sp: ForceSetpoint) -> MotorCommand:
        u = np.array([sp.thrust_sp, sp.tau_sp_b[0], sp.tau_sp_b[1], sp.tau_sp_b[2]], dtype=float)
        f_rot = self._alloc_pinv @ u
        f_rot = np.clip(f_rot, 0.0, self._f_per_rotor_max)

        omega_des = np.sqrt(np.maximum(f_rot, 0.0) / max(float(self.rb.k_eta), 1e-12))
        i_cmd = omega_des / max(float(self.p.k_current), 1e-12)
        i_cmd = np.clip(i_cmd, self.p.current_min, self.p.current_max)
        return MotorCommand(t=sp.t, motor_current_cmd=i_cmd)


class BasicController:
    """
    Unified PX4-like basic control entry.

    Supported path:
      PositionSetpoint -> VelocitySetpoint -> AttThrustSetpoint ->
      RateThrustSetpoint -> ForceSetpoint -> MotorCommand

    Transitional path:
      ControlCommand -> ForceSetpoint -> MotorCommand
    """

    def __init__(
        self,
        rb_params: RigidBodyParams,
        *,
        position_controller: PositionController | None = None,
        velocity_controller: VelocityController | None = None,
        attitude_controller: AttitudeController | None = None,
    ):
        self.rb_params = rb_params
        self.position_controller = position_controller or PositionController(
            PositionControllerParams()
        )
        self.velocity_controller = velocity_controller or VelocityController(
            VelocityControllerParams(mass=rb_params.mass, g=rb_params.g)
        )
        self.attitude_controller = attitude_controller or AttitudeController(
            AttitudeControllerParams()
        )
        self.rate_controller = RateController(
            RateControllerParams(
                kp=rb_params.k_w * np.array([1.0, 1.0, 0.6], dtype=float),
                ki=rb_params.k_w * np.array([0.05, 0.05, 0.02], dtype=float),
                thrust_min=float(rb_params.thrust_min),
                thrust_max=float(rb_params.thrust_max),
            )
        )
        self.allocation = _ControlAllocator(
            AllocationParams(
                current_min=float(getattr(rb_params, "motor_current_min", 0.0)),
                current_max=float(getattr(rb_params, "motor_current_max", 40.0)),
                k_current=float(getattr(rb_params, "motor_k_current", 40.0)),
            ),
            rb_params,
        )
        self.mode = CTRL_MODE.RATE_THRUST
        self._sp: object | None = None

    def reset(self) -> None:
        self.position_controller.reset()
        self.velocity_controller.reset()
        self.attitude_controller.reset()
        self.rate_controller.reset()

    def update_setpoint(self, sp: object) -> None:
        self._sp = sp
        if isinstance(sp, PositionSetpoint):
            self.mode = CTRL_MODE.POSITION
        elif isinstance(sp, VelocitySetpoint):
            self.mode = CTRL_MODE.VELOCITY
        elif isinstance(sp, AttThrustSetpoint):
            self.mode = CTRL_MODE.ATT_THRUST
        elif isinstance(sp, RateThrustSetpoint):
            self.mode = CTRL_MODE.RATE_THRUST
        else:
            raise TypeError(f"Unsupported setpoint type: {type(sp)!r}")

    def _resolve_rate_setpoint(self, uav_state: UAVState, sp: object) -> RateThrustSetpoint:
        if isinstance(sp, PositionSetpoint):
            sp = self.position_controller.step(uav_state, sp)
        if isinstance(sp, VelocitySetpoint):
            sp = self.velocity_controller.step(uav_state, sp)
        if isinstance(sp, AttThrustSetpoint):
            sp = self.attitude_controller.step(uav_state, sp)
        if not isinstance(sp, RateThrustSetpoint):
            raise TypeError(f"Expected RateThrustSetpoint after cascade, got {type(sp)!r}")
        return sp

    def step(
        self,
        uav_state: UAVState,
        obs: Observation | None = None,
        t_now: float | None = None,
        dt: float = 0.0,
    ) -> MotorCommand:
        del obs, t_now
        if self._sp is None:
            raise RuntimeError("No setpoint provided to BasicController")
        rate_sp = self._resolve_rate_setpoint(uav_state, self._sp)
        force_sp = self.rate_controller.step(uav_state, rate_sp, dt)
        return self.allocation.step(force_sp)

    def step_with_force(
        self,
        uav_state: UAVState,
        sp: RateThrustSetpoint,
        dt: float,
    ) -> tuple[ForceSetpoint, MotorCommand]:
        force_sp = self.rate_controller.step(uav_state, sp, dt)
        motor_cmd = self.allocation.step(force_sp)
        return force_sp, motor_cmd

    def step_from_command(self, uav: UAVState, cmd: ControlCommand, dt: float) -> tuple[ForceSetpoint, MotorCommand]:
        sp = RateThrustSetpoint(omega_sp_b=cmd.omega_cmd_b, thrust_sp=cmd.thrust)
        return self.step_with_force(uav, sp, dt)
