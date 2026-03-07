from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from models.rigid_body import RigidBodyParams


@dataclass(slots=True)
class MotorOutputs:
    i_cmd: np.ndarray
    omega: np.ndarray
    thrusts: np.ndarray
    force_b: np.ndarray
    torque_b: np.ndarray


class Motors:
    """
    Rotor/motor dynamics:
    - input: motor current commands (A)
    - state: rotor speed omega_i (rad/s)
    - output: total body force/torque from all rotors
    """

    def __init__(self, rb: RigidBodyParams):
        self.rb = rb
        self.n = int(rb.num_rotors)
        self.omega = np.zeros(self.n, dtype=float)

        if rb.rotor_pos is None:
            d = float(rb.arm_length) / np.sqrt(2.0)
            self.rotor_pos_b = np.array(
                [[d, d, 0.0], [d, -d, 0.0], [-d, -d, 0.0], [-d, d, 0.0]],
                dtype=float,
            )
        else:
            self.rotor_pos_b = np.array([rb.rotor_pos[k] for k in sorted(rb.rotor_pos.keys())], dtype=float)

        if rb.rotor_directions is None:
            self.rotor_dirs = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
        else:
            self.rotor_dirs = np.asarray(rb.rotor_directions, dtype=float).reshape(-1)

        if self.rotor_pos_b.shape != (self.n, 3):
            raise ValueError(f"rotor_pos shape must be ({self.n},3), got {self.rotor_pos_b.shape}")
        if self.rotor_dirs.shape != (self.n,):
            raise ValueError(f"rotor_directions shape must be ({self.n},), got {self.rotor_dirs.shape}")

    def reset(self) -> None:
        self.omega[:] = 0.0

    def step(self, i_cmd: np.ndarray, dt: float) -> MotorOutputs:
        i_cmd = np.asarray(i_cmd, dtype=float).reshape(self.n)

        i_cmd = np.clip(
            i_cmd,
            float(getattr(self.rb, "motor_current_min", 0.0)),
            float(getattr(self.rb, "motor_current_max", 40.0)),
        )

        k_i = max(float(getattr(self.rb, "motor_k_current", 40.0)), 1e-12)
        tau_m = max(float(self.rb.tau_m), 1e-4)

        omega_des = k_i * i_cmd
        omega_des += float(self.rb.motor_noise_std) * np.random.randn(self.n)

        omega_dot = (omega_des - self.omega) / tau_m
        self.omega = self.omega + omega_dot * dt
        self.omega = np.clip(self.omega, float(self.rb.rotor_speed_min), float(self.rb.rotor_speed_max))

        thrusts = float(self.rb.k_eta) * self.omega**2

        # Body force: each rotor thrust along -z_b
        force_b = np.array([0.0, 0.0, -np.sum(thrusts)], dtype=float)

        torque_b = np.zeros(3, dtype=float)
        for r_i, f_i in zip(self.rotor_pos_b, thrusts):
            f_vec = np.array([0.0, 0.0, -f_i], dtype=float)
            torque_b += np.cross(r_i, f_vec)

        # Yaw reaction torque from rotor drag
        torque_b[2] += np.sum(self.rotor_dirs * float(self.rb.k_m) * self.omega**2)

        return MotorOutputs(
            i_cmd=i_cmd.copy(),
            omega=self.omega.copy(),
            thrusts=thrusts.copy(),
            force_b=force_b,
            torque_b=torque_b,
        )
