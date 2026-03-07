# models/rigid_body.py
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np

from models.state import UAVState
from utils.math3d import quat_to_R, integrate_quat_body_rate


@dataclass(slots=True)
class RigidBodyParams:
    """
    Rigid body parameters.

    Defaults are set to AscTec Hummingbird values where applicable.
    """
    # Inertial properties
    mass: float = 0.500        # kg
    g: float = 9.81            # m/s^2 (positive Down in NED)
    Ixx: float = 3.65e-3       # kg*m^2
    Iyy: float = 3.68e-3       # kg*m^2
    Izz: float = 7.03e-3       # kg*m^2
    Ixy: float = 0.0           # kg*m^2
    Iyz: float = 0.0           # kg*m^2
    Ixz: float = 0.0           # kg*m^2

    # Command / actuator abstraction for current L1 controller interface
    omega_max: float = 1.0     # rad/s (rate command saturation)
    thrust_min: float = 0.0    # N
    thrust_max: float = 10.0   # N
    rate_kp: float = 0.08      # N*m/(rad/s), body-rate tracking gain
    motor_current_min: float = 0.0
    motor_current_max: float = 40.0
    motor_k_current: float = 40.0  # rad/s per Amp

    # Geometry / aero / motor
    arm_length: float = 0.17
    num_rotors: int = 4
    rotor_radius: float = 0.10
    rotor_pos: dict | None = None
    rotor_directions: np.ndarray | None = None
    rI: np.ndarray | None = None
    c_Dx: float = 0.5e-2
    c_Dy: float = 0.5e-2
    c_Dz: float = 1.0e-2
    k_eta: float = 5.57e-06
    k_m: float = 1.36e-07
    k_d: float = 1.19e-04
    k_z: float = 2.32e-04
    k_h: float = 3.39e-3
    k_flap: float = 0.0
    tau_m: float = 0.005
    rotor_speed_min: float = 0.0
    rotor_speed_max: float = 1500.0
    motor_noise_std: float = 0.0
    k_w: float = 1.0
    k_v: float = 10.0
    kp_att: float = 544.0
    kd_att: float = 46.64

    def inertia_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.Ixx, self.Ixy, self.Ixz],
                [self.Ixy, self.Iyy, self.Iyz],
                [self.Ixz, self.Iyz, self.Izz],
            ],
            dtype=float,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "RigidBodyParams":
        valid = {f.name for f in fields(cls)}
        kw = {k: v for k, v in data.items() if k in valid}

        float_keys = [
            "mass", "g", "Ixx", "Iyy", "Izz", "Ixy", "Iyz", "Ixz",
            "omega_max", "thrust_min", "thrust_max", "rate_kp",
            "motor_current_min", "motor_current_max", "motor_k_current",
            "arm_length", "rotor_radius",
            "c_Dx", "c_Dy", "c_Dz",
            "k_eta", "k_m", "k_d", "k_z", "k_h", "k_flap",
            "tau_m", "rotor_speed_min", "rotor_speed_max", "motor_noise_std",
            "k_w", "k_v", "kp_att", "kd_att",
        ]
        int_keys = ["num_rotors"]
        for k in float_keys:
            if k in kw and kw[k] is not None:
                kw[k] = float(kw[k])
        for k in int_keys:
            if k in kw and kw[k] is not None:
                kw[k] = int(kw[k])

        if "rotor_directions" in kw and kw["rotor_directions"] is not None:
            kw["rotor_directions"] = np.asarray(kw["rotor_directions"], dtype=float)
        if "rI" in kw and kw["rI"] is not None:
            kw["rI"] = np.asarray(kw["rI"], dtype=float)
        if "rotor_pos" in kw and kw["rotor_pos"] is not None:
            kw["rotor_pos"] = {k: np.asarray(v, dtype=float) for k, v in kw["rotor_pos"].items()}

        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RigidBodyParams":
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("PyYAML is required to load rigid body params from yaml") from e

        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"RigidBody yaml must be a mapping: {p}")
        return cls.from_dict(data)


class RigidBody6DoF:
    """
    Rigid body dynamics driven by body force/torque (from motor model).

    State:
      p_e, v_e in world NED
      q_eb (body->world), w_b (body rates in FRD)

    Inputs per step:
      force_b: total force in body frame {b}, N
      torque_b: total torque in body frame {b}, N*m
    """

    def __init__(self, params: RigidBodyParams):
        self.p = params
        self.I_b = self.p.inertia_matrix()
        self.I_b_inv = np.linalg.inv(self.I_b)

    def step(self, x: UAVState, force_b: np.ndarray, torque_b: np.ndarray, dt: float) -> UAVState:
        force_b = np.asarray(force_b, dtype=float).reshape(3)
        torque_b = np.asarray(torque_b, dtype=float).reshape(3)

        # Angular dynamics: I w_dot + w x (I w) = tau
        w_b = np.asarray(x.w_b, dtype=float).reshape(3)
        coriolis = np.cross(w_b, self.I_b @ w_b)
        w_dot = self.I_b_inv @ (torque_b - coriolis)
        w_next = w_b + w_dot * dt
        w_next = np.clip(w_next, -self.p.omega_max, self.p.omega_max)

        # Rotation body->world
        R_e_b = quat_to_R(x.q_eb)

        # Forces: gravity in NED is +g in Down axis
        g_e = np.array([0.0, 0.0, self.p.g], dtype=float)

        # Simple parasitic drag model in body frame
        v_b = R_e_b.T @ x.v_e
        f_drag_b = np.array(
            [
                -self.p.c_Dx * v_b[0] * abs(v_b[0]),
                -self.p.c_Dy * v_b[1] * abs(v_b[1]),
                -self.p.c_Dz * v_b[2] * abs(v_b[2]),
            ],
            dtype=float,
        )

        f_total_b = force_b + f_drag_b
        a_e = g_e + (R_e_b @ f_total_b) / self.p.mass

        # Semi-implicit Euler (stable enough for L1)
        v_next = x.v_e + a_e * dt
        p_next = x.p_e + v_next * dt

        q_next = integrate_quat_body_rate(x.q_eb, w_next, dt)

        return UAVState(
            t=x.t + dt,
            p_e=p_next,
            v_e=v_next,
            q_eb=q_next,
            w_b=w_next,
        )
