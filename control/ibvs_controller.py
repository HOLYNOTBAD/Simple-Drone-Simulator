# control/ibvs_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

from control.controller_base import ControllerBase
from models.state import ControlCommand, Observation
from utils.math3d import clamp_norm, quat_to_euler_ZYX

if TYPE_CHECKING:
    from visualization.monitor import Monitor


@dataclass(slots=True)
class IBVSControllerParams:
    mass: float = 0.5
    g: float = 9.81
    foc: float = 320.0

    k1: float = 1.0
    k2: float = 0.05
    k3: float = 3.0
    k4: float = 1.0
    k5: float = 0.008
    k6: float = 3.0

    theta_th: float = 0.0
    phi_d: float = 0.0

    omega_max: float = 6.0
    thrust_min: float = 0.0
    thrust_max: float = 30.0
    cos_theta_min: float = 0.2
    eps: float = 1e-6


class IBVSController(ControllerBase):
    """
    Classical image-based controller with body-rate + thrust output.

    Control law:
      bω = R_c^b * [ -k2*ey - k3*(theta - theta_d),
                      k5*ex,
                      k6*(phi - phi_d) ]

      f = m/cos(theta) * ( k4 * (c_vy - k1*ey) + g )

    Conventions used here:
    - ex, ey are image errors on the normalized plane converted back to focal units:
        ex = foc * p_norm[0], ey = foc * p_norm[1]
    - theta_d = max(theta_th, arctan(ey / foc))
    - phi_d is a constant trim value from config
    - c_vy is the target relative velocity expressed in the camera frame, y component
    """

    def __init__(self, p: IBVSControllerParams):
        self.p = p
        self.monitor: Monitor | None = None
        self._monitor_step = 0
        self._R_b_c = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )

    def reset(self) -> None:
        self._monitor_step = 0

    def set_monitor(self, monitor: "Monitor | None") -> None:
        self.monitor = monitor

    def _push_monitor(self, name: str, color: str, data: float, group: str, t: float, step: int) -> None:
        if self.monitor is None:
            return
        self.monitor.push(name=name, color=color, data=float(data), t=t, group=group, step=step)

    @staticmethod
    def _relative_velocity_camera(obs: Observation) -> np.ndarray:
        vr_e = getattr(obs, "v_r", None)
        pr_e = getattr(obs, "p_r", None)
        if vr_e is None or pr_e is None:
            return np.zeros(3, dtype=float)

        # p_r = p_uav - p_tgt, v_r = v_uav - v_tgt
        # target wrt UAV is negative of these relative quantities.
        v_tgt_rel_e = -np.asarray(vr_e, dtype=float).reshape(3)
        from utils.math3d import quat_to_R

        R_e_b = quat_to_R(obs.q_eb)
        v_rel_b = R_e_b.T @ v_tgt_rel_e
        return np.array([v_rel_b[1], v_rel_b[2], v_rel_b[0]], dtype=float)

    def compute(self, obs: Observation) -> ControlCommand:
        monitor_step = self._monitor_step
        self._monitor_step += 1
        _, theta, phi = quat_to_euler_ZYX(obs.q_eb)

        if (not obs.has_target) or (obs.p_norm is None):
            thrust = float(np.clip(self.p.mass * self.p.g / max(np.cos(theta), self.p.cos_theta_min), self.p.thrust_min, self.p.thrust_max))
            self._push_monitor("theta", "tab:orange", theta, "angle", obs.t, monitor_step)
            self._push_monitor("theta_d", "black", self.p.theta_th, "angle", obs.t, monitor_step)
            self._push_monitor("theta_th", "tab:red", self.p.theta_th, "angle", obs.t, monitor_step)
            self._push_monitor("phi", "tab:green", phi, "angle", obs.t, monitor_step)
            return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=np.zeros(3, dtype=float))

        ex = float(self.p.foc * obs.p_norm[0])
        ey = float(self.p.foc * obs.p_norm[1])
        theta_d = float(max(self.p.theta_th, np.arctan2(-ey, self.p.foc)))
        phi_d = float(self.p.phi_d)

        v_rel_c = self._relative_velocity_camera(obs)
        c_vy = float(v_rel_c[1])

        omega_c = np.array(
            [
                -self.p.k2 * ey - self.p.k3 * (theta - theta_d),
                self.p.k5 * ex,
                self.p.k6 * (phi - phi_d),
            ],
            dtype=float,
        )
        omega_b = self._R_b_c @ omega_c
        omega_b = clamp_norm(omega_b, self.p.omega_max)

        cos_theta = float(np.clip(np.cos(theta), self.p.cos_theta_min, 1.0))
        thrust = (self.p.mass / cos_theta) * (self.p.k4 * (c_vy - self.p.k1 * ey) + self.p.g)
        thrust = float(np.clip(thrust, self.p.thrust_min, self.p.thrust_max))
        self._push_monitor("ex", "tab:blue", ex, "image_error", obs.t, monitor_step)
        self._push_monitor("ey", "tab:orange", ey, "image_error", obs.t, monitor_step)
        self._push_monitor("theta", "tab:orange", theta, "angle", obs.t, monitor_step)
        #self._push_monitor("atan2(ey, foc)", "yellow", -np.arctan2(ey, self.p.foc), "angle", obs.t, monitor_step)
        self._push_monitor("theta_d", "black", theta_d, "angle", obs.t, monitor_step)
        self._push_monitor("theta_th", "tab:red", self.p.theta_th, "angle", obs.t, monitor_step)
        self._push_monitor("phi", "tab:green", phi, "angle", obs.t, monitor_step)
        self._push_monitor("c_vy", "tab:purple", c_vy, "cam_vel", obs.t, monitor_step)

        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=omega_b)
