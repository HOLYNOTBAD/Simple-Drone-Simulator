# control/ibvs_so3_controller.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from control.controller_base import ControllerBase
from models.state import Observation, ControlCommand
from utils.math3d import clamp_norm, quat_to_R


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _skew(x: np.ndarray) -> np.ndarray:
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    return np.array(
        [[0.0, -x3,  x2],
         [x3,  0.0, -x1],
         [-x2, x1,  0.0]],
        dtype=float,
    )


def _vex(S: np.ndarray) -> np.ndarray:
    # For S = [x]x, vex(S) = x
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def _rodrigues(axis_unit: np.ndarray, angle: float) -> np.ndarray:
    # R = I + [r]x sin(phi) + [r]x^2 (1-cos(phi))
    K = _skew(axis_unit)
    I = np.eye(3, dtype=float)
    s = float(np.sin(angle))
    c = float(np.cos(angle))
    return I + K * s + (K @ K) * (1.0 - c)


@dataclass
class IBVSSO3ControllerParams:
    """
    Paper-style high-speed interception controller (IBVS).

    World: NED
    Body:  FRD
    Camera: assumed strapdown, and image normalized coord p_norm = [x/z, y/z]
            (x right, y down), and camera forward aligns with body +x.

    Required in Observation for full paper behavior:
      - p_r: relative position (interceptor - target) in earth/NED, shape (3,)
      - v_r: relative velocity (interceptor - target) in earth/NED, shape (3,)

    Optional:
      - is_new_image: bool, indicates a fresh camera update (multi-rate logic)
    """
    mass: float = 1.0
    g: float = 9.81

    # Collinear controller gains (Eq. 19)
    k1: float = 1.0       # velocity/position coupling (1/s)
    k2: float = 2.0       # z2 damping (1/s)
    k_p: float = 1.0      # position gain (1/s^2)  (paper writes "-pr", i.e., k_p=1)

    # Barrier parameter for LOS constraint (Eq. 12/13): require |z1| < kb
    kb: float = 0.9

    # Saturations (Eq. 23/28/29)
    omega_max: float = 6.0      # ωm
    thrust_max: float = 30.0    # fm (N)

    # Numerics / safety
    eps: float = 1e-6
    # Reacquisition gains when target is temporarily out of FOV
    reacq_k_yaw: float = 2.5
    reacq_k_pitch: float = 2.0
    reacq_forward_bias: float = 0.5  # N
    fov_guard_margin: float = 0.7    # normalized image margin
    fov_guard_k_yaw: float = 3.0
    fov_guard_k_pitch: float = 2.5


class IBVSSO3Controller(ControllerBase):
    """
    Implements:
      ω1 (Eq. 13), ad (Eq. 19), nfd (Eq. 21), Rd (Eq. 22),
      fd (Eq. 23), ω2 (Eq. 26), ωd+fd saturation (Eq. 28/29).

    Multi-rate (Algorithm 1 idea):
      - Always compute inner-loop (ω2, fd) using stored (ad, Rd)
      - Update (ω1, ad, Rd) only when a NEW image arrives
    If Observation does not provide is_new_image, we treat every call as new.
    """
    def __init__(self, p: IBVSSO3ControllerParams):
        self.p = p
        self._ad = np.zeros(3, dtype=float)
        self._Rd = np.eye(3, dtype=float)
        self._omega1_b = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self._ad = np.zeros(3, dtype=float)
        self._Rd = np.eye(3, dtype=float)
        self._omega1_b = np.zeros(3, dtype=float)

    def _hover_thrust(self, R_e_b: np.ndarray) -> float:
        # Same idea as your old code: thrust = m*g / c_zz, where c_zz = R[2,2]
        c_zz = float(np.clip(R_e_b[2, 2], 0.2, 1.0))
        return self.p.mass * self.p.g / c_zz

    def compute(self, obs: Observation) -> ControlCommand:
        R_e_b = quat_to_R(obs.q_eb)

        # Relative states (paper convention in this project):
        # p_r = p_uav - p_tgt, so vector from UAV to target is -p_r.
        pr_e = getattr(obs, "p_r", None)
        vr_e = getattr(obs, "v_r", None)
        if pr_e is None:
            pr_e = getattr(obs, "pr_e", None)  # backward compatibility
        if vr_e is None:
            vr_e = getattr(obs, "vr_e", None)  # backward compatibility
        have_rel = (pr_e is not None) and (vr_e is not None)

        # No target -> actively reacquire with relative geometry (if available)
        if (not obs.has_target) or (obs.p_norm is None):
            thrust = self._hover_thrust(R_e_b)
            if have_rel:
                pr_e = np.asarray(pr_e, dtype=float).reshape(3)
                # target direction in body frame
                dir_b = R_e_b.T @ (-pr_e)
                x_fwd = float(dir_b[0])
                y_right = float(dir_b[1])
                z_down = float(dir_b[2])

                if x_fwd > self.p.eps:
                    x_img = y_right / x_fwd
                    y_img = z_down / x_fwd
                    omega_cmd_b = np.array(
                        [0.0, self.p.reacq_k_pitch * y_img, self.p.reacq_k_yaw * x_img],
                        dtype=float,
                    )
                    # Slight forward thrust bias helps bring target back to center quickly.
                    thrust = min(thrust + self.p.reacq_forward_bias, self.p.thrust_max)
                else:
                    # Target is behind: spin towards side with larger bearing.
                    spin = np.sign(y_right) if abs(y_right) > self.p.eps else 1.0
                    omega_cmd_b = np.array([0.0, 0.0, 0.8 * self.p.omega_max * spin], dtype=float)

                omega_cmd_b = clamp_norm(omega_cmd_b, self.p.omega_max)
                return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=omega_cmd_b)

            return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=np.zeros(3, dtype=float))

        # --- Get nt (target unit LOS vector in earth frame) ---
        # In camera: dir_c ~ [x, y, 1]. With strapdown mapping camera->body:
        # body dir_b ~ [1, x, y] (camera forward -> body +x, camera right -> body +y, camera down -> body +z)
        x_img = float(obs.p_norm[0])
        y_img = float(obs.p_norm[1])

        dir_b = _normalize(np.array([1.0, x_img, y_img], dtype=float), self.p.eps)
        nt_e = _normalize(R_e_b @ dir_b, self.p.eps)

        # Designed LOS vector ntd: choose optical axis (center of image)
        ntd_e = _normalize(R_e_b @ np.array([1.0, 0.0, 0.0], dtype=float), self.p.eps)

        # Multi-rate flag
        is_new_image = bool(getattr(obs, "is_new_image", True))

        if is_new_image and have_rel:
            pr_e = np.asarray(pr_e, dtype=float).reshape(3)
            vr_e = np.asarray(vr_e, dtype=float).reshape(3)

            r = float(np.linalg.norm(pr_e))
            r = max(r, self.p.eps)

            # z1 = 1 - ntd^T nt  (Eq. in Sec III-B)
            z1 = float(1.0 - np.dot(ntd_e, nt_e))

            # Barrier coefficient: z1/(kb^2 - z1^2)
            kb2 = float(self.p.kb * self.p.kb)
            denom = float(max(kb2 - z1 * z1, self.p.eps))
            coeff = float(z1 / denom)

            # ω1 in BODY (Eq. 13): bω1 = coeff * Reb^T (ntd × nt)
            cross_e = np.cross(ntd_e, nt_e)
            self._omega1_b = coeff * (R_e_b.T @ cross_e)

            # z2 = vr - vrd, with vrd = -k1 pr  => z2 = vr + k1 pr  (Eq. 16/17)
            z2 = vr_e + self.p.k1 * pr_e

            # (-I + nt nt^T) ntd
            I = np.eye(3, dtype=float)
            proj = (-I + np.outer(nt_e, nt_e)) @ ntd_e

            # desired acceleration ad (Eq. 19, treat eat as disturbance -> ignore)
            self._ad = (
                -self.p.k1 * vr_e
                -self.p.k2 * z2
                -self.p.k_p * pr_e
                + coeff * (self.p.mass / r) * proj
            )

            # Compute nfd (Eq. 21) with no drag
            g_e = np.array([0.0, 0.0, self.p.g], dtype=float)
            nfd_e = _normalize(self._ad - g_e, self.p.eps)

            # nf: current controllable force direction in earth frame.
            # In FRD, thrust acts along -z_b => nf = Reb * [0,0,-1]
            nf_e = _normalize(R_e_b @ np.array([0.0, 0.0, -1.0], dtype=float), self.p.eps)

            # Rd (Eq. 22): Rd = Rtilt * Reb, with axis r = nf × nfd, angle phi = arccos(nf^T nfd)
            axis = np.cross(nf_e, nfd_e)
            axis_n = float(np.linalg.norm(axis))
            if axis_n < self.p.eps:
                Rtilt = np.eye(3, dtype=float)
            else:
                axis_u = axis / axis_n
                c = float(np.clip(np.dot(nf_e, nfd_e), -1.0, 1.0))
                phi = float(np.arccos(c))
                Rtilt = _rodrigues(axis_u, phi)

            self._Rd = Rtilt @ R_e_b

        # FOV guard: when estimated target bearing is near image edge, prioritize recentering.
        if have_rel:
            pr_e_arr = np.asarray(pr_e, dtype=float).reshape(3)
            dir_b = R_e_b.T @ (-pr_e_arr)
            x_fwd = float(dir_b[0])
            if x_fwd > self.p.eps:
                x_hat = float(dir_b[1] / x_fwd)
                y_hat = float(dir_b[2] / x_fwd)
                if max(abs(x_hat), abs(y_hat)) > self.p.fov_guard_margin:
                    omega_guard = np.array(
                        [0.0, self.p.fov_guard_k_pitch * y_hat, self.p.fov_guard_k_yaw * x_hat],
                        dtype=float,
                    )
                    omega_guard = clamp_norm(omega_guard, self.p.omega_max)
                    thrust_guard = self._hover_thrust(R_e_b)
                    return ControlCommand(t=obs.t, thrust=thrust_guard, omega_cmd_b=omega_guard)

        # --- Inner-loop every tick (Eq. 23/26/28) ---

        # ω2 (Eq. 26): bω2 = -vex(Rd^T Reb - Reb^T Rd)
        E = (self._Rd.T @ R_e_b) - (R_e_b.T @ self._Rd)
        omega2_b = -_vex(E)

        # ωd (Eq. 28): sat(ω1 + ω2, ωm)
        omega_cmd_b = clamp_norm(self._omega1_b + omega2_b, self.p.omega_max)

        # fd (Eq. 23): fd = min(max(nf^T (m ad - m g - fdrag), 0), fm)
        g_e = np.array([0.0, 0.0, self.p.g], dtype=float)
        nf_e = _normalize(R_e_b @ np.array([0.0, 0.0, -1.0], dtype=float), self.p.eps)
        thrust = float(np.dot(nf_e, self.p.mass * (self._ad - g_e)))
        thrust = float(np.clip(thrust, 0.0, self.p.thrust_max))

        # If we don't have relative states, at least hover (avoid “free fall”)
        if not have_rel:
            thrust = min(self._hover_thrust(R_e_b), self.p.thrust_max)

        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=omega_cmd_b)


# Backward compatibility while the rest of the codebase is being renamed.
IBVSController = IBVSSO3Controller
IBVSControllerParams = IBVSSO3ControllerParams
IBVSControllerL1 = IBVSSO3Controller
