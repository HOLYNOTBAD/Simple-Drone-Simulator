# utils/math3d.py
"""
3D math utilities for the IBVS interception simulation.

Frames / conventions (IMPORTANT):
- World frame {e}: NED  (x=N, y=E, z=D)
- Body  frame {b}: FRD  (x=F, y=R, z=D)
Both are right-handed.

Quaternion convention:
- q = [w, x, y, z]  (scalar-first)
- q_eb represents rotation from body {b} to world {e}:
    v_e = R_e_b(q_eb) @ v_b

Angular rate convention:
- w_b is angular velocity expressed in body frame {b} (FRD), rad/s.
Quaternion kinematics for q_eb with body-rate w_b:
    q_dot = 0.5 * q ⊗ [0, w_b]
So discrete update:
    q_{k+1} = q_k ⊗ delta_q(w_b * dt)

Euler angles:
- You asked for XYX order.
  Here we define the Euler angles (a, b, c) as an *intrinsic* (body-fixed) rotation:
      R = R_x(a) @ R_y(b) @ R_x(c)
  producing a rotation matrix from {b} to {e}.

Note:
- In aerospace, ZYX (yaw-pitch-roll) is more common, but we follow your XYX request.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

Array = np.ndarray


# -----------------------------
# Small helpers
# -----------------------------

def clamp_norm(x: Array, max_norm: float, eps: float = 1e-12) -> Array:
    """Scale vector x down to have norm <= max_norm (direction preserved)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = np.linalg.norm(x)
    if n <= max_norm or n < eps:
        return x
    return x * (max_norm / n)


def normalize(x: Array, eps: float = 1e-12) -> Array:
    """Normalize a vector (no-op if too small)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


# -----------------------------
# SO(3): hat / vee
# -----------------------------

def hat(w: Array) -> Array:
    """Skew-symmetric matrix [w]_x such that [w]_x v = w × v."""
    w = np.asarray(w, dtype=float).reshape(3)
    wx, wy, wz = w
    return np.array([[0.0, -wz,  wy],
                     [wz,  0.0, -wx],
                     [-wy, wx,  0.0]], dtype=float)


def vee(W: Array) -> Array:
    """Inverse of hat: extract vector from a skew-symmetric matrix."""
    W = np.asarray(W, dtype=float)
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)


# -----------------------------
# Quaternions
# -----------------------------

def quat_normalize(q: Array, eps: float = 1e-12) -> Array:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < eps:
        # Fallback to identity rotation
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_conj(q: Array) -> Array:
    """Quaternion conjugate."""
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(q1: Array, q2: Array) -> Array:
    """
    Hamilton product q = q1 ⊗ q2 (scalar-first).
    """
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(4)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_to_R(q_eb: Array) -> Array:
    """
    Convert quaternion q_eb (body->world) to rotation matrix R_e_b.
    """
    q = quat_normalize(q_eb)
    w, x, y, z = q

    # Standard scalar-first quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def R_to_quat(R: Array) -> Array:
    """
    Convert rotation matrix to quaternion (scalar-first).
    Returns q_eb (body->world).
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # Find the major diagonal element
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    return quat_normalize(np.array([w, x, y, z], dtype=float))


def integrate_quat_body_rate(q_eb: Array, w_b: Array, dt: float, eps: float = 1e-12) -> Array:
    """
    Integrate quaternion q_eb forward by dt using body-rate w_b (FRD), rad/s.

    Uses exponential map:
      delta_theta = ||w|| * dt
      delta_q = [cos(delta/2), axis*sin(delta/2)]
      q_next = q ⊗ delta_q
    """
    q = quat_normalize(q_eb)
    w = np.asarray(w_b, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w) * dt)

    if theta < 1e-10:
        # Small-angle approximation: sin(theta/2) ~ theta/2
        half = 0.5 * dt
        dq = np.array([1.0, half*w[0], half*w[1], half*w[2]], dtype=float)
        return quat_normalize(quat_mul(q, dq), eps=eps)

    axis = w / (np.linalg.norm(w) + eps)
    half_theta = 0.5 * theta
    dq = np.array([
        np.cos(half_theta),
        axis[0] * np.sin(half_theta),
        axis[1] * np.sin(half_theta),
        axis[2] * np.sin(half_theta),
    ], dtype=float)

    return quat_normalize(quat_mul(q, dq), eps=eps)


def rotate_b_to_e(q_eb: Array, v_b: Array) -> Array:
    """Rotate vector from body {b} to world {e} using q_eb."""
    R = quat_to_R(q_eb)
    v_b = np.asarray(v_b, dtype=float).reshape(3)
    return R @ v_b


def rotate_e_to_b(q_eb: Array, v_e: Array) -> Array:
    """Rotate vector from world {e} to body {b} using q_eb."""
    R = quat_to_R(q_eb)
    v_e = np.asarray(v_e, dtype=float).reshape(3)
    return R.T @ v_e


# -----------------------------
# Euler XYX (intrinsic)
# -----------------------------

def _Rx(a: float) -> Array:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, ca, -sa],
                     [0.0, sa,  ca]], dtype=float)


def _Ry(b: float) -> Array:
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[ cb, 0.0, sb],
                     [0.0, 1.0, 0.0],
                     [-sb, 0.0, cb]], dtype=float)

# -----------------------------
# Euler ZYX (intrinsic): yaw-pitch-roll
# -----------------------------

def _Rx(roll: float) -> Array:
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, cr, -sr],
                     [0.0, sr,  cr]], dtype=float)

def _Ry(pitch: float) -> Array:
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[ cp, 0.0, sp],
                     [0.0, 1.0, 0.0],
                     [-sp, 0.0, cp]], dtype=float)

def _Rz(yaw: float) -> Array:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0.0],
                     [sy,  cy, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def euler_ZYX_to_R(yaw: float, pitch: float, roll: float) -> Array:
    """
    Intrinsic ZYX (yaw-pitch-roll) Euler angles to rotation matrix.

    Definition (intrinsic rotations about body axes):
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Returns:
        R_e_b (body->world) consistent with q_eb convention.
    """
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)

def euler_ZYX_to_quat(yaw: float, pitch: float, roll: float) -> Array:
    """Intrinsic ZYX Euler angles -> quaternion q_eb (scalar-first)."""
    return R_to_quat(euler_ZYX_to_R(yaw, pitch, roll))

def quat_to_euler_ZYX(q_eb: Array) -> Tuple[float, float, float]:
    """
    Quaternion q_eb -> intrinsic ZYX Euler angles (yaw, pitch, roll) such that:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Notes:
    - Singularity at pitch = ±pi/2 (gimbal lock).
    - We return one valid solution.
    """
    R = quat_to_R(q_eb)

    # For R = Rz(yaw) Ry(pitch) Rx(roll):
    # pitch = asin(-R31)  (with 0-index: R[2,0])
    # yaw   = atan2(R21, R11)  (R[1,0], R[0,0])
    # roll  = atan2(R32, R33)  (R[2,1], R[2,2])
    r20 = R[2, 0]
    pitch = float(np.arcsin(np.clip(-r20, -1.0, 1.0)))

    cp = np.cos(pitch)
    if abs(cp) < 1e-9:
        # Gimbal lock: pitch ~ ±90deg
        # yaw and roll are coupled; set roll=0 and solve yaw from R12/R22
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
        return yaw, pitch, roll

    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll