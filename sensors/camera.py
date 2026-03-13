# sensors/camera.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from models.state import UAVState, TargetState, CameraMeasurement
from utils.math3d import quat_to_R


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480


@dataclass(slots=True)
class CameraExtrinsics:
    """
    Camera rigidly attached to body with a fixed axis mapping and a mount pitch.

    We define camera frame {c} as:
      x_c: right
      y_c: down
      z_c: forward (optical axis)

    Body frame {b} is FRD:
      x_b: forward
      y_b: right
      z_b: down

    The zero-mount mapping from body vector to camera vector is:
      z_c = x_b
      x_c = y_b
      y_c = z_b

    We additionally support a fixed camera mount pitch relative to the body.
    Positive `mount_pitch_deg` means the camera is pitched downward relative
    to the aircraft forward axis.
    """
    use_default_frd_to_camera: bool = True
    mount_pitch_deg: float = 0.0


class PinholeCamera:
    def __init__(self, K: CameraIntrinsics, ext: CameraExtrinsics | None = None):
        self.K = K
        self.ext = ext if ext is not None else CameraExtrinsics()
        self._R_c_b0 = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _rot_y(angle_rad: float) -> np.ndarray:
        c = float(np.cos(angle_rad))
        s = float(np.sin(angle_rad))
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=float,
        )

    def _body_to_camera(self, v_b: np.ndarray) -> np.ndarray:
        # Positive mount_pitch_deg means camera looks downward, so the camera
        # frame is body-rotated by a negative pitch in FRD coordinates.
        pitch_rad = np.deg2rad(float(self.ext.mount_pitch_deg))
        R_mount = self._rot_y(-pitch_rad)
        v_b_mount = R_mount.T @ np.asarray(v_b, dtype=float).reshape(3)
        return self._R_c_b0 @ v_b_mount

    def measure(self, uav: UAVState, tgt: TargetState, t_meas: float) -> CameraMeasurement:
        # Relative position in world
        p_rel_e = tgt.p_e - uav.p_e

        # World -> body: v_b = R^T v_e   (since R is body->world)
        R_e_b = quat_to_R(uav.q_eb)
        p_rel_b = R_e_b.T @ p_rel_e

        # Body -> camera
        p_rel_c = self._body_to_camera(p_rel_b)
        range_m = float(np.linalg.norm(p_rel_c))

        x, y, z = p_rel_c
        if z <= 1e-6:
            return CameraMeasurement(t_meas=t_meas, valid=False, p_norm=None, uv_px=None, range_m=range_m)

        p_norm = np.array([x / z, y / z], dtype=float)
        u = self.K.fx * p_norm[0] + self.K.cx
        v = self.K.fy * p_norm[1] + self.K.cy

        # FOV check
        valid = (0.0 <= u < self.K.width) and (0.0 <= v < self.K.height)
        if not valid:
            # Keep p_norm even when outside image bounds so controller can still
            # steer back toward the target direction; `valid` indicates in-frame status.
            return CameraMeasurement(
                t_meas=t_meas,
                valid=False,
                p_norm=p_norm,
                uv_px=np.array([u, v], dtype=float),
                range_m=range_m,
            )

        return CameraMeasurement(
            t_meas=t_meas,
            valid=True,
            p_norm=p_norm,
            uv_px=np.array([u, v], dtype=float),
            range_m=range_m,
        )
