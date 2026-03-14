# sensors/camera.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np

from models.state import UAVState, TargetState, CameraMeasurement
from utils.math3d import quat_to_R


@dataclass(slots=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 480
    fx: float | None = 320.0
    fy: float | None = 320.0
    cx: float | None = None
    cy: float | None = None
    fov_x_deg: float | None = None
    fov_y_deg: float | None = None

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera image size must be positive")

        self.fx = self._resolve_focal(self.fx, self.fov_x_deg, self.width)
        self.fy = self._resolve_focal(self.fy, self.fov_y_deg, self.height)
        if self.fx is None and self.fy is None:
            self.fx = self.width / 2.0
            self.fy = self.fx
        elif self.fx is None:
            self.fx = float(self.fy)
        elif self.fy is None:
            self.fy = float(self.fx)

        self.fx = float(self.fx)
        self.fy = float(self.fy)
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive")

        self.cx = self.width / 2.0 if self.cx is None else float(self.cx)
        self.cy = self.height / 2.0 if self.cy is None else float(self.cy)

    @staticmethod
    def _resolve_focal(focal: float | None, fov_deg: float | None, image_size_px: int) -> float | None:
        if focal is not None:
            return float(focal)
        if fov_deg is None:
            return None
        fov_rad = np.deg2rad(float(fov_deg))
        if not 0.0 < fov_rad < np.pi:
            raise ValueError("Camera FOV must be in (0, 180) degrees")
        return float(image_size_px / (2.0 * np.tan(fov_rad / 2.0)))


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

    We additionally support a fixed camera mount pitch relative to the body,
    a full body->camera rotation, and a camera origin offset in body frame.
    Positive `mount_pitch_deg` means the camera is pitched downward relative
    to the aircraft forward axis.
    """
    use_default_frd_to_camera: bool = True
    mount_pitch_deg: float = 0.0
    R_cb: np.ndarray | None = None
    t_cb_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        if self.R_cb is not None:
            self.R_cb = np.asarray(self.R_cb, dtype=float).reshape(3, 3)
            if not np.all(np.isfinite(self.R_cb)):
                raise ValueError("CameraExtrinsics.R_cb must be finite")
        self.t_cb_b = np.asarray(self.t_cb_b, dtype=float).reshape(3)
        if not np.all(np.isfinite(self.t_cb_b)):
            raise ValueError("CameraExtrinsics.t_cb_b must be finite")


@dataclass(slots=True)
class CameraMeasurementConfig:
    noise_std_px: float = 0.0
    noise_std_range_scale_px: float = 0.0
    detection_prob: float = 1.0
    detection_range_decay_m: float = 0.0
    delay_s: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.noise_std_px = float(self.noise_std_px)
        self.noise_std_range_scale_px = float(self.noise_std_range_scale_px)
        self.detection_prob = float(self.detection_prob)
        self.detection_range_decay_m = float(self.detection_range_decay_m)
        self.delay_s = float(self.delay_s)
        if self.noise_std_px < 0.0:
            raise ValueError("CameraMeasurementConfig.noise_std_px must be >= 0")
        if self.noise_std_range_scale_px < 0.0:
            raise ValueError("CameraMeasurementConfig.noise_std_range_scale_px must be >= 0")
        if not 0.0 <= self.detection_prob <= 1.0:
            raise ValueError("CameraMeasurementConfig.detection_prob must be in [0, 1]")
        if self.detection_range_decay_m < 0.0:
            raise ValueError("CameraMeasurementConfig.detection_range_decay_m must be >= 0")
        if self.delay_s < 0.0:
            raise ValueError("CameraMeasurementConfig.delay_s must be >= 0")


class PinholeCamera:
    def __init__(
        self,
        K: CameraIntrinsics,
        ext: CameraExtrinsics | None = None,
        meas_cfg: CameraMeasurementConfig | None = None,
    ):
        self.K = K
        self.ext = ext if ext is not None else CameraExtrinsics()
        self.meas_cfg = meas_cfg if meas_cfg is not None else CameraMeasurementConfig()
        self._R_c_b0 = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self._R_c_b = self._build_body_to_camera_rotation()
        self._delay_queue: deque[CameraMeasurement] = deque()
        self._rng = None if self.meas_cfg.seed is None else np.random.default_rng(self.meas_cfg.seed)

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

    def _build_body_to_camera_rotation(self) -> np.ndarray:
        if self.ext.R_cb is not None:
            return self.ext.R_cb

        R_base = self._R_c_b0 if self.ext.use_default_frd_to_camera else np.eye(3, dtype=float)
        pitch_rad = np.deg2rad(float(self.ext.mount_pitch_deg))
        R_mount = self._rot_y(-pitch_rad)
        return R_base @ R_mount.T

    def _body_to_camera(self, v_b: np.ndarray) -> np.ndarray:
        v_b = np.asarray(v_b, dtype=float).reshape(3)
        return self._R_c_b @ (v_b - self.ext.t_cb_b)

    def _pixel_noise_std(self, range_m: float) -> float:
        return self.meas_cfg.noise_std_px + self.meas_cfg.noise_std_range_scale_px * max(range_m, 0.0)

    def _detect_target(self, range_m: float) -> bool:
        p_detect = self.meas_cfg.detection_prob
        if self.meas_cfg.detection_range_decay_m > 0.0:
            p_detect *= float(np.exp(-range_m / self.meas_cfg.detection_range_decay_m))
        p_detect = float(np.clip(p_detect, 0.0, 1.0))
        rnd = float(np.random.random()) if self._rng is None else float(self._rng.random())
        return bool(rnd <= p_detect)

    def _measure_now(self, uav: UAVState, tgt: TargetState, t_meas: float) -> CameraMeasurement:
        # Relative position in world
        p_rel_e = tgt.p_e - uav.p_e

        # World -> body: v_b = R^T v_e   (since R is body->world)
        R_e_b = quat_to_R(uav.q_eb)
        p_rel_b = R_e_b.T @ p_rel_e

        # Body -> camera
        p_rel_c = self._body_to_camera(p_rel_b)
        range_m = float(np.linalg.norm(p_rel_c))
        bearing_c = None
        if range_m > 1e-12:
            bearing_c = p_rel_c / range_m

        x, y, z = p_rel_c
        if z <= 0.0:
            return CameraMeasurement(
                t_meas=t_meas,
                p_cam=p_rel_c,
                bearing_c=bearing_c,
                valid=False,
                p_norm=None,
                uv_px=None,
                range_m=range_m,
            )

        z_safe = max(float(z), 1e-6)
        u = self.K.fx * (x / z_safe) + self.K.cx
        v = self.K.fy * (y / z_safe) + self.K.cy

        sigma_px = self._pixel_noise_std(range_m)
        if sigma_px > 0.0:
            if self._rng is None:
                u += float(np.random.normal(0.0, sigma_px))
                v += float(np.random.normal(0.0, sigma_px))
            else:
                u += float(self._rng.normal(0.0, sigma_px))
                v += float(self._rng.normal(0.0, sigma_px))

        p_norm = np.array(
            [
                (u - self.K.cx) / self.K.fx,
                (v - self.K.cy) / self.K.fy,
            ],
            dtype=float,
        )

        if not self._detect_target(range_m):
            return CameraMeasurement(
                t_meas=t_meas,
                p_cam=p_rel_c,
                bearing_c=bearing_c,
                valid=False,
                p_norm=None,
                uv_px=None,
                range_m=range_m,
            )

        # FOV check
        valid = (0.0 <= u < self.K.width) and (0.0 <= v < self.K.height)
        if not valid:
            # Keep p_norm even when outside image bounds so controller can still
            # steer back toward the target direction; `valid` indicates in-frame status.
            return CameraMeasurement(
                t_meas=t_meas,
                p_cam=p_rel_c,
                bearing_c=bearing_c,
                valid=False,
                p_norm=p_norm,
                uv_px=np.array([u, v], dtype=float),
                range_m=range_m,
            )

        return CameraMeasurement(
            t_meas=t_meas,
            p_cam=p_rel_c,
            bearing_c=bearing_c,
            valid=True,
            p_norm=p_norm,
            uv_px=np.array([u, v], dtype=float),
            range_m=range_m,
        )

    def measure(self, uav: UAVState, tgt: TargetState, t_meas: float) -> CameraMeasurement | None:
        meas = self._measure_now(uav, tgt, t_meas=t_meas)
        if self.meas_cfg.delay_s <= 0.0:
            return meas

        self._delay_queue.append(meas)
        ready_meas = None
        cutoff_t = float(t_meas - self.meas_cfg.delay_s)
        while self._delay_queue and self._delay_queue[0].t_meas <= cutoff_t + 1e-12:
            ready_meas = self._delay_queue.popleft()
        return ready_meas


def build_camera_from_config(cam_cfg: dict | None) -> PinholeCamera:
    cam_cfg = {} if cam_cfg is None else cam_cfg
    intrinsics = CameraIntrinsics(
        width=int(cam_cfg.get("width", 640)),
        height=int(cam_cfg.get("height", 480)),
        fx=None if cam_cfg.get("fx") is None else float(cam_cfg.get("fx")),
        fy=None if cam_cfg.get("fy") is None else float(cam_cfg.get("fy")),
        cx=None if cam_cfg.get("cx") is None else float(cam_cfg.get("cx")),
        cy=None if cam_cfg.get("cy") is None else float(cam_cfg.get("cy")),
        fov_x_deg=None if cam_cfg.get("fov_x_deg") is None else float(cam_cfg.get("fov_x_deg")),
        fov_y_deg=None if cam_cfg.get("fov_y_deg") is None else float(cam_cfg.get("fov_y_deg")),
    )
    extrinsics = CameraExtrinsics(
        use_default_frd_to_camera=bool(cam_cfg.get("use_default_frd_to_camera", True)),
        mount_pitch_deg=float(cam_cfg.get("mount_pitch_deg", 0.0)),
        R_cb=None if cam_cfg.get("R_cb") is None else np.asarray(cam_cfg.get("R_cb"), dtype=float),
        t_cb_b=np.asarray(cam_cfg.get("t_cb_b", cam_cfg.get("t_cb", [0.0, 0.0, 0.0])), dtype=float),
    )
    meas_cfg = CameraMeasurementConfig(
        noise_std_px=float(cam_cfg.get("noise_std_px", 0.0)),
        noise_std_range_scale_px=float(cam_cfg.get("noise_std_range_scale_px", 0.0)),
        detection_prob=float(cam_cfg.get("detection_prob", 1.0)),
        detection_range_decay_m=float(cam_cfg.get("detection_range_decay_m", 0.0)),
        delay_s=float(cam_cfg.get("delay_s", 0.0)),
        seed=None if cam_cfg.get("seed") is None else int(cam_cfg.get("seed")),
    )
    return PinholeCamera(intrinsics, extrinsics, meas_cfg)
