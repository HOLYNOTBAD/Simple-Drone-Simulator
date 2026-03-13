# observe/obj_tracker.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.state import CameraMeasurement


@dataclass(slots=True)
class Bbox:
    """
    Simple image-space bounding box description.

    Attributes:
    - `u`, `v`: target center in pixel coordinates
    - `bw`, `bh`: target width and height in pixels
    """

    u: float
    v: float
    bw: float
    bh: float

    def as_array(self) -> np.ndarray:
        return np.array([self.u, self.v, self.bw, self.bh], dtype=float)


@dataclass(slots=True)
class ObjTrackerParams:
    fx: float = 320.0
    fy: float = 320.0
    target_width_m: float = 1.0
    target_height_m: float = 1.0
    min_range_m: float = 1e-6


class ObjTracker:
    """
    Convert a `CameraMeasurement` into a simple `Bbox` estimate.

    This tracker is geometry-based rather than detector-based:
    - box center comes from `cam.uv_px`
    - box size comes from pinhole scaling using target physical size and range

    Approximation used:
        bw ≈ fx * target_width_m / range_m
        bh ≈ fy * target_height_m / range_m

    Optical distortion and perspective deformation are intentionally ignored.
    """

    def __init__(self, p: ObjTrackerParams):
        self.p = p

    def reset(self) -> None:
        pass

    def track(self, cam: CameraMeasurement) -> Bbox | None:
        if cam.uv_px is None or cam.range_m is None:
            return None

        uv_px = np.asarray(cam.uv_px, dtype=float).reshape(2)
        range_m = max(float(cam.range_m), self.p.min_range_m)
        bw = float(self.p.fx * self.p.target_width_m / range_m)
        bh = float(self.p.fy * self.p.target_height_m / range_m)

        return Bbox(
            u=float(uv_px[0]),
            v=float(uv_px[1]),
            bw=max(bw, 0.0),
            bh=max(bh, 0.0),
        )
