# utils/metrics.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class MetricsConfig:
    hit_radius: float = 0.5


class Metrics:
    """
    Basic metrics for interception:
      - distance(t)
      - min distance
      - hit condition and hit time
    """

    def __init__(self, cfg: MetricsConfig):
        self.cfg = cfg
        self.min_dist = float("inf")
        self.hit = False
        self.t_hit = None

    def update(self, t: float, uav_p: np.ndarray, tgt_p: np.ndarray) -> float:
        dist = float(np.linalg.norm(tgt_p - uav_p))
        if dist < self.min_dist:
            self.min_dist = dist
        if (not self.hit) and dist <= self.cfg.hit_radius:
            self.hit = True
            self.t_hit = t
        return dist

    def summary(self) -> dict:
        return {
            "hit": bool(self.hit),
            "t_hit": None if self.t_hit is None else float(self.t_hit),
            "min_dist": float(self.min_dist),
            "hit_radius": float(self.cfg.hit_radius),
        }