# sim/scheduler.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RateConfig:
    physics_hz: float = 1000.0
    control_hz: float = 200.0
    camera_hz: float = 30.0
    visualization_hz: float = 10.0


class MultiRateScheduler:
    """
    Simple step-based multi-rate scheduler.

    Base time step dt is chosen by physics_hz: dt = 1/physics_hz.
    Other loops run every N steps:
      N_control = round(physics_hz / control_hz)
      N_camera  = round(physics_hz / camera_hz)
      N_visual  = round(physics_hz / visualization_hz)
    """

    def __init__(self, rates: RateConfig):
        self.r = rates
        self.dt = 1.0 / self.r.physics_hz

        self.n_control = max(1, int(round(self.r.physics_hz / self.r.control_hz)))
        self.n_camera = max(1, int(round(self.r.physics_hz / self.r.camera_hz)))
        self.n_visualization = max(1, int(round(self.r.physics_hz / self.r.visualization_hz)))

    def should_control(self, step: int) -> bool:
        return (step % self.n_control) == 0

    def should_camera(self, step: int) -> bool:
        return (step % self.n_camera) == 0

    def should_visualization(self, step: int) -> bool:
        return (step % self.n_visualization) == 0
