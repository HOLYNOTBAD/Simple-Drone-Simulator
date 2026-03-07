# control/controller_base.py
from __future__ import annotations

from abc import ABC, abstractmethod

from models.state import Observation, ControlCommand


class ControllerBase(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset internal states (if any)."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, obs: Observation) -> ControlCommand:
        """Compute control command from observation."""
        raise NotImplementedError