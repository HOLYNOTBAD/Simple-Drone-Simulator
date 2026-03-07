# utils/log.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np


@dataclass(slots=True)
class RunMeta:
    name: str = "l1_run"
    seed: int = 0


class NPZLogger:
    """
    Simple logger that collects arrays and saves to a .npz file.

    Usage:
      logger = NPZLogger(run_dir="runs")
      logger.push("t", t)
      logger.push("uav_p", p_vec3)
      ...
      path = logger.save(meta_dict)
    """

    def __init__(self, run_dir: str = "runs"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, list[np.ndarray]] = {}

    def push(self, key: str, value) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(np.asarray(value))

    def to_arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for k, vs in self.data.items():
            try:
                out[k] = np.asarray(vs)
            except Exception:
                # fallback for ragged objects
                out[k] = np.array(vs, dtype=object)
        return out

    def save(self, meta: dict | None = None, filename: str | None = None) -> str:
        arrs = self.to_arrays()
        meta = meta or {}

        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}.npz"

        path = self.run_dir / filename
        np.savez_compressed(path, **arrs, __meta__=meta)
        return str(path)