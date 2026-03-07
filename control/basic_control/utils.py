from __future__ import annotations

import numpy as np


def clamp_norm(x: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < eps or n <= max_norm:
        return x
    return x * (max_norm / n)


def sat(x: np.ndarray, lo, hi) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def anti_windup_clip(x: np.ndarray, lim: np.ndarray) -> np.ndarray:
    lim = np.asarray(lim, dtype=float)
    return np.clip(np.asarray(x, dtype=float), -lim, lim)

