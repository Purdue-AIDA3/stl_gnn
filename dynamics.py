"""
Dynamics models used by the planning and learning pipeline.

Currently implemnted:
- 2D double integrator (discrete-time)
    state x = [px, py, vx, vy]
    control u = [ax, ay]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DoubleIntegrator2D:
    dt: float

    def matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time dynamics:
        x_{t+1} = A x_t + B u_t
        """
        dt = float(self.dt)
        A = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        B = np.array(
            [
                [0.5 * dt**2, 0],
                [0, 0.5 * dt**2],
                [dt, 0],
                [0, dt],
            ],
            dtype=float,
        )
        return A, B
    
    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Roll out the dynamics forward given initial state and control sequence.

        Args:
            x0: Initial state, shape (4,)
            U: Control sequence, shape (T, 2)

        Returns:
            X: State trajectory, shape (T+1, 4) where X[0] = x0
        """
        x0 = np.asarray(x0, dtype=float).reshape(4)
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError(f"Control sequence U must have shape (T, 2), got {U.shape}")        
        
        A, B = self.matrices()
        T = int(U.shape[0])

        X = np.zeros((T + 1, 4), dtype=float)
        X[0] = x0
        for k in range(T):
            X[k + 1] = A @ X[k] + B @ U[k]
        return X
    

__all__ = [
    "DoubleIntegrator2D",
]