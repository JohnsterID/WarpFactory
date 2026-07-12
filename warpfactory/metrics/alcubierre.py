"""Alcubierre warp drive metric."""

from typing import Dict

import numpy as np

from .base import Metric


class AlcubierreMetric(Metric):
    """Alcubierre warp drive metric."""

    def calculate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float,
        v_s: float = 2.0,
        R: float = 1.0,
        sigma: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """Calculate metric components.

        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinates
        t : float
            Time coordinate
        v_s : float
            Ship velocity (in c)
        R : float
            Radius of warp bubble
        sigma : float
            Thickness parameter

        Returns
        -------
        Dict[str, np.ndarray]
            Metric components
        """
        x_s = v_s * t
        r = np.sqrt((x - x_s) ** 2 + y**2 + z**2)
        f = self.shape_function(r, R, sigma)

        v_x = v_s * f
        g_tt = -(1 - v_x**2)
        g_tx = -v_x

        return {
            "g_tt": g_tt,
            "g_tx": g_tx,
            "g_xx": np.ones_like(x),
            "g_yy": np.ones_like(x),
            "g_zz": np.ones_like(x),
        }

    @staticmethod
    def shape_function(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
        """Canonical Alcubierre top-hat shape function.

        f(r) = [tanh(sigma(r + R)) - tanh(sigma(r - R))] / [2 tanh(sigma R)]

        as defined in Alcubierre (1994) and used by the MATLAB
        shapeFunction_Alcubierre.m: f -> 1 inside the bubble,
        f -> 0 far outside.
        """
        return (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / (
            2 * np.tanh(sigma * R)
        )
