"""Minkowski (flat) spacetime metric."""

import numpy as np
from .base import Metric

class MinkowskiMetric(Metric):
    """Minkowski metric in Cartesian coordinates."""
    
    def calculate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 t: float, **kwargs) -> dict:
        """Calculate Minkowski metric components.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinates
        t : float
            Time coordinate
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        dict
            Metric components
        """
        # Diagonal components
        g_tt = -np.ones_like(x)  # Time-time component
        g_xx = np.ones_like(x)   # Space-space components
        
        return {
            "g_tt": g_tt,
            "g_xx": g_xx,
            "g_yy": g_xx,  # Same as g_xx in Cartesian coordinates
            "g_zz": g_xx,  # Same as g_xx in Cartesian coordinates
            "g_tx": np.zeros_like(x),
            "g_ty": np.zeros_like(x),
            "g_tz": np.zeros_like(x),
            "g_xy": np.zeros_like(x),
            "g_xz": np.zeros_like(x),
            "g_yz": np.zeros_like(x)
        }