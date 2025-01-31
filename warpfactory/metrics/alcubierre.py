"""Alcubierre warp drive metric."""

import numpy as np
from typing import Dict
from .base import Metric

class AlcubierreMetric(Metric):
    """Alcubierre warp drive metric."""
    
    def calculate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 t: float, v_s: float = 2.0, R: float = 1.0,
                 sigma: float = 0.5) -> Dict[str, np.ndarray]:
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
        # Calculate ship position
        x_s = v_s * t
        
        # Calculate r (distance from ship center)
        r = np.sqrt((x - x_s)**2 + y**2 + z**2)
        
        # Shape function
        f = np.exp(-sigma * r**2 / R**2)
        
        # Add cutoff for numerical stability
        f = np.where(r > 5*R, np.zeros_like(f), f)
        
        # Calculate metric components
        v_x = v_s * f
        g_tt = -(1 - v_x**2)
        g_tx = -v_x
        g_xx = np.ones_like(x)
        
        return {
            "g_tt": g_tt,
            "g_tx": g_tx,
            "g_xx": g_xx
        }