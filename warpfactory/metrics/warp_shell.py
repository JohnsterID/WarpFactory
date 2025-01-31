import numpy as np
from .base import Metric

class WarpShellMetric(Metric):
    """Warp Shell metric with finite thickness shell."""
    
    def calculate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float,
                 v_s: float = 1.0, R: float = 1.0, thickness: float = 0.2,
                 sigma: float = 1.0, **kwargs) -> dict:
        """Calculate the Warp Shell metric components.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinates
        t : float
            Time coordinate
        v_s : float
            Ship velocity (in units of c)
        R : float
            Inner radius of warp shell
        thickness : float
            Thickness of the shell
        sigma : float
            Thickness parameter of shell walls
        **kwargs
            Additional parameters
            
        Returns
        -------
        dict
            Metric components
        """
        # Calculate ship position
        x_s = v_s * t
        
        # Calculate r (distance from ship center)
        r = np.sqrt((x - x_s)**2 + y**2 + z**2)
        
        # Double sigmoid shape function for shell with improved asymptotic behavior
        # Use a smoother transition function
        f = np.exp(-sigma * ((r - (R + thickness/2))**2) / (thickness**2))
        
        # Add cutoff for numerical stability
        f[r > R + thickness + 3*thickness] = 0.0
        
        # Calculate lapse function and shift vector
        alpha = 1.0 / np.sqrt(1 + (v_s * f)**2)
        beta_x = v_s * f
        
        # Calculate metric components
        g_tt = -(alpha**2 - beta_x**2)
        g_tx = -beta_x
        g_xx = np.ones_like(x)
        
        return {
            "g_tt": g_tt,
            "g_tx": g_tx,
            "g_xx": g_xx
        }