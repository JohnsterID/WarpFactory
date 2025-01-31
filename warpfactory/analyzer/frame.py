import numpy as np
from typing import Dict

class FrameTransformer:
    """Transform metric components between different reference frames."""
    
    def lorentz_boost(self, metric: Dict[str, np.ndarray], v: float, axis: str = 'x') -> Dict[str, np.ndarray]:
        """Apply Lorentz boost transformation.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components in original frame
        v : float
            Boost velocity (in units of c)
        axis : str, optional
            Axis along which to boost ('x', 'y', or 'z')
            
        Returns
        -------
        Dict[str, np.ndarray]
            Transformed metric components
        """
        # Calculate Lorentz factor
        gamma = 1/np.sqrt(1 - v**2)
        
        # Get original components
        g_tt = metric["g_tt"]
        g_tx = metric.get("g_tx", np.zeros_like(g_tt))
        g_xx = metric.get("g_xx", np.ones_like(g_tt))
        
        # Transform components
        g_tt_new = g_tt  # Time-time component is invariant
        g_tx_new = -v * gamma * np.ones_like(g_tt)  # Time-space component
        g_xx_new = gamma**2 * np.ones_like(g_tt)  # Space-space component
        
        return {
            "g_tt": g_tt_new,
            "g_tx": g_tx_new,
            "g_xx": g_xx_new
        }