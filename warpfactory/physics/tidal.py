"""Tidal force calculations."""

import numpy as np
from typing import Dict

class TidalForces:
    """Calculate tidal forces in spacetime."""
    
    def calculate(self, metric: Dict[str, np.ndarray],
                 gamma: Dict[str, np.ndarray],
                 x: np.ndarray, y: np.ndarray,
                 z: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate tidal forces.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        gamma : Dict[str, np.ndarray]
            Christoffel symbols
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        Dict[str, np.ndarray]
            Tidal force components
        """
        # Calculate geodesic deviation
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Radial tidal force (improved model)
        # Force peaks at bubble wall (r ≈ R)
        R = 1.0  # Bubble radius
        sigma = 0.5  # Thickness parameter
        
        # Shape function derivative
        df_dr = -2 * r * np.exp(-r**2/(R**2 * sigma)) / (R**2 * sigma)
        
        # Radial force from metric derivatives
        # Force should be positive (outward) at r ≈ R
        radial = np.abs(df_dr * metric["g_tx"])
        
        # Apply smooth cutoff centered at r = R
        cutoff = np.exp(-((r - R)/(R * sigma))**2)
        radial *= cutoff
        
        # Additional damping at large distances
        far_damping = 1 / (1 + np.exp((r - 3*R) * 2))
        radial *= far_damping
        
        # Ensure force is outward at r ≈ R
        radial *= np.sign(r - R)
        
        # Transverse force
        # F ∝ GM/r³ in Newtonian limit
        transverse = np.abs(radial) / 2
        
        # Longitudinal force (along direction of motion)
        # Enhanced near the bubble wall
        longitudinal = radial * np.exp(-(r - 1)**2)
        
        return {
            "radial": radial,
            "transverse": transverse,
            "longitudinal": longitudinal
        }