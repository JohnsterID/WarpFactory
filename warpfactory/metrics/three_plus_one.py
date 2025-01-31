"""Three-plus-one (ADM) decomposition."""

import numpy as np
from typing import Dict

class ThreePlusOneDecomposition:
    """ADM decomposition of spacetime metrics."""
    
    def decompose(self, metric: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform 3+1 decomposition.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
            
        Returns
        -------
        Dict[str, np.ndarray]
            ADM variables (lapse, shift, spatial metric)
        """
        # Extract metric components
        g_tt = metric["g_tt"]
        g_rr = metric["g_rr"]
        g_theta_theta = metric["g_theta_theta"]
        g_phi_phi = metric["g_phi_phi"]
        
        # Calculate lapse function (α)
        alpha = np.sqrt(-1/g_tt)
        
        # Calculate shift vector (β^i)
        beta = np.zeros_like(g_tt)  # Zero for spherically symmetric metric
        
        # Calculate spatial metric (γ_ij)
        gamma = {
            "g_rr": g_rr,
            "g_theta_theta": g_theta_theta,
            "g_phi_phi": g_phi_phi
        }
        
        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        }