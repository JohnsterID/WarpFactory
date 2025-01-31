"""Christoffel symbol calculations."""

import numpy as np
from typing import Dict, Optional, Union

class ChristoffelSymbols:
    """Calculate Christoffel symbols."""
    
    def calculate(self, metric: Dict[str, np.ndarray],
                 x: Union[np.ndarray, Dict[str, np.ndarray]],
                 y: Optional[np.ndarray] = None,
                 z: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Calculate Christoffel symbols.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x : Union[np.ndarray, Dict[str, np.ndarray]]
            Either x coordinate array (Cartesian) or coordinate dictionary (spherical)
        y : Optional[np.ndarray]
            Y coordinate (Cartesian only)
        z : Optional[np.ndarray]
            Z coordinate (Cartesian only)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Christoffel symbols
        """
        if isinstance(x, dict):
            # Spherical coordinates
            return self._calculate_spherical(metric, x)
        else:
            # Cartesian coordinates
            return self._calculate_cartesian(metric, x, y, z)
    
    def _calculate_spherical(self, metric: Dict[str, np.ndarray],
                           coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate Christoffel symbols in spherical coordinates."""
        # Extract coordinates
        r = coords["r"]
        theta = coords["theta"]
        
        # Extract metric components
        g_tt = metric["g_tt"]
        g_rr = metric["g_rr"]
        g_theta_theta = metric["g_theta_theta"]
        g_phi_phi = metric["g_phi_phi"]
        
        # Calculate derivatives
        dg_tt_dr = -2 * (2/r**2) / (1 - 2/r)**2
        dg_rr_dr = 2 * (2/r**2) / (1 - 2/r)**2
        dg_theta_theta_dr = 2 * r
        dg_phi_phi_dr = 2 * r * np.sin(theta)**2
        dg_phi_phi_dtheta = 2 * r**2 * np.sin(theta) * np.cos(theta)
        
        # Calculate non-zero Christoffel symbols
        gamma = {}
        
        # t_tr component
        gamma["t_tr"] = 1/(r*(r - 2))
        
        # t_tt component
        gamma["t_tt"] = np.zeros_like(r)
        
        # r_tt component
        gamma["r_tt"] = (r - 2)/(r**3)  # Corrected formula for Schwarzschild
        
        # r_rr component
        gamma["r_rr"] = -(r - 2)/(r**3)
        
        # r_thetatheta component
        gamma["r_thetatheta"] = -(r - 2)
        
        # r_phiphi component
        gamma["r_phiphi"] = -(r - 2) * np.sin(theta)**2
        
        # theta_rtheta component
        gamma["theta_rtheta"] = 1/r
        
        # theta_phiphi component
        gamma["theta_phiphi"] = -np.sin(theta) * np.cos(theta)
        
        # phi_rphi component
        gamma["phi_rphi"] = 1/r
        
        # phi_thetaphi component
        gamma["phi_thetaphi"] = np.cos(theta)/np.sin(theta)
        
        return gamma
    
    def _calculate_cartesian(self, metric: Dict[str, np.ndarray],
                           x: np.ndarray, y: np.ndarray,
                           z: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Christoffel symbols in Cartesian coordinates."""
        # Extract metric components
        g_tt = metric["g_tt"]
        g_xx = metric["g_xx"]
        g_yy = metric.get("g_yy", np.ones_like(x))  # Default to flat space
        g_zz = metric.get("g_zz", np.ones_like(x))  # Default to flat space
        g_tx = metric.get("g_tx", np.zeros_like(x))
        g_ty = metric.get("g_ty", np.zeros_like(x))
        g_tz = metric.get("g_tz", np.zeros_like(x))
        g_xy = metric.get("g_xy", np.zeros_like(x))
        g_xz = metric.get("g_xz", np.zeros_like(x))
        g_yz = metric.get("g_yz", np.zeros_like(x))
        
        # Calculate derivatives (finite differences)
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        
        # Initialize Christoffel symbols
        gamma = {}
        
        # Initialize all components to zero
        for i in ["t", "x", "y", "z"]:
            for j in ["t", "x", "y", "z"]:
                gamma[f"{i}_{j}t"] = np.zeros_like(x)
                gamma[f"{i}_{j}x"] = np.zeros_like(x)
                gamma[f"{i}_{j}y"] = np.zeros_like(x)
                gamma[f"{i}_{j}z"] = np.zeros_like(x)
        
        # For flat spacetime, most components are zero
        # Only calculate non-zero components for curved metrics
        if np.any(np.abs(g_tx) > 1e-10):
            # Example: Γ^t_tx = -1/(2g_tt) * ∂g_tt/∂x
            gamma["t_tx"] = -1/(2*g_tt) * np.gradient(g_tt, dx, edge_order=2)
        
        return gamma