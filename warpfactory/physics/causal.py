"""Causal structure analysis."""

import numpy as np
from typing import Dict, List

class CausalStructure:
    """Analyze causal structure of spacetime."""
    
    def find_horizons(self, metric: Dict[str, np.ndarray],
                     x: np.ndarray, y: np.ndarray,
                     z: np.ndarray) -> Dict[str, List[float]]:
        """Find horizon locations.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        Dict[str, List[float]]
            Horizon locations
        """
        # Calculate horizon locations where g_tt = 0
        g_tt = metric["g_tt"]
        
        # Find zero crossings
        zeros = np.where(np.diff(np.signbit(g_tt)))[0]
        
        # Separate inner and outer horizons
        if len(zeros) >= 2:
            inner = [x[zeros[0]]]
            outer = [x[zeros[-1]]]
        else:
            inner = []
            outer = []
        
        return {
            "inner": inner,
            "outer": outer
        }
    
    def classify_regions(self, metric: Dict[str, np.ndarray],
                       x: np.ndarray, y: np.ndarray,
                       z: np.ndarray) -> np.ndarray:
        """Classify spacetime regions.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        np.ndarray
            Region classification array
        """
        g_tt = metric["g_tt"]
        g_tx = metric["g_tx"]
        
        # Initialize regions
        regions = np.full_like(g_tt, "normal", dtype=object)
        
        # Ergosphere: g_tt > 0
        regions[g_tt > 0] = "ergo"
        
        # Trapped regions: |g_tx| > 1
        regions[np.abs(g_tx) > 1] = "trapped"
        
        return regions
    
    def light_cone_tilt(self, metric: Dict[str, np.ndarray],
                       x: np.ndarray, y: np.ndarray,
                       z: np.ndarray) -> np.ndarray:
        """Calculate light cone tilt angles.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        np.ndarray
            Tilt angles in radians
        """
        g_tt = metric["g_tt"]
        g_tx = metric["g_tx"]
        
        # Calculate tilt angle
        # θ = arctan(g_tx/√(-g_tt)) when g_tt < 0
        tilt = np.zeros_like(g_tt)
        mask = g_tt < 0
        tilt[mask] = np.arctan2(g_tx[mask], np.sqrt(-g_tt[mask]))
        
        return tilt
    
    def find_causality_violations(self, metric: Dict[str, np.ndarray],
                                x: np.ndarray, y: np.ndarray,
                                z: np.ndarray) -> np.ndarray:
        """Find regions with causality violations.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        np.ndarray
            Boolean mask of violation regions
        """
        g_tt = metric["g_tt"]
        g_tx = metric["g_tx"]
        g_xx = metric["g_xx"]
        
        # Calculate determinant
        det = g_tt * g_xx - g_tx**2
        
        # Violations occur when det ≥ 0 or g_xx ≤ 0
        violations = (det >= 0) | (g_xx <= 0)
        
        return violations