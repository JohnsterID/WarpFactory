"""Singularity detection and analysis."""

import numpy as np
from typing import Dict, List
from scipy.spatial.distance import cdist

class SingularityDetector:
    """Detect and analyze spacetime singularities."""
    
    def calculate_invariants(self, metric: Dict[str, np.ndarray],
                           position: np.ndarray) -> Dict[str, float]:
        """Calculate curvature invariants at a point.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        position : np.ndarray
            Position to calculate at
            
        Returns
        -------
        Dict[str, float]
            Curvature invariants
        """
        # Create interpolation grid
        x = np.linspace(-5, 5, len(metric["g_tt"]))
        
        # Calculate metric derivatives
        dx = 1e-6
        x_plus = position[0] + dx
        x_minus = position[0] - dx
        
        # Interpolate metric components
        g_plus = {
            k: float(np.interp(x_plus, x, v))
            for k, v in metric.items()
        }
        g_minus = {
            k: float(np.interp(x_minus, x, v))
            for k, v in metric.items()
        }
        g = {
            k: float(np.interp(position[0], x, v))
            for k, v in metric.items()
        }
        
        # Calculate derivatives
        dg = {k: (g_plus[k] - g_minus[k])/(2*dx) for k in metric}
        d2g = {k: (g_plus[k] - 2*g[k] + g_minus[k])/dx**2 for k in metric}
        
        # Calculate Ricci scalar (simplified)
        R = d2g["g_tt"] + d2g.get("g_xx", 0.0)
        
        # Calculate Kretschmann scalar (simplified)
        K = (d2g["g_tt"]**2 + d2g.get("g_xx", 0.0)**2 +
             dg.get("g_tx", 0.0)**4)
        
        return {
            "ricci_scalar": float(R),
            "kretschmann": float(K)
        }
    
    def _classify_singularity(self, metric: Dict[str, np.ndarray],
                            position: np.ndarray) -> str:
        """Classify type of singularity.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        position : np.ndarray
            Singularity position
            
        Returns
        -------
        str
            Singularity type
        """
        # Create interpolation grid
        x = np.linspace(-5, 5, len(metric["g_tt"]))
        
        # Get metric components at position
        g_tt = float(np.interp(position[0], x, metric["g_tt"]))
        g_xx = float(np.interp(position[0], x, metric.get("g_xx", np.ones_like(x))))
        
        # Classify based on metric signature
        if g_tt > 0:
            return "spacelike"
        elif g_xx < 0:
            return "timelike"
        else:
            return "null"
    
    def _calculate_strength(self, metric: Dict[str, np.ndarray],
                          position: np.ndarray) -> float:
        """Calculate singularity strength.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        position : np.ndarray
            Singularity position
            
        Returns
        -------
        float
            Singularity strength
        """
        # Calculate invariants
        inv = self.calculate_invariants(metric, position)
        
        # Strength based on curvature invariants
        return float(np.sqrt(abs(inv["ricci_scalar"]) + abs(inv["kretschmann"])))
    
    def find_singularities(self, metric: Dict[str, np.ndarray],
                          x: np.ndarray, y: np.ndarray,
                          z: np.ndarray) -> Dict[str, List]:
        """Find singularities in spacetime.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        Dict[str, List]
            Singularity information
        """
        # Create 2D grid in xz-plane (y=0)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        
        # Find potential singularities where metric components diverge
        singular_points = []
        for i in range(len(x)):
            pos = np.array([x[i], 0.0, 0.0])
            try:
                inv = self.calculate_invariants(metric, pos)
                if (abs(inv["ricci_scalar"]) > 1e6 or
                    abs(inv["kretschmann"]) > 1e6):
                    singular_points.append(pos)
            except (ValueError, ZeroDivisionError):
                # Potential singularity at numerical failure
                singular_points.append(pos)
        
        # Cluster nearby points
        if len(singular_points) > 0:
            singular_points = np.array(singular_points)
            clusters = []
            while len(singular_points) > 0:
                # Start new cluster
                cluster = [singular_points[0]]
                singular_points = singular_points[1:]
                
                # Add nearby points
                i = 0
                while i < len(singular_points):
                    if np.min(cdist([cluster[-1]], [singular_points[i]])) < 0.1:
                        cluster.append(singular_points[i])
                        singular_points = np.delete(singular_points, i, axis=0)
                    else:
                        i += 1
                
                # Add cluster center
                clusters.append(np.mean(cluster, axis=0))
            
            # Analyze each singularity
            locations = np.array(clusters)
            types = [self._classify_singularity(metric, pos)
                    for pos in locations]
            strengths = [self._calculate_strength(metric, pos)
                        for pos in locations]
        else:
            locations = np.array([])
            types = []
            strengths = []
        
        return {
            "locations": locations,
            "types": types,
            "strengths": strengths
        }