"""Event horizon finder."""

import numpy as np
from typing import Dict, List
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError

class HorizonFinder:
    """Find and analyze event horizons."""
    
    def _find_horizon_surface(self, metric: Dict[str, np.ndarray],
                            x: np.ndarray, y: np.ndarray,
                            z: np.ndarray, condition: str) -> np.ndarray:
        """Find a horizon surface using a given condition.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
        condition : str
            Type of horizon to find
            
        Returns
        -------
        np.ndarray
            Surface points
        """
        # Create 2D grid in xz-plane (y=0)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        
        # Calculate horizon condition
        if condition == "outer":
            # Outer horizon: g_tt = 0
            values = np.interp(X.flatten(), x, metric["g_tt"]).reshape(X.shape)
        elif condition == "inner":
            # Inner horizon: det(g) = 0
            g_tt = np.interp(X.flatten(), x, metric["g_tt"]).reshape(X.shape)
            g_tx = np.interp(X.flatten(), x, metric.get("g_tx", np.zeros_like(x))).reshape(X.shape)
            g_xx = np.interp(X.flatten(), x, metric.get("g_xx", np.ones_like(x))).reshape(X.shape)
            values = g_tt * g_xx - g_tx**2
        else:  # ergosphere
            # Ergosphere: g_tt > 0
            values = np.interp(X.flatten(), x, metric["g_tt"]).reshape(X.shape)
        
        # Find zero crossings
        if condition == "ergosphere":
            mask = values > 0
        else:
            mask = np.abs(values) < 1e-6
        
        # Extract surface points
        points = np.column_stack([
            X[mask], Y[mask], Z[mask]
        ])
        
        # Order points to form a closed surface
        if len(points) > 3:
            try:
                # Project to xz-plane and find convex hull
                xy_points = points[:, [0, 2]]
                hull = ConvexHull(xy_points)
                ordered_points = points[hull.vertices]
                
                # Add first point at end to close the surface
                ordered_points = np.vstack([ordered_points, ordered_points[0]])
                points = ordered_points
            except (ValueError, QhullError):
                # If convex hull fails, order points by angle
                center = np.mean(points, axis=0)
                angles = np.arctan2(points[:, 2] - center[2],
                                  points[:, 0] - center[0])
                order = np.argsort(angles)
                points = points[order]
                points = np.vstack([points, points[0]])
        
        return points
    
    def find_horizons(self, metric: Dict[str, np.ndarray],
                     x: np.ndarray, y: np.ndarray,
                     z: np.ndarray) -> Dict[str, np.ndarray]:
        """Find all horizons in spacetime.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
            
        Returns
        -------
        Dict[str, np.ndarray]
            Horizon surfaces
        """
        horizons = {}
        
        # Find each type of horizon
        for horizon_type in ["outer", "inner", "ergosphere"]:
            surface = self._find_horizon_surface(metric, x, y, z, horizon_type)
            horizons[horizon_type] = surface
        
        return horizons
    
    def analyze_horizons(self, metric: Dict[str, np.ndarray],
                        horizons: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze horizon properties.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        horizons : Dict[str, np.ndarray]
            Horizon surfaces
            
        Returns
        -------
        Dict[str, float]
            Horizon properties
        """
        properties = {}
        
        # Calculate area (if outer horizon exists)
        if len(horizons["outer"]) > 0:
            points = horizons["outer"]
            try:
                # Project to xz-plane for area calculation
                hull = ConvexHull(points[:, [0, 2]])
                properties["area"] = hull.area
            except (ValueError, QhullError):
                properties["area"] = 0.0
            
            # Calculate surface gravity
            # κ = 1/(2√-g) ∂_r√(-g_tt)
            r = np.sqrt(points[0, 0]**2 + points[0, 2]**2)
            dr = 1e-6
            x = np.linspace(-5, 5, len(metric["g_tt"]))
            g_tt_plus = np.interp(r + dr, np.abs(x), metric["g_tt"])
            g_tt_minus = np.interp(r - dr, np.abs(x), metric["g_tt"])
            kappa = abs((np.sqrt(-g_tt_plus) - np.sqrt(-g_tt_minus))/(2*dr))
            properties["surface_gravity"] = float(kappa)
            
            # Calculate angular velocity (if rotating)
            if "g_tx" in metric:
                omega = -metric["g_tx"]/metric["g_tt"]
                idx = np.argmin(np.abs(x - points[0, 0]))
                properties["angular_velocity"] = float(omega[idx])
            else:
                properties["angular_velocity"] = 0.0
        else:
            properties["area"] = 0.0
            properties["surface_gravity"] = 0.0
            properties["angular_velocity"] = 0.0
        
        return properties