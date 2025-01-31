"""Gravitational lensing calculations."""

import numpy as np
from typing import Dict, List, Tuple
from scipy.integrate import solve_ivp

class GravitationalLensing:
    """Calculate gravitational lensing effects."""
    
    def _setup_ray_bundle(self, source_pos: np.ndarray,
                         observer_pos: np.ndarray,
                         bundle_radius: float,
                         n_rays: int) -> List[Dict]:
        """Set up initial conditions for light ray bundle.
        
        Parameters
        ----------
        source_pos : np.ndarray
            Source position
        observer_pos : np.ndarray
            Observer position
        bundle_radius : float
            Radius of ray bundle
        n_rays : int
            Number of rays
            
        Returns
        -------
        List[Dict]
            Initial conditions for each ray
        """
        # Direction from source to observer
        direction = observer_pos - source_pos
        direction /= np.linalg.norm(direction)
        
        # Create orthonormal basis
        if abs(direction[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])
        right = np.cross(direction, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        
        # Create ray bundle
        rays = []
        for i in range(n_rays):
            theta = 2 * np.pi * i / n_rays
            r = bundle_radius
            
            # Initial position offset from source
            offset = r * (right * np.cos(theta) + up * np.sin(theta))
            pos = source_pos + offset
            
            rays.append({
                "position": pos,
                "direction": direction,
                "path": [pos],
                "time_delay": 0.0
            })
        
        return rays
    
    def _propagate_ray(self, metric: Dict[str, np.ndarray],
                      ray: Dict, dt: float = 0.1) -> None:
        """Propagate a light ray through spacetime.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        ray : Dict
            Light ray data
        dt : float, optional
            Time step
        """
        def null_geodesic(t, y):
            """Null geodesic equation."""
            pos = y[:3]
            vel = y[3:]
            
            # Interpolate metric at current position
            g_tt = np.interp(pos[0], np.linspace(-5, 5, len(metric["g_tt"])),
                           metric["g_tt"])
            g_tx = np.interp(pos[0], np.linspace(-5, 5, len(metric.get("g_tx", np.zeros_like(metric["g_tt"])))),
                           metric.get("g_tx", np.zeros_like(metric["g_tt"])))
            g_xx = np.interp(pos[0], np.linspace(-5, 5, len(metric.get("g_xx", np.ones_like(metric["g_tt"])))),
                           metric.get("g_xx", np.ones_like(metric["g_tt"])))
            
            # Calculate acceleration
            a = np.zeros_like(vel)
            
            # Simplified null geodesic equation
            a[0] = -g_tx/g_tt * vel[0]**2
            a[1] = -g_xx/g_tt * vel[1]**2
            a[2] = -g_xx/g_tt * vel[2]**2
            
            return np.concatenate([vel, a])
        
        # Initial conditions
        y0 = np.concatenate([ray["position"], ray["direction"]])
        
        # Integrate
        sol = solve_ivp(
            null_geodesic,
            t_span=(0, dt),
            y0=y0,
            method='RK45',
            rtol=1e-3,
            atol=1e-3,
            max_step=dt/10
        )
        
        # Update ray data
        ray["position"] = sol.y[:3, -1]
        ray["direction"] = sol.y[3:, -1]
        ray["direction"] /= np.linalg.norm(ray["direction"])
        ray["path"].append(ray["position"])
        ray["time_delay"] += dt
    
    def trace_light_rays(self, metric: Dict[str, np.ndarray],
                        source_pos: np.ndarray,
                        observer_pos: np.ndarray,
                        bundle_radius: float,
                        n_rays: int) -> List[Dict]:
        """Trace bundle of light rays through spacetime.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        source_pos : np.ndarray
            Source position
        observer_pos : np.ndarray
            Observer position
        bundle_radius : float
            Radius of ray bundle
        n_rays : int
            Number of rays
            
        Returns
        -------
        List[Dict]
            Ray trajectories and properties
        """
        # Set up ray bundle
        rays = self._setup_ray_bundle(source_pos, observer_pos,
                                    bundle_radius, n_rays)
        
        # Propagate rays until they reach observer
        max_steps = 100
        for _ in range(max_steps):
            # Update each ray
            for ray in rays:
                self._propagate_ray(metric, ray)
                
                # Check if ray reached observer
                dist = np.linalg.norm(ray["position"] - observer_pos)
                if dist < bundle_radius:
                    ray["magnification"] = 1.0  # Default magnification
                    ray["shear"] = 0.0  # Default shear
                    ray["convergence"] = 0.0  # Default convergence
            
            # Check if all rays have reached observer
            if all("magnification" in ray for ray in rays):
                break
        
        return rays
    
    def analyze_bundle(self, rays: List[Dict]) -> Dict[str, float]:
        """Analyze optical properties of ray bundle.
        
        Parameters
        ----------
        rays : List[Dict]
            Light ray data
            
        Returns
        -------
        Dict[str, float]
            Optical properties
        """
        # Calculate total magnification
        magnification = np.mean([ray.get("magnification", 1.0)
                               for ray in rays])
        
        # Calculate shear and convergence
        shear = np.mean([ray.get("shear", 0.0)
                        for ray in rays])
        convergence = np.mean([ray.get("convergence", 0.0)
                             for ray in rays])
        
        return {
            "magnification": magnification,
            "shear": shear,
            "convergence": convergence
        }