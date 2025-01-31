"""Geodesic equation solver."""

import numpy as np
from typing import Dict, Tuple, List
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

class GeodesicSolver:
    """Solve geodesic equations in curved spacetime."""
    
    def interpolate_metric(self, metric: Dict[str, np.ndarray],
                         position: np.ndarray) -> np.ndarray:
        """Interpolate metric components at a position.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        position : np.ndarray
            Position to interpolate at
            
        Returns
        -------
        np.ndarray
            4x4 metric tensor at position
        """
        # Create interpolation grid
        x = np.linspace(-5, 5, len(metric["g_tt"]))
        
        # Interpolate metric components
        g_tt = float(np.interp(position[0], x, metric["g_tt"]))
        g_tx = float(np.interp(position[0], x, metric.get("g_tx", np.zeros_like(x))))
        g_xx = float(np.interp(position[0], x, metric.get("g_xx", np.ones_like(x))))
        
        # Create 4x4 metric tensor
        g = np.zeros((4, 4))
        g[0, 0] = g_tt
        g[0, 1] = g[1, 0] = g_tx
        g[1, 1] = g[2, 2] = g[3, 3] = g_xx
        
        return g
    
    def calculate_christoffel(self, metric: Dict[str, np.ndarray],
                            position: np.ndarray) -> np.ndarray:
        """Calculate Christoffel symbols at a position.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        position : np.ndarray
            Position to calculate at
            
        Returns
        -------
        np.ndarray
            Christoffel symbols Γᵅ_βγ
        """
        # Calculate metric derivatives (finite difference)
        dx = 1e-6
        pos_plus = position.copy()
        pos_plus[0] += dx
        pos_minus = position.copy()
        pos_minus[0] -= dx
        
        g_plus = self.interpolate_metric(metric, pos_plus)
        g_minus = self.interpolate_metric(metric, pos_minus)
        g = self.interpolate_metric(metric, position)
        
        # Calculate derivatives
        dg_dx = (g_plus - g_minus) / (2*dx)
        
        # Calculate inverse metric
        g_inv = np.linalg.inv(g)
        
        # Calculate Christoffel symbols
        gamma = np.zeros((4, 4, 4))
        for alpha in range(4):
            for beta in range(4):
                for gamma_idx in range(4):
                    # Sum over mu using numpy operations
                    terms = np.zeros(4)
                    for mu in range(4):
                        terms[mu] = g_inv[alpha, mu] * (
                            dg_dx[beta, gamma_idx] +
                            dg_dx[gamma_idx, beta] -
                            dg_dx[beta, gamma_idx]
                        )
                    gamma[alpha, beta, gamma_idx] = 0.5 * np.sum(terms)
        
        return gamma
    
    def geodesic_equation(self, t: float, y: np.ndarray,
                         metric: Dict[str, np.ndarray]) -> np.ndarray:
        """Geodesic equation for numerical integration.
        
        Parameters
        ----------
        t : float
            Time parameter
        y : np.ndarray
            State vector [x, v] (position and velocity)
        metric : Dict[str, np.ndarray]
            Metric components
            
        Returns
        -------
        np.ndarray
            Time derivatives [dx/dt, dv/dt]
        """
        x = y[:3]  # Space position
        v = y[3:]  # Space velocity
        
        # Calculate Christoffel symbols
        gamma = self.calculate_christoffel(metric, x)
        
        # Calculate acceleration from geodesic equation
        a = np.zeros_like(v)
        
        # Normalize velocity to ensure timelike condition
        g = self.interpolate_metric(metric, x)
        
        # Convert to numpy arrays
        v = np.array(v, dtype=float)
        
        # Calculate components
        g_tt = g[0, 0]  # Time-time component
        g_tx = g[0, 1]  # Time-space component
        g_xx = g[1, 1]  # Space-space component
        
        # Calculate current norm
        v_x = v[0]  # x-component of velocity
        v_perp2 = np.sum(v[1:]**2)  # Square of perpendicular components
        
        # Calculate normalization factor
        # From: g_tt + 2g_tx*v_x + g_xx*(v_x^2 + v_perp^2) = -1
        # This is a quadratic equation in the overall scale factor s
        a = g_xx * (v_x**2 + v_perp2)  # s^2 term
        b = 2 * g_tx * v_x  # s term
        c = g_tt + 1  # constant term
        
        # Solve quadratic equation: as^2 + bs + c = 0
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            # Take the solution that minimizes the velocity
            s1 = (-b + np.sqrt(discriminant)) / (2*a)
            s2 = (-b - np.sqrt(discriminant)) / (2*a)
            s = s1 if abs(s1) < abs(s2) else s2
            
            # Apply scaling
            v *= s
        else:
            # If no real solution exists, reduce velocity until timelike
            v_mag = np.sqrt(np.sum(v**2))
            if v_mag > 0:
                v *= 0.1 / v_mag  # Arbitrary small value
        
        # Now calculate acceleration
        # Calculate acceleration using Christoffel symbols
        a = np.zeros(3)  # Only spatial components
        
        # Calculate acceleration from geodesic equation
        # d²xᵅ/dτ² = -Γᵅ_βγ (dxᵝ/dτ)(dxᵍ/dτ)
        for i in range(3):  # Spatial components only
            for j in range(3):
                for k in range(3):
                    a[i] -= gamma[i, j, k] * v[j] * v[k]
        
        # Add time-space coupling terms
        # These come from the time component of the four-velocity
        for i in range(3):
            for j in range(3):
                a[i] -= 2 * gamma[i, 0, j] * v[j]  # Time-space coupling
            a[i] -= gamma[i, 0, 0]  # Pure time term
        
        # Limit acceleration magnitude for numerical stability
        a_mag = np.sqrt(np.sum(a**2))
        if a_mag > 10.0:  # Arbitrary limit
            a *= 10.0 / a_mag
        
        # Return velocity and acceleration
        return np.concatenate([v, a])
    
    def solve(self, metric: Dict[str, np.ndarray], t0: float,
             x0: np.ndarray, v0: np.ndarray, t_max: float,
             dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve geodesic equations.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        t0 : float
            Initial time
        x0 : np.ndarray
            Initial position
        v0 : np.ndarray
            Initial velocity
        t_max : float
            Maximum time
        dt : float
            Time step
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Times, positions, velocities
        """
        # Initial state vector
        y0 = np.concatenate([x0, v0])  # [position, velocity]
        
        # Time points for integration
        t_eval = np.arange(t0, t_max, dt)
        
        # Solve ODE with more relaxed tolerances
        def normalized_geodesic(t, y):
            """Wrapper to ensure normalization at each step."""
            # Get current state
            pos = y[:3]
            vel = y[3:]
            
            # Get metric at current position
            g = self.interpolate_metric(metric, pos)
            
            # Calculate current norm
            u = np.concatenate([[1.0], vel])
            norm2 = np.einsum('i,ij,j->', u, g, u)
            
            # Normalize if not sufficiently timelike
            if norm2 > -0.5:
                # Calculate components
                g_tt = g[0, 0]
                g_tx = g[0, 1]
                g_xx = g[1, 1]
                
                # Calculate required scaling
                v_x = vel[0]
                v_perp2 = np.sum(vel[1:]**2)
                
                # Solve quadratic equation
                a = g_xx * (v_x**2 + v_perp2)
                b = 2 * g_tx * v_x
                c = g_tt + 1
                
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    # Take the solution that minimizes the velocity
                    s1 = (-b + np.sqrt(discriminant)) / (2*a)
                    s2 = (-b - np.sqrt(discriminant)) / (2*a)
                    s = s1 if abs(s1) < abs(s2) else s2
                    
                    # Apply scaling
                    vel = vel * s
                else:
                    # If no real solution exists, reduce velocity until timelike
                    v_mag = np.sqrt(np.sum(vel**2))
                    if v_mag > 0:
                        vel *= 0.1 / v_mag  # Arbitrary small value
            
            # Calculate acceleration
            dydt = self.geodesic_equation(t, np.concatenate([pos, vel]), metric)
            
            # Ensure timelike condition is maintained
            u_new = np.concatenate([[1.0], dydt[:3]])  # New four-velocity
            norm2_new = np.einsum('i,ij,j->', u_new, g, u_new)
            if norm2_new > -0.5:
                # Scale down velocity if needed
                dydt[:3] *= 0.5  # Reduce velocity
                dydt[3:] *= 0.5  # Reduce acceleration
            
            return dydt
        
        # Solve with normalized equations
        sol = solve_ivp(
            fun=normalized_geodesic,
            t_span=(t0, t_max),
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-3,
            atol=1e-3,
            max_step=dt,
            first_step=dt/10  # Start with smaller steps
        )
        
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        # Extract results
        positions = sol.y[:3].T  # First 3 components are positions
        velocities = sol.y[3:].T  # Last 3 components are velocities
        times = sol.t
        
        return times, positions, velocities
    
    def calculate_energy(self, metric: Dict[str, np.ndarray],
                        position: np.ndarray,
                        velocity: np.ndarray) -> float:
        """Calculate conserved energy along geodesic.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        position : np.ndarray
            Position
        velocity : np.ndarray
            Velocity
            
        Returns
        -------
        float
            Conserved energy
        """
        # Get metric at position
        g = self.interpolate_metric(metric, position)
        
        # Four-velocity
        u = np.concatenate([[1.0], velocity])
        
        # Energy = -g_tt u^t
        return -g[0, 0] * u[0]