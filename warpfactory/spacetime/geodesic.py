"""Geodesic equation solver."""

from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .interpolation import MetricLine


class GeodesicSolver:
    """Solve timelike geodesic equations in coordinate-time form.

    Trajectories are parametrized by coordinate time t; the geodesic
    equation is reduced accordingly (see MetricLine.coordinate_acceleration),
    which is exact and avoids ad hoc renormalization during integration.
    """

    def interpolate_metric(
        self, metric: Dict[str, np.ndarray], position: np.ndarray
    ) -> np.ndarray:
        """4x4 metric tensor interpolated at a position."""
        return MetricLine(metric).tensor_at(position[0])

    def calculate_christoffel(
        self, metric: Dict[str, np.ndarray], position: np.ndarray
    ) -> np.ndarray:
        """Christoffel symbols Gamma^a_bc at a position."""
        return MetricLine(metric).christoffel_at(position[0])

    def four_velocity(
        self, metric: Dict[str, np.ndarray], position: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """Normalized timelike 4-velocity for coordinate velocity v.

        u^a = (dt/dtau) (1, v) with dt/dtau fixed by g_ab u^a u^b = -1.

        Raises
        ------
        ValueError
            If the coordinate velocity is not timelike at this position
        """
        g = self.interpolate_metric(metric, position)
        w = np.concatenate([[1.0], velocity])
        norm2 = w @ g @ w
        if norm2 >= 0:
            raise ValueError(
                f"Coordinate velocity {velocity} is not timelike at "
                f"x={position[0]:.3f} (w.g.w = {norm2:.3e} >= 0)"
            )
        return w / np.sqrt(-norm2)

    def solve(
        self,
        metric: Dict[str, np.ndarray],
        t0: float,
        x0: np.ndarray,
        v0: np.ndarray,
        t_max: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate a timelike geodesic.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components sampled on the standard x grid
        t0 : float
            Initial coordinate time
        x0 : np.ndarray
            Initial position (3-vector)
        v0 : np.ndarray
            Initial coordinate velocity dx/dt (3-vector); must be
            timelike at x0
        t_max : float
            Final coordinate time
        dt : float
            Output sampling step

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Times, positions (N, 3), coordinate velocities (N, 3)
        """
        line = MetricLine(metric)

        # Fail fast on unphysical initial data instead of silently rescaling
        self.four_velocity(metric, x0, v0)

        def rhs(t, y):
            pos, vel = y[:3], y[3:]
            acc = line.coordinate_acceleration(pos, vel)
            return np.concatenate([vel, acc])

        sol = solve_ivp(
            fun=rhs,
            t_span=(t0, t_max),
            y0=np.concatenate([x0, v0]),
            t_eval=np.arange(t0, t_max, dt),
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.t, sol.y[:3].T, sol.y[3:].T

    def calculate_energy(
        self, metric: Dict[str, np.ndarray], position: np.ndarray, velocity: np.ndarray
    ) -> float:
        """Conserved energy E = -g_ab (dt)^a u^b = -(g_tt u^t + g_tx u^x + ...).

        Exactly conserved along geodesics of stationary metrics
        (Killing vector d/dt).
        """
        g = self.interpolate_metric(metric, position)
        u = self.four_velocity(metric, position, velocity)
        return float(-(g[0] @ u))
