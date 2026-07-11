"""Event horizon finder."""

import numpy as np
from typing import Dict

from .interpolation import MetricLine


def _zero_crossings(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linearly interpolated roots of a sampled function."""
    sign_change = np.diff(np.signbit(values))
    idx = np.flatnonzero(sign_change)
    roots = []
    for i in idx:
        x0, x1 = x[i], x[i + 1]
        v0, v1 = values[i], values[i + 1]
        roots.append(x0 - v0 * (x1 - x0) / (v1 - v0))
    return np.array(roots)


class HorizonFinder:
    """Find and analyze horizon-like surfaces on the x line.

    On a stationary 1-D axial slice the detectable surfaces are:

    * "ergosphere": where g_tt changes sign (coordinate-static observers
      cease to be timelike)
    * "outer"/"inner": outermost/innermost roots of the t-x block
      determinant g_tt g_xx - g_tx^2 (the light cone closes in x)

    Surfaces are returned as rings in the xz-plane around each root, the
    revolution of the crossing point about the x axis being the natural
    completion for the axially symmetric metrics used here.
    """

    N_RING = 32

    def _ring(self, x_root: float) -> np.ndarray:
        """Closed circular ring of radius |x_root| in the xz-plane."""
        theta = np.linspace(0.0, 2 * np.pi, self.N_RING + 1)
        radius = abs(x_root)
        return np.column_stack([
            radius * np.cos(theta),
            np.zeros_like(theta),
            radius * np.sin(theta)
        ])

    def find_horizons(self, metric: Dict[str, np.ndarray],
                      x: np.ndarray, y: np.ndarray,
                      z: np.ndarray) -> Dict[str, np.ndarray]:
        """Find horizon surfaces.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates

        Returns
        -------
        Dict[str, np.ndarray]
            "outer", "inner", "ergosphere" surfaces (each (N, 3) points,
            closed, or empty when absent)
        """
        x = np.asarray(x, dtype=float)
        g_tt = np.asarray(metric["g_tt"], dtype=float)
        g_tx = np.asarray(metric.get("g_tx", np.zeros_like(x)), dtype=float)
        g_xx = np.asarray(metric.get("g_xx", np.ones_like(x)), dtype=float)

        ergo_roots = _zero_crossings(x, g_tt)
        det_roots = _zero_crossings(x, g_tt * g_xx - g_tx**2)

        horizons = {
            "outer": np.array([]),
            "inner": np.array([]),
            "ergosphere": np.array([]),
        }
        if len(ergo_roots) > 0:
            horizons["ergosphere"] = self._ring(np.max(np.abs(ergo_roots)))
        if len(det_roots) > 0:
            horizons["outer"] = self._ring(np.max(np.abs(det_roots)))
            if len(det_roots) > 1:
                horizons["inner"] = self._ring(np.min(np.abs(det_roots)))
        return horizons

    def analyze_horizons(self, metric: Dict[str, np.ndarray],
                         horizons: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze properties of the found horizons.

        Returns
        -------
        Dict[str, float]
            "area" (of the outer ring's disk of revolution),
            "surface_gravity" kappa = |d_x g_tt| / (2 sqrt(g_xx |g_tt|'s
            regular part)) evaluated at the outer root, and
            "angular_velocity" omega = -g_tx / g_tt there
        """
        properties = {"area": 0.0, "surface_gravity": 0.0,
                      "angular_velocity": 0.0}
        outer = horizons.get("outer")
        reference = outer if outer is not None and len(outer) > 0 \
            else horizons.get("ergosphere")
        if reference is None or len(reference) == 0:
            return properties

        radius = float(np.max(np.linalg.norm(reference[:, [0, 2]], axis=1)))
        properties["area"] = 4 * np.pi * radius**2

        line = MetricLine(metric)
        dx = 1e-5
        g_p = line.tensor_at(radius + dx)
        g_m = line.tensor_at(radius - dx)
        g_0 = line.tensor_at(radius)
        dgtt_dx = (g_p[0, 0] - g_m[0, 0]) / (2 * dx)
        g_xx_val = g_0[1, 1]
        properties["surface_gravity"] = float(
            abs(dgtt_dx) / (2 * np.sqrt(abs(g_xx_val))))

        if abs(g_0[0, 0]) > 1e-12:
            properties["angular_velocity"] = float(-g_0[0, 1] / g_0[0, 0])
        return properties
