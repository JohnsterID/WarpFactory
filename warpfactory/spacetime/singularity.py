"""Singularity detection and analysis."""

from typing import Dict, List

import numpy as np

from ..solver import RicciScalar, RiemannTensor
from .interpolation import X_MAX, X_MIN


class SingularityDetector:
    """Detect and analyze curvature singularities on the x line.

    Real curvature invariants (Ricci scalar and Kretschmann scalar,
    computed from the metric via the solver machinery) are evaluated on
    the sampling grid; points where they blow up are flagged and
    clustered into singularity candidates.
    """

    def __init__(self, order: int = 4, threshold: float = 1e6):
        self.ricci_scalar = RicciScalar(order=order)
        self.riemann = RiemannTensor(order=order)
        self.threshold = threshold

    def _grid(self, metric: Dict[str, np.ndarray]) -> np.ndarray:
        return np.linspace(X_MIN, X_MAX, len(np.asarray(metric["g_tt"])))

    def calculate_invariants(
        self, metric: Dict[str, np.ndarray], position: np.ndarray
    ) -> Dict[str, float]:
        """Curvature invariants interpolated at a position.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components sampled on the standard grid
        position : np.ndarray
            Position (only the x component is used on the 1-D slice)

        Returns
        -------
        Dict[str, float]
            {"ricci_scalar": ..., "kretschmann": ...}
        """
        x = self._grid(metric)
        R = self.ricci_scalar.calculate(metric, {"x": x})
        K = self.riemann.kretschmann(metric, x)
        return {
            "ricci_scalar": float(np.interp(position[0], x, R)),
            "kretschmann": float(np.interp(position[0], x, K)),
        }

    def _classify_singularity(
        self, metric: Dict[str, np.ndarray], position: np.ndarray
    ) -> str:
        """Classify by the local metric signature."""
        x = self._grid(metric)
        g_tt = float(np.interp(position[0], x, metric["g_tt"]))
        g_xx = float(np.interp(position[0], x, metric.get("g_xx", np.ones_like(x))))
        if g_tt > 0:
            return "spacelike"
        if g_xx < 0:
            return "timelike"
        return "null"

    def _calculate_strength(
        self, metric: Dict[str, np.ndarray], position: np.ndarray
    ) -> float:
        inv = self.calculate_invariants(metric, position)
        return float(np.sqrt(abs(inv["ricci_scalar"]) + abs(inv["kretschmann"])))

    def find_singularities(
        self, metric: Dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Dict[str, List]:
        """Locate curvature singularities along the x line.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates

        Returns
        -------
        Dict[str, List]
            "locations" (N, 3), "types", "strengths"
        """
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            R = self.ricci_scalar.calculate(metric, {"x": x})
            K = self.riemann.kretschmann(metric, x)

        blown_up = (
            (~np.isfinite(R))
            | (~np.isfinite(K))
            | (np.abs(R) > self.threshold)
            | (np.abs(K) > self.threshold)
        )
        indices = np.flatnonzero(blown_up)

        # Cluster contiguous flagged points into single candidates
        clusters: List[np.ndarray] = []
        if len(indices) > 0:
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            group = [indices[0]]
            for i in indices[1:]:
                if (x[i] - x[group[-1]]) < 3 * dx:
                    group.append(i)
                else:
                    clusters.append(np.array([x[group].mean(), 0.0, 0.0]))
                    group = [i]
            clusters.append(np.array([x[group].mean(), 0.0, 0.0]))

        locations = np.array(clusters) if clusters else np.array([])
        types = [self._classify_singularity(metric, pos) for pos in locations]
        strengths = [self._calculate_strength(metric, pos) for pos in locations]

        return {"locations": locations, "types": types, "strengths": strengths}
