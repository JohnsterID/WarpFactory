"""Christoffel symbol calculations.

Two coordinate modes are supported:

* Cartesian (t, x, y, z): the metric is sampled on a 1-D x line (the y/z
  arrays give the line's position). Derivatives along t, y, z are not
  available from such a slice and are treated as zero, which is exact for
  the stationary axial slices used throughout this package.
* Spherical (t, r, theta, phi): static spherically symmetric metrics,
  delegated to SphericalCurvature which differentiates the supplied
  radial profiles numerically.
"""

from typing import Dict, Optional, Union

import numpy as np

from .finite_difference import FiniteDifference
from .spherical import SphericalCurvature
from .tensor_utils import COORDS, components_to_tensor, inverse_tensor


class ChristoffelSymbols:
    """Calculate Christoffel symbols Gamma^a_bc = g^{ad}(d_b g_dc + d_c g_db - d_d g_bc)/2."""

    def __init__(self, order: int = 4):
        self.fd = FiniteDifference(order=order)
        self.spherical = SphericalCurvature(order=order)

    def calculate(
        self,
        metric: Dict[str, np.ndarray],
        x: Union[np.ndarray, Dict[str, np.ndarray]],
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Calculate Christoffel symbols.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components ("g_tt", "g_tx", ... or "g_rr", ...)
        x : Union[np.ndarray, Dict[str, np.ndarray]]
            Either x coordinate array (Cartesian) or coordinate dict
            with "r" (and optionally "theta") for spherical metrics
        y, z : Optional[np.ndarray]
            Unused for the 1-D slice; retained for API compatibility

        Returns
        -------
        Dict[str, np.ndarray]
            Christoffel symbols keyed "a_bc" (upper index first)
        """
        if isinstance(x, dict):
            return self.spherical.christoffel(metric, x)
        return self._calculate_cartesian(metric, x)

    def calculate_array(
        self, metric: Dict[str, np.ndarray], x: np.ndarray
    ) -> np.ndarray:
        """Christoffel symbols as an array of shape (4, 4, 4) + grid."""
        g = components_to_tensor(metric, "g")
        g_inv = inverse_tensor(g)

        # Only x-derivatives (index 1) are available on a 1-D slice.
        dg = np.zeros((4,) + g.shape)
        for mu in range(4):
            for nu in range(4):
                dg[1, mu, nu] = self.fd.derivative1(g[mu, nu], x, axis=0)

        gamma = np.zeros((4, 4, 4) + g.shape[2:])
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    total = np.zeros(g.shape[2:])
                    for d in range(4):
                        total += g_inv[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                    gamma[a, b, c] = 0.5 * total
        return gamma

    def _calculate_cartesian(
        self, metric: Dict[str, np.ndarray], x: np.ndarray
    ) -> Dict[str, np.ndarray]:
        gamma_array = self.calculate_array(metric, np.asarray(x, dtype=float))
        gamma = {}
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    gamma[f"{COORDS[a]}_{COORDS[b]}{COORDS[c]}"] = gamma_array[a, b, c]
        return gamma
