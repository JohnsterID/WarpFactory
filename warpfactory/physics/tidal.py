"""Tidal force calculations via geodesic deviation."""

import numpy as np
from typing import Dict

from ..solver import RiemannTensor, components_to_tensor


class TidalForces:
    """Tidal accelerations from the Riemann tensor.

    The geodesic deviation equation gives the relative (tidal)
    acceleration between neighboring free-falling observers with
    4-velocity u^a and separation s^b:

        D^2 s^a / dtau^2 = -R^a_bcd u^b s^c u^d

    For observers at rest in the coordinate frame (u ~ dt) with unit
    spatial separations this reduces to the components -R^i_tjt.
    """

    def __init__(self, order: int = 4):
        self.riemann = RiemannTensor(order=order)

    def calculate(self, metric: Dict[str, np.ndarray],
                  gamma: Dict[str, np.ndarray],
                  x: np.ndarray, y: np.ndarray,
                  z: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate tidal accelerations along the x axis slice.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        gamma : Dict[str, np.ndarray]
            Christoffel symbols (unused; kept for API compatibility)
        x, y, z : np.ndarray
            Spatial coordinates

        Returns
        -------
        Dict[str, np.ndarray]
            "radial" (x-separation), "transverse" (y/z-separation) and
            "longitudinal" (alias of radial for motion along x) tidal
            accelerations per unit separation
        """
        x = np.asarray(x, dtype=float)
        riemann = self.riemann.calculate_array(metric, x)

        # Normalized coordinate-static observer: u^a = (1/sqrt(-g_tt), 0, 0, 0)
        g = components_to_tensor(metric, "g")
        u_t2 = -1.0 / g[0, 0]

        radial = -riemann[1, 0, 1, 0] * u_t2
        transverse_y = -riemann[2, 0, 2, 0] * u_t2
        transverse_z = -riemann[3, 0, 3, 0] * u_t2

        return {
            "radial": radial,
            "transverse": 0.5 * (transverse_y + transverse_z),
            "longitudinal": radial
        }
