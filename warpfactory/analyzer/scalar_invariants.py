"""Scalar invariant calculations for spherically symmetric metrics."""

import numpy as np
from typing import Dict
from ..solver import SphericalCurvature


class ScalarInvariants:
    """Curvature scalar invariants computed from the supplied metric."""

    def __init__(self, order: int = 4):
        self.curvature = SphericalCurvature(order=order)

    def kretschmann(self, metric: Dict[str, np.ndarray],
                    coords: Dict[str, np.ndarray]) -> np.ndarray:
        """Kretschmann scalar K = R_munurhosigma R^munurhosigma."""
        return self.curvature.kretschmann(metric, coords)

    def ricci_scalar(self, metric: Dict[str, np.ndarray],
                     coords: Dict[str, np.ndarray]) -> np.ndarray:
        """Ricci scalar R = g^munu R_munu."""
        return self.curvature.ricci_scalar(metric, coords)
