"""Scalar invariant calculations for spherically symmetric metrics.

Gauge-independent curvature scalars distinguish physical curvature from
coordinate artifacts: a grid point with a divergent metric component but
a finite Kretschmann scalar is a coordinate problem, not a physical
singularity. The Gauss-Bonnet invariant additionally powers the
effective-field-theory validity check: where curvature radii approach a
chosen cutoff length (e.g. the Planck length), higher-derivative
corrections dominate and classical general relativity stops being a
trustworthy description of the metric.
"""

from typing import Dict

import numpy as np

from ..solver import SphericalCurvature
from ..units import Constants


class ScalarInvariants:
    """Curvature scalar invariants computed from the supplied metric."""

    def __init__(self, order: int = 4):
        self.curvature = SphericalCurvature(order=order)

    def kretschmann(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Kretschmann scalar K = R_munurhosigma R^munurhosigma."""
        return self.curvature.kretschmann(metric, coords)

    def ricci_scalar(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Ricci scalar R = g^munu R_munu."""
        return self.curvature.ricci_scalar(metric, coords)

    def ricci_squared(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Ricci tensor square R_munu R^munu."""
        return self.curvature.ricci_squared(metric, coords)

    def gauss_bonnet(
        self, metric: Dict[str, np.ndarray], coords: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Gauss-Bonnet invariant G = R^2 - 4 R_munu R^munu + K.

        Topological in 4-D (its integral is the Euler characteristic),
        but its local value measures the strength of quadratic-curvature
        terms in extended gravity theories. For vacuum solutions it
        reduces to the Kretschmann scalar (48 M^2/r^6 for
        Schwarzschild).
        """
        R = self.ricci_scalar(metric, coords)
        return (
            R**2
            - 4 * self.ricci_squared(metric, coords)
            + self.kretschmann(metric, coords)
        )

    def eft_validity(
        self,
        metric: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        cutoff_length: float = None,
        threshold: float = 1e-2,
    ) -> Dict[str, np.ndarray]:
        """Effective-field-theory validity watchdog.

        Compares the local curvature radius L = K^(-1/4) (from the
        Kretschmann scalar, which never vanishes accidentally for
        curved vacuum) against a UV cutoff length. Dimensionless
        expansion parameter epsilon = (cutoff/L)^2 is the relative size
        of quadratic-curvature corrections in the EFT of gravity;
        where epsilon exceeds the threshold, the classical
        Einstein-Hilbert description of the metric is suspect.

        Parameters
        ----------
        metric, coords : Dict[str, np.ndarray]
            Spherical metric components and coordinates
        cutoff_length : float, optional
            UV cutoff in the same geometric length units as the
            coordinates. Defaults to the Planck length in meters --
            only meaningful if the grid is in meters.
        threshold : float
            Fraction of the cutoff scale above which a point is flagged

        Returns
        -------
        Dict[str, np.ndarray]
            "curvature_radius" : local L = K^(-1/4)
            "expansion_parameter" : epsilon = (cutoff/L)^2
            "valid" : boolean mask, True where the EFT is trustworthy
        """
        if cutoff_length is None:
            const = Constants()
            hbar = const.h / (2 * np.pi)
            cutoff_length = float(np.sqrt(hbar * const.G / const.c**3))
        if cutoff_length <= 0:
            raise ValueError("cutoff_length must be positive")

        K = np.abs(self.kretschmann(metric, coords))
        # Flat points (K = 0) have an infinite curvature radius.
        with np.errstate(divide="ignore"):
            curvature_radius = K**-0.25
        epsilon = cutoff_length**2 * np.sqrt(K)
        return {
            "curvature_radius": curvature_radius,
            "expansion_parameter": epsilon,
            "valid": epsilon < threshold,
        }
