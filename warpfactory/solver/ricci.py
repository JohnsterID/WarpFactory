"""Ricci tensor and scalar calculations.

R_bc = d_a Gamma^a_bc - d_c Gamma^a_ba
       + Gamma^a_ad Gamma^d_bc - Gamma^a_cd Gamma^d_ba

Cartesian mode computes the full contraction from numerically
differentiated Christoffel symbols on a 1-D x slice (t, y, z derivatives
vanish for the stationary slices used here). Spherical mode uses the
closed-form profile expressions in SphericalCurvature.
"""

import numpy as np
from typing import Dict

from .christoffel import ChristoffelSymbols
from .finite_difference import FiniteDifference
from .spherical import SphericalCurvature
from .tensor_utils import COORDS, components_to_tensor, inverse_tensor


def _is_spherical(metric: Dict[str, np.ndarray]) -> bool:
    return "g_rr" in metric


class RicciTensor:
    """Calculate the Ricci tensor from an arbitrary metric."""

    def __init__(self, order: int = 4):
        self.christoffel = ChristoffelSymbols(order=order)
        self.fd = FiniteDifference(order=order)
        self.spherical = SphericalCurvature(order=order)

    def calculate(self, metric: Dict[str, np.ndarray],
                  coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate Ricci tensor components.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components g_munu
        coords : Dict[str, np.ndarray]
            Coordinate arrays; "x" for Cartesian slices, "r"/"theta"
            for spherical metrics

        Returns
        -------
        Dict[str, np.ndarray]
            Ricci tensor components R_munu
        """
        if _is_spherical(metric):
            return self.spherical.ricci_tensor(metric, coords)

        ricci_array = self.calculate_array(metric, np.asarray(coords["x"], dtype=float))
        components = {}
        for mu in range(4):
            for nu in range(mu, 4):
                components[f"R_{COORDS[mu]}{COORDS[nu]}"] = ricci_array[mu, nu]
        return components

    def calculate_array(self, metric: Dict[str, np.ndarray],
                        x: np.ndarray) -> np.ndarray:
        gamma = self.christoffel.calculate_array(metric, x)
        grid_shape = gamma.shape[3:]

        # Only x-derivatives (coordinate index 1) survive on a 1-D slice.
        dgamma = np.zeros((4,) + gamma.shape)
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    dgamma[1, a, b, c] = self.fd.derivative1(gamma[a, b, c], x, axis=0)

        ricci = np.zeros((4, 4) + grid_shape)
        for b in range(4):
            for c in range(4):
                term = np.zeros(grid_shape)
                for a in range(4):
                    term += dgamma[a, a, b, c] - dgamma[c, a, b, a]
                    for d in range(4):
                        term += gamma[a, a, d] * gamma[d, b, c]
                        term -= gamma[a, c, d] * gamma[d, b, a]
                ricci[b, c] = term
        return ricci


class RiemannTensor:
    """Riemann curvature tensor R^a_bcd on Cartesian 1-D slices."""

    def __init__(self, order: int = 4):
        self.christoffel = ChristoffelSymbols(order=order)
        self.fd = FiniteDifference(order=order)

    def calculate_array(self, metric: Dict[str, np.ndarray],
                        x: np.ndarray) -> np.ndarray:
        """Riemann tensor as an array of shape (4, 4, 4, 4) + grid.

        R^a_bcd = d_c Gamma^a_db - d_d Gamma^a_cb
                  + Gamma^a_ce Gamma^e_db - Gamma^a_de Gamma^e_cb
        """
        gamma = self.christoffel.calculate_array(metric, x)
        grid_shape = gamma.shape[3:]

        # Only x-derivatives (coordinate index 1) survive on a 1-D slice.
        dgamma = np.zeros((4,) + gamma.shape)
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    dgamma[1, a, b, c] = self.fd.derivative1(gamma[a, b, c], x, axis=0)

        riemann = np.zeros((4, 4, 4, 4) + grid_shape)
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        term = dgamma[c, a, d, b] - dgamma[d, a, c, b]
                        for e in range(4):
                            term = term + gamma[a, c, e] * gamma[e, d, b]
                            term = term - gamma[a, d, e] * gamma[e, c, b]
                        riemann[a, b, c, d] = term
        return riemann

    def kretschmann(self, metric: Dict[str, np.ndarray],
                    x: np.ndarray) -> np.ndarray:
        """Kretschmann scalar K = R_abcd R^abcd."""
        riemann = self.calculate_array(metric, x)
        g = components_to_tensor(metric, "g")
        g_inv = inverse_tensor(g)
        r_down = np.einsum("ae...,ebcd...->abcd...", g, riemann)
        r_up = np.einsum("bf...,cg...,dh...,afgh...->abcd...", g_inv, g_inv, g_inv, riemann)
        return np.einsum("abcd...,abcd...->...", r_down, r_up)


class RicciScalar:
    """Calculate the Ricci scalar R = g^munu R_munu."""

    def __init__(self, order: int = 4):
        self.ricci_tensor = RicciTensor(order=order)
        self.spherical = SphericalCurvature(order=order)

    def calculate(self, metric: Dict[str, np.ndarray],
                  coords: Dict[str, np.ndarray]) -> np.ndarray:
        if _is_spherical(metric):
            return self.spherical.ricci_scalar(metric, coords)

        x = np.asarray(coords["x"], dtype=float)
        ricci = self.ricci_tensor.calculate_array(metric, x)
        g_inv = inverse_tensor(components_to_tensor(metric, "g"))
        scalar = np.zeros(ricci.shape[2:])
        for mu in range(4):
            for nu in range(4):
                scalar += g_inv[mu, nu] * ricci[mu, nu]
        return scalar
