"""Stress-energy tensor calculations.

The core WarpFactory pipeline (MATLAB getEnergyTensor/met2den): given a
metric, compute the Einstein tensor G_munu = R_munu - R g_munu / 2 via
finite differences and read off the stress-energy tensor from the
Einstein field equations, T_munu = G_munu / (8 pi) in geometric units
(G = c = 1).
"""

from typing import Dict

import numpy as np

from .ricci import RicciTensor
from .tensor_utils import COORDS, components_to_tensor, inverse_tensor


class EnergyTensor:
    """Derive the stress-energy tensor from a metric, or build simple sources."""

    def __init__(self, order: int = 4):
        self.ricci = RicciTensor(order=order)

    def calculate_from_metric(
        self, metric: Dict[str, np.ndarray], x: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Stress-energy tensor T_munu of a metric via the field equations.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Covariant metric components on a 1-D x slice
        x : np.ndarray
            x coordinate array

        Returns
        -------
        Dict[str, np.ndarray]
            Covariant stress-energy components "T_tt", "T_tx", ...
        """
        x = np.asarray(x, dtype=float)
        g = components_to_tensor(metric, "g")
        g_inv = inverse_tensor(g)
        ricci = self.ricci.calculate_array(metric, x)

        ricci_scalar = np.zeros(ricci.shape[2:])
        for mu in range(4):
            for nu in range(4):
                ricci_scalar += g_inv[mu, nu] * ricci[mu, nu]

        einstein = ricci - 0.5 * ricci_scalar * g
        stress_energy = einstein / (8 * np.pi)

        components = {}
        for mu in range(4):
            for nu in range(mu, 4):
                components[f"T_{COORDS[mu]}{COORDS[nu]}"] = stress_energy[mu, nu]
        return components

    def calculate_perfect_fluid(
        self, rho: np.ndarray, p: np.ndarray, x: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Perfect fluid stress-energy tensor: T_tt = rho, T_ii = p."""
        return {
            "T_tt": rho,
            "T_xx": p,
            "T_yy": p,
            "T_zz": p,
            "T_tx": np.zeros_like(x),
            "T_ty": np.zeros_like(x),
            "T_tz": np.zeros_like(x),
        }
