"""Stress-energy conservation analysis."""

import numpy as np
from typing import Dict

from ..solver import (
    ChristoffelSymbols,
    EnergyTensor,
    FiniteDifference,
    components_to_tensor,
    inverse_tensor,
)

COORD_NAMES = ("t", "x", "y", "z")


class StressEnergyConservation:
    """Check covariant conservation of the EFE-derived stress-energy tensor.

    nabla_mu T^{mu nu} = d_mu T^{mu nu} + Gamma^mu_{mu a} T^{a nu}
                         + Gamma^nu_{mu a} T^{mu a}

    The contracted Bianchi identity guarantees this vanishes for
    stress-energy obtained from the Einstein tensor, so any residual is
    discretization error and converges to zero with grid resolution.
    """

    def __init__(self, order: int = 4):
        self.christoffel = ChristoffelSymbols(order=order)
        self.energy = EnergyTensor(order=order)
        self.fd = FiniteDifference(order=order)

    def calculate_divergence(self, metric: Dict[str, np.ndarray],
                             gamma: Dict[str, np.ndarray],
                             x: np.ndarray, y: np.ndarray,
                             z: np.ndarray) -> Dict[str, np.ndarray]:
        """Covariant divergence components of T^{mu nu}.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        gamma : Dict[str, np.ndarray]
            Christoffel symbols (unused; recomputed internally as an
            array for the contraction)
        x, y, z : np.ndarray
            Spatial coordinates (1-D x slice)

        Returns
        -------
        Dict[str, np.ndarray]
            Divergence 4-vector components keyed "t", "x", "y", "z"
        """
        x = np.asarray(x, dtype=float)

        T_cov = components_to_tensor(
            self.energy.calculate_from_metric(metric, x), "T")
        g_inv = inverse_tensor(components_to_tensor(metric, "g"))
        T_con = np.einsum("ma...,nb...,ab...->mn...", g_inv, g_inv, T_cov)

        gamma_array = self.christoffel.calculate_array(metric, x)

        divergence = {}
        for nu in range(4):
            # Only x-derivatives (mu=1) are nonzero on the 1-D slice
            div_nu = self.fd.derivative1(T_con[1, nu], x, axis=0)
            for mu in range(4):
                for a in range(4):
                    div_nu = div_nu + gamma_array[mu, mu, a] * T_con[a, nu]
                    div_nu = div_nu + gamma_array[nu, mu, a] * T_con[mu, a]
            divergence[COORD_NAMES[nu]] = div_nu
        return divergence

    def check_conservation_laws(self, metric: Dict[str, np.ndarray],
                                gamma: Dict[str, np.ndarray],
                                x: np.ndarray, y: np.ndarray,
                                z: np.ndarray,
                                atol: float = 1e-3) -> Dict[str, bool]:
        """Check energy and momentum conservation on interior points.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        gamma : Dict[str, np.ndarray]
            Christoffel symbols (see calculate_divergence)
        x, y, z : np.ndarray
            Spatial coordinates
        atol : float
            Tolerance on the discretization residual

        Returns
        -------
        Dict[str, bool]
            {"energy": ..., "momentum": ...} conservation status
        """
        div = self.calculate_divergence(metric, gamma, x, y, z)
        # One-sided boundary stencils are lower order; judge interior only
        interior = slice(4, -4)
        return {
            "energy": bool(np.allclose(div["t"][interior], 0.0, atol=atol)),
            "momentum": bool(np.allclose(div["x"][interior], 0.0, atol=atol))
        }
