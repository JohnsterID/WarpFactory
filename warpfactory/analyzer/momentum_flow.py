"""Momentum flow analysis for warp drive metrics."""

from typing import Dict

import numpy as np

from ..solver import (
    ChristoffelSymbols,
    EnergyTensor,
    FiniteDifference,
    components_to_tensor,
    inverse_tensor,
)


class MomentumFlow:
    """Calculate momentum flow lines in spacetime."""

    def __init__(self, order: int = 4):
        self.christoffel = ChristoffelSymbols(order=order)
        self.energy = EnergyTensor(order=order)
        self.fd = FiniteDifference(order=order)

    def calculate_flow_lines(
        self,
        metric: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float,
    ) -> Dict[str, np.ndarray]:
        """Calculate Eulerian momentum flow lines.

        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
        t : float
            Time coordinate

        Returns
        -------
        Dict[str, np.ndarray]
            Flow line data including positions and velocities
        """
        g_tt = metric["g_tt"]
        g_tx = metric["g_tx"]

        # Eulerian observers: u^t = sqrt(-1/(g_tt + g_tx^2)) where timelike
        denom = g_tt + g_tx**2
        u_t = np.zeros_like(denom)
        timelike = denom < 0
        u_t[timelike] = np.sqrt(-1 / denom[timelike])
        u_x = -g_tx * u_t

        return {
            "positions": np.column_stack([x, y, z]),
            "velocities": np.column_stack([u_x, np.zeros_like(x), np.zeros_like(x)]),
            "metric": metric,
            "x": x,
        }

    def check_conservation(
        self, flow_lines: Dict[str, np.ndarray], metric: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Covariant divergence of the EFE-derived stress-energy tensor.

        nabla_mu T^{mu nu} = d_mu T^{mu nu} + Gamma^mu_{mu a} T^{a nu}
                             + Gamma^nu_{mu a} T^{mu a}

        By the contracted Bianchi identity this vanishes identically for
        stress-energy obtained from the Einstein tensor, so any residual
        measures pure discretization error.

        Returns
        -------
        np.ndarray
            Magnitude of the divergence 4-vector at each grid point
        """
        x = flow_lines["x"]

        T_cov = components_to_tensor(self.energy.calculate_from_metric(metric, x), "T")
        g_inv = inverse_tensor(components_to_tensor(metric, "g"))
        T_con = np.einsum("ma...,nb...,ab...->mn...", g_inv, g_inv, T_cov)

        gamma = self.christoffel.calculate_array(metric, x)

        divergence = np.zeros((4,) + T_con.shape[2:])
        for nu in range(4):
            # Only x-derivatives (mu=1) are nonzero on the 1-D slice
            divergence[nu] = self.fd.derivative1(T_con[1, nu], x, axis=0)
            for mu in range(4):
                for a in range(4):
                    divergence[nu] += gamma[mu, mu, a] * T_con[a, nu]
                    divergence[nu] += gamma[nu, mu, a] * T_con[mu, a]

        return np.sqrt(np.sum(divergence**2, axis=0))
