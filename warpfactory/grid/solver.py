"""Stress-energy solver on full 4-D grids.

Python port of the MATLAB met2den/met2den2/getEnergyTensor pipeline:
finite-difference the metric on a uniform (t, x, y, z) grid, build the
Ricci and Einstein tensors, and read the stress-energy tensor off the
Einstein field equations. Works in geometric units (G = c = 1), matching
the 1-D slice solver in warpfactory.solver; the MATLAB original's SI
factors (1/c time scaling, c^4/8piG) are intentionally absent.

Derivatives along any grid axis with fewer than the minimum stencil
points are treated as zero, so single-time-slice grids (Nt = 1) work the
same way the MATLAB comoving metrics do.
"""

import numpy as np

from ..solver.finite_difference import FiniteDifference
from ..solver.tensor_utils import inverse_tensor
from .tensor import SpacetimeTensor, change_tensor_index, verify_tensor


class GridSolver:
    """Einstein field equation solver for metrics on 4-D grids.

    Parameters
    ----------
    order : int
        Finite difference accuracy order, 2 or 4 (MATLAB 'second' /
        'fourth').
    """

    def __init__(self, order: int = 4):
        self.fd = FiniteDifference(order=order)
        self.order = order

    def metric_derivative1(self, g: np.ndarray,
                           scaling) -> np.ndarray:
        """d_k g_munu for all k; shape (4, 4, 4) + grid (k first)."""
        dg = np.zeros((4,) + g.shape)
        for k in range(4):
            if g.shape[2 + k] < 2:
                continue
            for mu in range(4):
                for nu in range(mu, 4):
                    d = self.fd.derivative1_delta(g[mu, nu], scaling[k], axis=k)
                    dg[k, mu, nu] = d
                    if mu != nu:
                        dg[k, nu, mu] = d
        return dg

    def metric_derivative2(self, g: np.ndarray,
                           scaling) -> np.ndarray:
        """d_k d_n g_munu for all k, n; shape (4, 4, 4, 4) + grid."""
        d2g = np.zeros((4, 4) + g.shape)
        for k in range(4):
            if g.shape[2 + k] < 2:
                continue
            for n in range(k, 4):
                if g.shape[2 + n] < 2:
                    continue
                for mu in range(4):
                    for nu in range(mu, 4):
                        if k == n:
                            d = self.fd.derivative2_delta(
                                g[mu, nu], scaling[k], axis=k)
                        else:
                            d = self.fd.mixed_derivative2_delta(
                                g[mu, nu], scaling[k], scaling[n], k, n)
                        d2g[k, n, mu, nu] = d
                        if mu != nu:
                            d2g[k, n, nu, mu] = d
                        if k != n:
                            d2g[n, k, mu, nu] = d
                            if mu != nu:
                                d2g[n, k, nu, mu] = d
        return d2g

    def christoffel(self, g_inv: np.ndarray, dg: np.ndarray) -> np.ndarray:
        """Gamma^a_bc = g^ad (d_b g_dc + d_c g_db - d_d g_bc) / 2.

        dg is the first-derivative array from metric_derivative1 with
        layout dg[k, mu, nu] = d_k g_munu.
        """
        # Build S[b, d, c] = d_b g_dc + d_c g_db - d_d g_bc from dg.
        symmetrized = dg + np.swapaxes(dg, 0, 2) - np.swapaxes(dg, 0, 1)
        return 0.5 * np.einsum("ad...,bdc...->abc...", g_inv, symmetrized)

    def ricci_tensor(self, g: np.ndarray, g_inv: np.ndarray,
                     scaling) -> np.ndarray:
        """Ricci tensor R_bc on the grid.

        R_bc = d_a Gamma^a_bc - d_c Gamma^a_ba
               + Gamma^a_ae Gamma^e_bc - Gamma^a_ce Gamma^e_ba

        The Christoffel derivatives are expanded analytically in the
        metric derivatives (using d_e g^ad = -g^am g^dn d_e g_mn), so a
        single finite-difference pass on the metric supplies everything;
        this matches the accuracy of the MATLAB ricciT.m direct
        expansion and avoids differentiating FD output.
        """
        dg = self.metric_derivative1(g, scaling)
        d2g = self.metric_derivative2(g, scaling)

        # S[b, d, c] = d_b g_dc + d_c g_db - d_d g_bc  (2 Gamma_dbc, all down)
        S = dg + np.swapaxes(dg, 0, 2) - np.swapaxes(dg, 0, 1)
        gamma = 0.5 * np.einsum("ad...,bdc...->abc...", g_inv, S)

        # d_e g^ad = -g^am g^dn d_e g_mn
        dg_inv = -np.einsum("am...,dn...,emn...->ead...", g_inv, g_inv, dg)

        # dS[e, b, d, c] = d_e d_b g_dc + d_e d_c g_db - d_e d_d g_bc
        dS = (d2g
              + np.swapaxes(d2g, 1, 3)          # d_e d_c g_db
              - np.swapaxes(d2g, 1, 2))         # d_e d_d g_bc
        dgamma = 0.5 * (np.einsum("ead...,bdc...->eabc...", dg_inv, S)
                        + np.einsum("ad...,ebdc...->eabc...", g_inv, dS))

        ricci = (
            np.einsum("aabc...->bc...", dgamma)
            - np.einsum("caba...->bc...", dgamma)
            + np.einsum("aae...,ebc...->bc...", gamma, gamma)
            - np.einsum("ace...,eba...->bc...", gamma, gamma)
        )
        # Symmetrize away FD round-off asymmetry.
        return 0.5 * (ricci + np.swapaxes(ricci, 0, 1))

    def solve(self, metric: SpacetimeTensor,
              contravariant: bool = True) -> SpacetimeTensor:
        """Stress-energy tensor of a grid metric (MATLAB getEnergyTensor).

        Parameters
        ----------
        metric : SpacetimeTensor
            Metric on a uniform 4-D grid
        contravariant : bool
            Return T^munu (MATLAB behavior) when True, T_munu otherwise

        Returns
        -------
        SpacetimeTensor
            Stress-energy tensor with type "stress-energy"
        """
        if not verify_tensor(metric):
            raise ValueError(
                "Metric failed verification; see verify_tensor(metric, quiet=False)")
        if metric.index.lower() != "covariant":
            metric = change_tensor_index(metric, "covariant")

        g = np.asarray(metric.tensor, dtype=float)
        g_inv = inverse_tensor(g)

        ricci = self.ricci_tensor(g, g_inv, metric.scaling)
        ricci_scalar = np.einsum("ij...,ij...->...", g_inv, ricci)

        einstein = ricci - 0.5 * ricci_scalar * g
        stress_energy = einstein / (8.0 * np.pi)
        index = "covariant"

        if contravariant:
            stress_energy = np.einsum(
                "ab...,ai...,bj...->ij...", stress_energy, g_inv, g_inv)
            index = "contravariant"

        return SpacetimeTensor(
            tensor=stress_energy, type="stress-energy", index=index,
            coords=metric.coords, scaling=metric.scaling,
            name=metric.name,
            params={"order": self.order})
