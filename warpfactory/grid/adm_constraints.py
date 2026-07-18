"""ADM constraint residuals for 4-D grid metrics.

The initial-data half of the 3+1 evolution problem: any metric slice,
together with the matter measured by the Eulerian observer, must
satisfy the Hamiltonian and momentum constraints (Arnowitt-Deser-Misner;
conventions of Baumgarte & Shapiro, "Numerical Relativity", eqs.
2.132/2.133, G = c = 1):

    H   = R3 + K^2 - K_ij K^ij - 16 pi rho      = 0
    M_i = D_j (K^j_i - delta^j_i K) - 8 pi S_i  = 0

with R3 the spatial Ricci scalar, K_ij the extrinsic curvature,
rho = n^a n^b T_ab, S_i = -n^a T_ai, and D the covariant derivative of
the spatial metric. For stress-energy computed from the same metric by
the EFE solver the residuals are pure discretization error (the
constraints are the G_ab n^b projections of the field equations), so
they quantify grid quality. For matter supplied from elsewhere -- a
modified-gravity effective source, an analytic matter model, a
perturbed field -- nonzero residuals mean the slice is NOT valid
initial data for that matter, which is exactly what a 3+1 evolution
code would reject.

Evolving the data forward in time (the other half of the problem)
additionally requires a hyperbolic reformulation (BSSN or generalized
harmonic), constraint-damping, and horizon-adapted gauge drivers; that
is Einstein Toolkit / GRChombo territory and out of scope here.

Sign convention: K_ij = -nabla_i n_j (MTW/ADM), so
K_ij = (D_i beta_j + D_j beta_i - d_t gamma_ij) / (2 alpha), and the
Alcubierre expansion theta = -K matches get_scalars.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .solver import GridSolver, christoffel_from_derivatives, ricci_from_derivatives
from .tensor import SpacetimeTensor, change_tensor_index, verify_tensor
from .three_plus_one import three_plus_one_decomposer


@dataclass
class ADMConstraintResult:
    """Constraint residual maps on the grid.

    Attributes
    ----------
    hamiltonian : np.ndarray
        H residual; shape grid_shape
    momentum : np.ndarray
        M_i residual (covariant spatial index); shape (3,) + grid_shape
    ricci_3_scalar : np.ndarray
        Spatial Ricci scalar R3 of the slice
    extrinsic_curvature : np.ndarray
        K_ij (covariant); shape (3, 3) + grid_shape
    mean_curvature : np.ndarray
        K = gamma^ij K_ij
    energy_density : np.ndarray
        Eulerian rho used in the Hamiltonian constraint
    momentum_density : np.ndarray
        Eulerian S_i used in the momentum constraint
    """

    hamiltonian: np.ndarray
    momentum: np.ndarray
    ricci_3_scalar: np.ndarray
    extrinsic_curvature: np.ndarray
    mean_curvature: np.ndarray
    energy_density: np.ndarray
    momentum_density: np.ndarray


class ADMConstraints:
    """Hamiltonian and momentum constraint evaluator.

    Parameters
    ----------
    order : int
        Finite difference accuracy order, 2 or 4
    """

    def __init__(self, order: int = 4):
        self.grid_solver = GridSolver(order=order)
        self.order = order

    def _spatial_derivative1(self, field: np.ndarray, scaling) -> np.ndarray:
        """d_k along the three spatial axes; k indexes x, y, z."""
        grid_shape = field.shape[-4:]
        out = np.zeros((3,) + field.shape)
        for k in range(3):
            if grid_shape[1 + k] < 2:
                continue
            out[k] = self.grid_solver.fd.derivative1_delta(
                field, scaling[1 + k], axis=field.ndim - 3 + k
            )
        return out

    def _spatial_metric_derivatives(self, gamma_down: np.ndarray, scaling):
        """First and second spatial derivatives of the 3-metric."""
        grid_shape = gamma_down.shape[2:]
        dg = np.zeros((3, 3, 3) + grid_shape)
        d2g = np.zeros((3, 3, 3, 3) + grid_shape)
        fd = self.grid_solver.fd
        for k in range(3):
            if grid_shape[1 + k] < 2:
                continue
            for i in range(3):
                for j in range(i, 3):
                    d = fd.derivative1_delta(
                        gamma_down[i, j], scaling[1 + k], axis=1 + k
                    )
                    dg[k, i, j] = d
                    if i != j:
                        dg[k, j, i] = d
        for k in range(3):
            if grid_shape[1 + k] < 2:
                continue
            for n in range(k, 3):
                if grid_shape[1 + n] < 2:
                    continue
                for i in range(3):
                    for j in range(i, 3):
                        if k == n:
                            d = fd.derivative2_delta(
                                gamma_down[i, j], scaling[1 + k], axis=1 + k
                            )
                        else:
                            d = fd.mixed_derivative2_delta(
                                gamma_down[i, j],
                                scaling[1 + k],
                                scaling[1 + n],
                                1 + k,
                                1 + n,
                            )
                        d2g[k, n, i, j] = d
                        d2g[k, n, j, i] = d
                        d2g[n, k, i, j] = d
                        d2g[n, k, j, i] = d
        return dg, d2g

    def evaluate(
        self,
        metric: SpacetimeTensor,
        stress_energy: Optional[SpacetimeTensor] = None,
    ) -> ADMConstraintResult:
        """Constraint residuals of a metric slice.

        Parameters
        ----------
        metric : SpacetimeTensor
            Metric on a uniform 4-D grid. Nt = 1 grids treat time
            derivatives as zero (matching the solver convention), which
            is exact for comoving/stationary slicings.
        stress_energy : SpacetimeTensor, optional
            Matter to test the slice against. Omitted means vacuum
            (rho = S_i = 0). Pass GridSolver output to measure pure
            discretization error, or any other source (modified
            gravity, analytic matter) to test whether the slice is
            valid initial data for it.

        Returns
        -------
        ADMConstraintResult
        """
        if not verify_tensor(metric):
            raise ValueError(
                "Metric failed verification; see verify_tensor(metric, quiet=False)"
            )
        metric = change_tensor_index(metric, "covariant")
        scaling = metric.scaling
        grid_shape = metric.grid_shape

        alpha, beta_down, gamma_down, beta_up, gamma_up = three_plus_one_decomposer(
            metric
        )

        dgamma, d2gamma = self._spatial_metric_derivatives(gamma_down, scaling)
        # christoffel/ricci_from_derivatives are dimension-generic:
        # feeding 3-index arrays yields the spatial (3-D) objects.
        gamma3 = christoffel_from_derivatives(gamma_up, dgamma)
        ricci3 = ricci_from_derivatives(gamma_down, gamma_up, dgamma, d2gamma)
        ricci_3_scalar = np.einsum("ij...,ij...->...", gamma_up, ricci3)

        # K_ij = (D_i beta_j + D_j beta_i - d_t gamma_ij) / (2 alpha)
        dbeta = self._spatial_derivative1(beta_down, scaling)
        D_beta = dbeta - np.einsum("kij...,k...->ij...", gamma3, beta_down)
        dt_gamma = np.zeros((3, 3) + grid_shape)
        if grid_shape[0] >= 2:
            for i in range(3):
                for j in range(i, 3):
                    d = self.grid_solver.fd.derivative1_delta(
                        gamma_down[i, j], scaling[0], axis=0
                    )
                    dt_gamma[i, j] = d
                    dt_gamma[j, i] = d
        K_down = (D_beta + np.swapaxes(D_beta, 0, 1) - dt_gamma) / (2.0 * alpha)
        K_mixed = np.einsum("ik...,kj...->ij...", gamma_up, K_down)
        K_trace = np.einsum("ii...->...", K_mixed)
        K_sq = np.einsum("ij...,ji...->...", K_mixed, K_mixed)

        if stress_energy is not None:
            T_down = change_tensor_index(stress_energy, "covariant", metric).tensor
            # n^mu = (1/alpha)(1, -beta^i); n_i = 0 so the projector
            # onto the slice is the identity on spatial indices.
            rho = (
                T_down[0, 0]
                - 2.0 * np.einsum("i...,i...->...", beta_up, T_down[0, 1:4])
                + np.einsum("i...,j...,ij...->...", beta_up, beta_up, T_down[1:4, 1:4])
            ) / alpha**2
            S_down = (
                -(
                    T_down[0, 1:4]
                    - np.einsum("j...,ji...->i...", beta_up, T_down[1:4, 1:4])
                )
                / alpha
            )
        else:
            rho = np.zeros(grid_shape)
            S_down = np.zeros((3,) + grid_shape)

        hamiltonian = ricci_3_scalar + K_trace**2 - K_sq - 16.0 * np.pi * rho

        # M_i = D_j A^j_i - 8 pi S_i with A^j_i = K^j_i - delta^j_i K
        delta3 = np.eye(3).reshape((3, 3) + (1,) * len(grid_shape))
        A = K_mixed - delta3 * K_trace
        dA = self._spatial_derivative1(A, scaling)
        div_A = (
            np.einsum("jji...->i...", dA)
            + np.einsum("jjk...,ki...->i...", gamma3, A)
            - np.einsum("kji...,jk...->i...", gamma3, A)
        )
        momentum = div_A - 8.0 * np.pi * S_down

        return ADMConstraintResult(
            hamiltonian=hamiltonian,
            momentum=momentum,
            ricci_3_scalar=ricci_3_scalar,
            extrinsic_curvature=K_down,
            mean_curvature=K_trace,
            energy_density=rho,
            momentum_density=S_down,
        )
