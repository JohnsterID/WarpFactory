"""Kinematic scalars of the Eulerian congruence on 4-D grids.

Python port of the MATLAB getScalars.m: expansion, shear, and vorticity
of the Eulerian (normal) observer 4-velocity field
u^mu = (1/alpha)(1, -beta^i). The covariant derivative of u is built
from the same finite-difference Christoffel symbols as the stress-energy
solver, then projected onto the spatial hypersurface with
P^a_b = delta^a_b + u^a u_b:

    theta_ij = P^a_i P^b_j nabla_(a u_b)   (expansion tensor)
    omega_ij = P^a_i P^b_j nabla_[a u_b]   (vorticity tensor)
    sigma_ij = theta_ij - (theta/3) P_ij   (shear tensor)

with scalars theta = g^ij theta_ij, sigma^2 = sigma_ij sigma^ij / 2,
omega^2 = omega_ij omega^ij / 2.
"""

from typing import Tuple

import numpy as np

from ..solver.tensor_utils import inverse_tensor
from .solver import GridSolver
from .tensor import SpacetimeTensor, change_tensor_index, verify_tensor
from .three_plus_one import three_plus_one_decomposer


def eulerian_velocity(metric: SpacetimeTensor) -> Tuple[np.ndarray, np.ndarray]:
    """Eulerian observer 4-velocity, contravariant and covariant.

    Returns
    -------
    u_up, u_down : np.ndarray
        Shape (4,) + grid_shape each; u^mu = (1/alpha)(1, -beta^i) and
        u_mu = g_munu u^nu = (-alpha, 0, 0, 0).
    """
    alpha, _, _, beta_up, _ = three_plus_one_decomposer(metric)
    grid_shape = metric.grid_shape

    u_up = np.empty((4,) + grid_shape)
    u_up[0] = 1.0 / alpha
    u_up[1:] = -beta_up / alpha

    g = change_tensor_index(metric, "covariant").tensor
    u_down = np.einsum("mn...,n...->m...", g, u_up)
    return u_up, u_down


def get_scalars(
    metric: SpacetimeTensor, order: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expansion, shear, and vorticity scalars (MATLAB getScalars).

    Parameters
    ----------
    metric : SpacetimeTensor
        Metric on a uniform 4-D grid
    order : int
        Finite difference accuracy order, 2 or 4

    Returns
    -------
    expansion : np.ndarray
        theta at every grid point; shape grid_shape
    shear : np.ndarray
        sigma^2 = sigma_ij sigma^ij / 2
    vorticity : np.ndarray
        omega^2 = omega_ij omega^ij / 2
    """
    if not verify_tensor(metric):
        raise ValueError(
            "Metric failed verification; see verify_tensor(metric, quiet=False)"
        )
    metric = change_tensor_index(metric, "covariant")
    g = np.asarray(metric.tensor, dtype=float)
    g_inv = inverse_tensor(g)

    solver = GridSolver(order=order)
    dg = solver.metric_derivative1(g, metric.scaling)
    gamma = solver.christoffel(g_inv, dg)

    u_up, u_down = eulerian_velocity(metric)

    # nabla_a u_b = d_a u_b - Gamma^e_ba u_e
    du = np.stack(
        [
            solver.fd.derivative1_delta(u_down[b], metric.scaling[a], axis=a)
            if g.shape[2 + a] >= 2
            else np.zeros(metric.grid_shape)
            for a in range(4)
            for b in range(4)
        ]
    )
    du = du.reshape((4, 4) + metric.grid_shape)
    nabla_u = du - np.einsum("eba...,e...->ab...", gamma, u_down)

    # Projector P^a_b = delta^a_b + u^a u_b and P_ab = g_ab + u_a u_b.
    delta = np.eye(4).reshape((4, 4) + (1,) * len(metric.grid_shape))
    P_mix = delta + np.einsum("a...,b...->ab...", u_up, u_down)
    P_down = g + np.einsum("a...,b...->ab...", u_down, u_down)

    sym = 0.5 * (nabla_u + np.swapaxes(nabla_u, 0, 1))
    antisym = 0.5 * (nabla_u - np.swapaxes(nabla_u, 0, 1))
    theta = np.einsum("ai...,bj...,ab...->ij...", P_mix, P_mix, sym)
    omega = np.einsum("ai...,bj...,ab...->ij...", P_mix, P_mix, antisym)

    expansion = np.einsum("ij...,ij...->...", g_inv, theta)

    shear_tensor = theta - expansion / 3.0 * P_down
    shear_up = np.einsum("ia...,jb...,ab...->ij...", g_inv, g_inv, shear_tensor)
    shear = 0.5 * np.einsum("ij...,ij...->...", shear_tensor, shear_up)

    omega_up = np.einsum("ia...,jb...,ab...->ij...", g_inv, g_inv, omega)
    vorticity = 0.5 * np.einsum("ij...,ij...->...", omega, omega_up)

    return expansion, shear, vorticity
