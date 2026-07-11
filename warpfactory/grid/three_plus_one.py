"""ADM (3+1) composition and decomposition on 4-D grids.

Ports of MATLAB threePlusOneBuilder.m, threePlusOneDecomposer.m and
setMinkowskiThreePlusOne.m operating on (Nt, Nx, Ny, Nz) arrays.
"""

from typing import Tuple

import numpy as np

from ..solver.tensor_utils import inverse_tensor
from .tensor import SpacetimeTensor, change_tensor_index


def minkowski_three_plus_one(grid_shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat-space ADM variables: alpha = 1, beta = 0, gamma = identity.

    Returns
    -------
    alpha : np.ndarray, shape grid_shape
    beta : np.ndarray, shape (3,) + grid_shape (covariant)
    gamma : np.ndarray, shape (3, 3) + grid_shape (covariant)
    """
    grid_shape = tuple(grid_shape)
    alpha = np.ones(grid_shape)
    beta = np.zeros((3,) + grid_shape)
    gamma = np.zeros((3, 3) + grid_shape)
    for i in range(3):
        gamma[i, i] = 1.0
    return alpha, beta, gamma


def three_plus_one_builder(alpha: np.ndarray, beta: np.ndarray,
                           gamma: np.ndarray) -> np.ndarray:
    """Assemble the covariant metric from ADM variables.

    g_00 = -alpha^2 + beta^i beta_i,  g_0i = beta_i,  g_ij = gamma_ij
    beta and gamma are covariant (indices down).

    Returns array of shape (4, 4) + grid_shape.
    """
    grid_shape = alpha.shape
    gamma_up = inverse_tensor(gamma)
    beta_up = np.einsum("ij...,j...->i...", gamma_up, beta)

    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -alpha**2 + np.einsum("i...,i...->...", beta_up, beta)
    for i in range(3):
        g[0, i + 1] = beta[i]
        g[i + 1, 0] = beta[i]
        for j in range(3):
            g[i + 1, j + 1] = gamma[i, j]
    return g


def three_plus_one_decomposer(metric: SpacetimeTensor):
    """Recover ADM variables from a covariant grid metric.

    Returns
    -------
    alpha : np.ndarray
        Lapse
    beta_down, gamma_down : np.ndarray
        Covariant shift (3,) + grid and spatial metric (3, 3) + grid
    beta_up, gamma_up : np.ndarray
        Contravariant counterparts
    """
    metric = change_tensor_index(metric, "covariant")
    g = metric.tensor

    beta_down = g[0, 1:4]
    gamma_down = g[1:4, 1:4]
    gamma_up = inverse_tensor(gamma_down)
    beta_up = np.einsum("ij...,j...->i...", gamma_up, beta_down)
    alpha = np.sqrt(np.einsum("i...,i...->...", beta_up, beta_down) - g[0, 0])
    return alpha, beta_down, gamma_down, beta_up, gamma_up
