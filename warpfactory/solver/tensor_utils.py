"""Shared tensor utilities.

Canonical component-dict <-> indexed-array conversion used by the solver
and analyzer packages. Metric dictionaries use keys like "g_tt", "g_tx";
stress-energy dictionaries use "T_tt", "T_tx". The coordinate order is
fixed as (t, x, y, z) and defined only here.
"""

import numpy as np
from typing import Dict, Tuple

COORDS: Tuple[str, ...] = ("t", "x", "y", "z")


def component_key(prefix: str, mu: int, nu: int) -> str:
    a, b = COORDS[mu], COORDS[nu]
    return f"{prefix}_{a}{b}"


def components_to_tensor(components: Dict[str, np.ndarray],
                         prefix: str = "g") -> np.ndarray:
    """Build a symmetric rank-2 tensor array from a component dict.

    Missing off-diagonal components default to zero. Missing diagonal
    components default to Minkowski values for the metric prefix "g"
    (-1, +1, +1, +1) and to zero otherwise, so partial dictionaries
    produced by the metric classes remain usable.

    Returns array of shape (4, 4) + grid_shape.
    """
    grid_shape = None
    for value in components.values():
        grid_shape = np.shape(value)
        break
    if grid_shape is None:
        raise ValueError("Empty component dictionary")

    tensor = np.zeros((4, 4) + grid_shape)
    minkowski_diag = (-1.0, 1.0, 1.0, 1.0)
    for mu in range(4):
        for nu in range(4):
            key = component_key(prefix, mu, nu)
            key_sym = component_key(prefix, nu, mu)
            if key in components:
                tensor[mu, nu] = components[key]
            elif key_sym in components:
                tensor[mu, nu] = components[key_sym]
            elif mu == nu and prefix == "g":
                tensor[mu, nu] = minkowski_diag[mu]
    return tensor


def tensor_to_components(tensor: np.ndarray,
                         prefix: str = "g") -> Dict[str, np.ndarray]:
    """Flatten the upper triangle of a symmetric rank-2 tensor to a dict."""
    components = {}
    for mu in range(4):
        for nu in range(mu, 4):
            components[component_key(prefix, mu, nu)] = tensor[mu, nu]
    return components


def inverse_tensor(tensor: np.ndarray) -> np.ndarray:
    """Pointwise inverse of a (4, 4) + grid_shape tensor field."""
    grid_shape = tensor.shape[2:]
    stacked = np.moveaxis(tensor.reshape(4, 4, -1), -1, 0)
    inverted = np.linalg.inv(stacked)
    return np.moveaxis(inverted, 0, -1).reshape((4, 4) + grid_shape)
