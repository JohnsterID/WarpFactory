"""Energy condition evaluation.

Follows the approach of the MATLAB getEnergyConditions.m: instead of
assuming a perfect fluid, each condition is evaluated pointwise by
contracting the stress-energy tensor with a sampled field of timelike or
null observer vectors. Directions are distributed on the unit sphere with
a Fibonacci lattice (the analogue of getEvenPointsOnSphere.m), so
anisotropic warp-drive stress-energy is handled correctly.

The tensor dict is interpreted as covariant components on a locally flat
(Minkowski) background, which matches how the rest of this package
produces stress-energy on asymptotically flat slices.
"""

import numpy as np
from typing import Dict

from ..solver import components_to_tensor

_MINKOWSKI = np.diag([-1.0, 1.0, 1.0, 1.0])


def _sphere_directions(count: int) -> np.ndarray:
    """Approximately even unit vectors via a Fibonacci lattice."""
    indices = np.arange(count) + 0.5
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    cos_theta = 1.0 - 2.0 * indices / count
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = golden_angle * indices
    return np.stack([sin_theta * np.cos(phi),
                     sin_theta * np.sin(phi),
                     cos_theta], axis=1)


class EnergyConditions:
    """Pointwise energy condition checks by observer sampling.

    Parameters
    ----------
    num_directions : int
        Number of spatial directions sampled on the unit sphere
    velocities : tuple of float
        Speeds (in c) of the timelike observers tested per direction
    tolerance : float
        Violations smaller than this are treated as numerical noise
    """

    def __init__(self, num_directions: int = 32,
                 velocities: tuple = (0.0, 0.3, 0.6, 0.9),
                 tolerance: float = 1e-9):
        self.directions = _sphere_directions(num_directions)
        self.velocities = velocities
        self.tolerance = tolerance

    def _tensor(self, T_munu: Dict[str, np.ndarray]) -> np.ndarray:
        T = components_to_tensor(T_munu, "T")
        grid_size = int(np.prod(T.shape[2:])) if T.ndim > 2 else 1
        return T.reshape(4, 4, grid_size)

    def _timelike_observers(self) -> np.ndarray:
        observers = []
        for v in self.velocities:
            gamma = 1.0 / np.sqrt(1.0 - v**2)
            if v == 0.0:
                observers.append(np.array([1.0, 0.0, 0.0, 0.0]))
                continue
            for n in self.directions:
                observers.append(gamma * np.concatenate([[1.0], v * n]))
        return np.array(observers)

    def _null_vectors(self) -> np.ndarray:
        return np.hstack([np.ones((len(self.directions), 1)), self.directions])

    def check_weak(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Weak: T_munu t^mu t^nu >= 0 for every timelike t^mu."""
        T = self._tensor(T_munu)
        for t_vec in self._timelike_observers():
            density = np.einsum("m,mng,n->g", t_vec, T, t_vec)
            if np.any(density < -self.tolerance):
                return False
        return True

    def check_null(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Null: T_munu k^mu k^nu >= 0 for every null k^mu."""
        T = self._tensor(T_munu)
        for k_vec in self._null_vectors():
            density = np.einsum("m,mng,n->g", k_vec, T, k_vec)
            if np.any(density < -self.tolerance):
                return False
        return True

    def check_strong(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Strong: (T_munu - T g_munu / 2) t^mu t^nu >= 0 for timelike t^mu."""
        T = self._tensor(T_munu)
        g_inv = np.linalg.inv(_MINKOWSKI)
        trace = np.einsum("mn,mng->g", g_inv, T)
        effective = T - 0.5 * trace[np.newaxis, np.newaxis, :] * _MINKOWSKI[:, :, np.newaxis]
        for t_vec in self._timelike_observers():
            density = np.einsum("m,mng,n->g", t_vec, effective, t_vec)
            if np.any(density < -self.tolerance):
                return False
        return True

    def check_dominant(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Dominant: -T^mu_nu t^nu is future-directed causal for timelike t^mu."""
        T = self._tensor(T_munu)
        g_inv = np.linalg.inv(_MINKOWSKI)
        T_mixed = np.einsum("ma,ang->mng", g_inv, T)
        for t_vec in self._timelike_observers():
            flux = -np.einsum("mng,n->mg", T_mixed, t_vec)
            norm = np.einsum("mg,mn,ng->g", flux, _MINKOWSKI, flux)
            future = flux[0] >= -self.tolerance
            causal = norm <= self.tolerance
            if not (np.all(future) and np.all(causal)):
                return False
        return True
