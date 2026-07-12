"""Pointwise metric interpolation and Christoffel symbols on an x line.

The spacetime analysis tools (geodesics, lensing, horizons,
singularities) need the metric and its derivatives at arbitrary
positions, not just grid nodes. This helper interpolates the sampled
1-D metric profiles and evaluates the exact Christoffel contraction

    Gamma^a_bc = g^{ad} (d_b g_dc + d_c g_db - d_d g_bc) / 2

with only x-derivatives nonzero (stationary axial slices).
"""

from typing import Dict

import numpy as np

X_MIN, X_MAX = -5.0, 5.0

_COMPONENT_INDEX = {
    "g_tt": (0, 0),
    "g_tx": (0, 1),
    "g_ty": (0, 2),
    "g_tz": (0, 3),
    "g_xx": (1, 1),
    "g_xy": (1, 2),
    "g_xz": (1, 3),
    "g_yy": (2, 2),
    "g_yz": (2, 3),
    "g_zz": (3, 3),
}


class MetricLine:
    """Interpolated metric along the x axis.

    Parameters
    ----------
    metric : Dict[str, np.ndarray]
        Sampled metric components
    x : np.ndarray, optional
        Grid the components were sampled on; defaults to the package
        standard uniform grid on [-5, 5]
    """

    def __init__(self, metric: Dict[str, np.ndarray], x: np.ndarray = None):
        self.metric = metric
        n = len(np.asarray(metric["g_tt"]))
        self.x = (
            np.asarray(x, dtype=float)
            if x is not None
            else np.linspace(X_MIN, X_MAX, n)
        )

    def tensor_at(self, x_pos: float) -> np.ndarray:
        """4x4 covariant metric tensor at position x_pos."""
        g = np.diag([-1.0, 1.0, 1.0, 1.0])
        for key, values in self.metric.items():
            mu, nu = _COMPONENT_INDEX[key]
            g[mu, nu] = g[nu, mu] = float(np.interp(x_pos, self.x, values))
        return g

    def christoffel_at(self, x_pos: float, dx: float = 1e-5) -> np.ndarray:
        """Christoffel symbols Gamma^a_bc (shape (4, 4, 4)) at x_pos."""
        g = self.tensor_at(x_pos)
        g_inv = np.linalg.inv(g)
        dg_dx = (self.tensor_at(x_pos + dx) - self.tensor_at(x_pos - dx)) / (2 * dx)

        # dg[d] = d_d g; only the x direction (index 1) is nonzero
        dg = np.zeros((4, 4, 4))
        dg[1] = dg_dx

        gamma = np.zeros((4, 4, 4))
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    total = 0.0
                    for d in range(4):
                        total += g_inv[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                    gamma[a, b, c] = 0.5 * total
        return gamma

    def null_velocity(self, position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Coordinate velocity of a light ray along a spatial direction.

        Scales the unit direction n onto the future light cone by
        solving g_tt + 2 g_ti (s n^i) + g_ij (s n^i)(s n^j) = 0 for the
        larger root s > 0 (future-directed ray along +n).
        """
        g = self.tensor_at(position[0])
        n = direction / np.linalg.norm(direction)
        a = n @ g[1:, 1:] @ n
        b = 2.0 * (g[0, 1:] @ n)
        c = g[0, 0]
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError(
                "No null direction along the requested spatial "
                f"direction at x={position[0]:.3f}"
            )
        s = (-b + np.sqrt(disc)) / (2 * a)
        return s * n

    def coordinate_acceleration(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """Geodesic acceleration in coordinate-time parametrization.

        For w^a = (1, v) the reduction of the geodesic equation to
        coordinate time reads

            dv^i/dt = -(Gamma^i_ab - v^i Gamma^t_ab) w^a w^b

        which is exact for timelike AND null geodesics (the affine
        parametrization factor cancels).
        """
        gamma = self.christoffel_at(position[0])
        w = np.concatenate([[1.0], velocity])
        contraction = np.einsum("abc,b,c->a", gamma, w, w)
        return -(contraction[1:] - velocity * contraction[0])
