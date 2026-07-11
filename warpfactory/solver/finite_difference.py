"""Finite difference methods for numerical derivatives.

Mirrors the MATLAB takeFiniteDifference1/takeFiniteDifference2 utilities:
central differences of selectable order (2nd or 4th) on uniform grids,
applicable along any axis of an N-dimensional field.
"""

import numpy as np


def _grid_spacing(x: np.ndarray) -> float:
    if len(x) > 1:
        return x[1] - x[0]
    return 1.0


class FiniteDifference:
    """Finite difference derivatives on uniform grids.

    Parameters
    ----------
    order : int
        Accuracy order of the interior stencil, 2 or 4. The original
        MATLAB solver defaults to 4th order; 2nd order matches its
        'second' option.
    """

    def __init__(self, order: int = 2):
        if order not in (2, 4):
            raise ValueError("order must be 2 or 4")
        self.order = order

    def derivative1(self, f: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """First derivative df/dx along the given axis."""
        return self.derivative1_delta(f, _grid_spacing(x), axis=axis)

    def derivative1_delta(self, f: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
        """First derivative with an explicit uniform grid spacing."""
        f = np.asarray(f, dtype=float)
        if f.shape[axis] < 2:
            return np.zeros_like(f)

        moved = np.moveaxis(f, axis, 0)
        df = np.empty_like(moved)

        if self.order == 4 and moved.shape[0] >= 5:
            df[2:-2] = (moved[:-4] - 8*moved[1:-3] + 8*moved[3:-1] - moved[4:]) / (12*dx)
            df[1] = (moved[2] - moved[0]) / (2*dx)
            df[-2] = (moved[-1] - moved[-3]) / (2*dx)
        else:
            df[1:-1] = (moved[2:] - moved[:-2]) / (2*dx)
        df[0] = (moved[1] - moved[0]) / dx
        df[-1] = (moved[-1] - moved[-2]) / dx
        return np.moveaxis(df, 0, axis)

    def derivative2(self, f: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Second derivative d2f/dx2 along the given axis."""
        return self.derivative2_delta(f, _grid_spacing(x), axis=axis)

    def derivative2_delta(self, f: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
        """Second derivative with an explicit uniform grid spacing."""
        f = np.asarray(f, dtype=float)
        if f.shape[axis] < 3:
            return np.zeros_like(f)

        moved = np.moveaxis(f, axis, 0)
        d2f = np.empty_like(moved)

        if self.order == 4 and moved.shape[0] >= 5:
            d2f[2:-2] = (-moved[:-4] + 16*moved[1:-3] - 30*moved[2:-2]
                         + 16*moved[3:-1] - moved[4:]) / (12*dx**2)
            d2f[1] = (moved[2] - 2*moved[1] + moved[0]) / dx**2
            d2f[-2] = (moved[-1] - 2*moved[-2] + moved[-3]) / dx**2
        else:
            d2f[1:-1] = (moved[2:] - 2*moved[1:-1] + moved[:-2]) / dx**2
        d2f[0] = (moved[2] - 2*moved[1] + moved[0]) / dx**2
        d2f[-1] = (moved[-1] - 2*moved[-2] + moved[-3]) / dx**2
        return np.moveaxis(d2f, 0, axis)

    def mixed_derivative2(self, f: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                          axis1: int, axis2: int) -> np.ndarray:
        """Mixed second derivative d2f/(dx1 dx2)."""
        return self.mixed_derivative2_delta(f, _grid_spacing(x1), _grid_spacing(x2),
                                            axis1, axis2)

    def mixed_derivative2_delta(self, f: np.ndarray, dx1: float, dx2: float,
                                axis1: int, axis2: int) -> np.ndarray:
        """Mixed second derivative with explicit uniform grid spacings."""
        return self.derivative1_delta(
            self.derivative1_delta(f, dx1, axis=axis1), dx2, axis=axis2)
