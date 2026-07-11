"""Grid interpolation utilities.

Ports of MATLAB trilinearInterp.m, quadrilinearInterp.m and
legendreRadialInterp.m using zero-based indexing: sample positions are
continuous array indices (0 <= pos <= N-1), matching how the MATLAB
originals treat positions as 1-based indices into the grid.
"""

import numpy as np


def trilinear_interp(field: np.ndarray, position) -> float:
    """Trilinear interpolation of a 3-D field at a fractional index.

    Parameters
    ----------
    field : np.ndarray
        3-D array
    position : sequence of 3 floats
        Fractional (i, j, k) index position
    """
    pos = np.asarray(position, dtype=float)
    lo = np.clip(np.floor(pos).astype(int), 0, np.array(field.shape) - 1)
    hi = np.clip(lo + 1, 0, np.array(field.shape) - 1)
    frac = pos - lo

    c00 = field[lo[0], lo[1], lo[2]]*(1 - frac[0]) + field[hi[0], lo[1], lo[2]]*frac[0]
    c01 = field[lo[0], lo[1], hi[2]]*(1 - frac[0]) + field[hi[0], lo[1], hi[2]]*frac[0]
    c10 = field[lo[0], hi[1], lo[2]]*(1 - frac[0]) + field[hi[0], hi[1], lo[2]]*frac[0]
    c11 = field[lo[0], hi[1], hi[2]]*(1 - frac[0]) + field[hi[0], hi[1], hi[2]]*frac[0]

    c0 = c00*(1 - frac[1]) + c10*frac[1]
    c1 = c01*(1 - frac[1]) + c11*frac[1]
    return float(c0*(1 - frac[2]) + c1*frac[2])


def quadrilinear_interp(field: np.ndarray, position) -> float:
    """Quadrilinear interpolation of a 4-D field at a fractional index.

    Trilinear interpolation on the two bracketing time slices followed
    by linear interpolation in time.
    """
    pos = np.asarray(position, dtype=float)
    t_lo = int(np.clip(np.floor(pos[0]), 0, field.shape[0] - 1))
    t_hi = int(np.clip(t_lo + 1, 0, field.shape[0] - 1))

    c_lo = trilinear_interp(field[t_lo], pos[1:])
    if t_hi == t_lo:
        return c_lo
    c_hi = trilinear_interp(field[t_hi], pos[1:])
    frac = pos[0] - t_lo
    return c_lo*(1 - frac) + c_hi*frac


def legendre_radial_interp(values: np.ndarray, r) -> np.ndarray:
    """Cubic Lagrange interpolation of a radial profile at index r.

    Port of legendreRadialInterp.m: third-order polynomial through the
    four samples bracketing the fractional index r (zero-based). Node
    indices are clamped to the array bounds while keeping the
    polynomial abscissae, as in the original. r may be a scalar or an
    array of fractional indices.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    r = np.asarray(r, dtype=float)
    x1 = np.floor(r).astype(int)
    x0, x2, x3 = x1 - 1, x1 + 1, x1 + 2

    def sample(xi):
        return values[np.clip(xi, 0, n - 1)]

    y0, y1, y2, y3 = sample(x0), sample(x1), sample(x2), sample(x3)

    return (
        y0*(r - x1)*(r - x2)*(r - x3)/((x0 - x1)*(x0 - x2)*(x0 - x3))
        + y1*(r - x0)*(r - x2)*(r - x3)/((x1 - x0)*(x1 - x2)*(x1 - x3))
        + y2*(r - x0)*(r - x1)*(r - x3)/((x2 - x0)*(x2 - x1)*(x2 - x3))
        + y3*(r - x0)*(r - x1)*(r - x2)/((x3 - x0)*(x3 - x1)*(x3 - x2))
    )
