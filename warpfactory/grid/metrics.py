"""Grid metric builders on full (t, x, y, z) grids.

Ports of the MATLAB metricGet_* functions producing SpacetimeTensor
objects. Conventions (deliberate divergences from MATLAB, see README):

* Geometric units, c = G = 1. The bubble center moves as xs = v * t
  (MATLAB uses xs = v*c*t with t measured in grid steps).
* Grid coordinates are physical: point (it, ix, iy, iz) sits at
  it*dt - t0, ix*dx - x0, ... where (t0, x0, y0, z0) is world_center.
  This matches MATLAB's i*gridScale - worldCenter indexing (zero-based
  here rather than one-based).
* metricGet_ModifiedTime.m indexing bugs (gridScale vs gridScaling,
  axis-offset mismatch, the (t,i,k,k) typo) are fixed: coordinates are
  computed from the correct axes and the symmetric assignment is
  (t,i,j,k) everywhere.
"""

from typing import Sequence, Tuple

import numpy as np

from .shape_functions import alcubierre_shape
from .tensor import SpacetimeTensor
from .three_plus_one import minkowski_three_plus_one, three_plus_one_builder

GridSize = Sequence[int]
Vector4 = Sequence[float]


def _world_coordinates(grid_size: GridSize, world_center: Vector4,
                       grid_scale: Vector4) -> Tuple[np.ndarray, ...]:
    """Physical (t, x, y, z) coordinates of every grid point, broadcastable."""
    axes = []
    for k in range(4):
        idx = np.arange(grid_size[k], dtype=float)
        shape = [1, 1, 1, 1]
        shape[k] = grid_size[k]
        axes.append((idx*grid_scale[k] - world_center[k]).reshape(shape))
    return tuple(axes)


def _base_metric(name: str, grid_scale: Vector4, params: dict,
                 tensor: np.ndarray) -> SpacetimeTensor:
    return SpacetimeTensor(
        tensor=tensor, type="metric", index="covariant",
        coords="cartesian", scaling=tuple(float(s) for s in grid_scale),
        name=name, params=params)


def _require_single_time_slice(grid_size: GridSize, name: str) -> None:
    if grid_size[0] != 1:
        raise ValueError(
            f"{name} requires a single time slice (grid_size[0] == 1)")


def minkowski_metric(grid_size: GridSize,
                     grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Flat Minkowski metric on the grid (MATLAB setMinkowski)."""
    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -1.0
    for i in range(1, 4):
        g[i, i] = 1.0
    return _base_metric("Minkowski", grid_scale, {"grid_size": grid_shape}, g)


def alcubierre_metric(grid_size: GridSize, world_center: Vector4, v: float,
                      R: float, sigma: float,
                      grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Alcubierre warp drive metric (metricGet_Alcubierre)."""
    t, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    xs = v*t
    r = np.sqrt((x - xs)**2 + y**2 + z**2)
    fs = alcubierre_shape(r, R, sigma)

    alpha, beta, gamma = minkowski_three_plus_one(tuple(int(n) for n in grid_size))
    beta[0] = -v*fs

    return _base_metric(
        "Alcubierre", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v, "R": R, "sigma": sigma},
        three_plus_one_builder(alpha, beta, gamma))


def alcubierre_comoving_metric(grid_size: GridSize, world_center: Vector4,
                               v: float, R: float, sigma: float,
                               grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Alcubierre metric in the Galilean comoving frame."""
    _require_single_time_slice(grid_size, "alcubierre_comoving_metric")
    _, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    r = np.sqrt(x**2 + y**2 + z**2)
    fs = alcubierre_shape(r, R, sigma)

    alpha, beta, gamma = minkowski_three_plus_one(tuple(int(n) for n in grid_size))
    beta[0] = v*(1 - fs)

    return _base_metric(
        "Alcubierre Comoving", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v, "R": R, "sigma": sigma},
        three_plus_one_builder(alpha, beta, gamma))


def _lentz_warp_factors(x: np.ndarray, y: np.ndarray,
                        scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Lentz soliton template (getWarpFactorByRegion), vectorized.

    Returns (WFX, WFY) shift factors for the rhombus regions of the
    Lentz positive-energy soliton.
    """
    ya = np.abs(y)
    wfx = np.zeros(np.broadcast(x, ya).shape)
    wfy = np.zeros_like(wfx)
    x_b, y_b = np.broadcast_arrays(x, ya)

    regions = [
        ((x_b >= scale) & (x_b <= 2*scale) & (x_b - scale >= y_b), -2.0, 0.0),
        ((x_b > scale) & (x_b <= 2*scale) & (x_b - scale <= y_b)
         & (-y_b + 3*scale >= x_b), -1.0, 1.0),
        ((x_b > 0) & (x_b <= scale) & (x_b + scale > y_b)
         & (-y_b + scale < x_b), 0.0, 1.0),
        ((x_b > 0) & (x_b <= scale) & (x_b + scale <= y_b)
         & (-y_b + 3*scale >= x_b), -0.5, 0.5),
        ((x_b > -scale) & (x_b <= 0) & (-x_b + scale < y_b)
         & (-y_b + 3*scale >= -x_b), 0.5, 0.5),
        ((x_b > -scale) & (x_b <= 0) & (x_b + scale <= y_b)
         & (-y_b + scale >= x_b), 1.0, 0.0),
        ((x_b >= -scale) & (x_b <= scale) & (x_b + scale > y_b), 1.0, 0.0),
    ]
    assigned = np.zeros_like(wfx, dtype=bool)
    # First matching region wins, replicating MATLAB's elseif chain.
    for mask, fx, fy in regions:
        select = mask & ~assigned
        wfx[select] = fx
        wfy[select] = fy
        assigned |= select

    return wfx, np.sign(y)*wfy


def lentz_metric(grid_size: GridSize, world_center: Vector4, v: float,
                 scale: float = None,
                 grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Lentz soliton metric (metricGet_Lentz)."""
    if scale is None:
        scale = max(grid_size[1:4])/7.0
    t, x, y, _ = _world_coordinates(grid_size, world_center, grid_scale)
    xs = v*t
    wfx, wfy = _lentz_warp_factors(x - xs, y, scale)

    alpha, beta, gamma = minkowski_three_plus_one(tuple(int(n) for n in grid_size))
    beta[0] = -wfx*v
    beta[1] = wfy*v

    return _base_metric(
        "Lentz", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v, "scale": scale},
        three_plus_one_builder(alpha, beta, gamma))


def lentz_comoving_metric(grid_size: GridSize, world_center: Vector4, v: float,
                          scale: float = None,
                          grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Lentz soliton metric in the Galilean comoving frame."""
    _require_single_time_slice(grid_size, "lentz_comoving_metric")
    if scale is None:
        scale = max(grid_size[1:4])/7.0
    _, x, y, _ = _world_coordinates(grid_size, world_center, grid_scale)
    wfx, wfy = _lentz_warp_factors(x, y, scale)

    alpha, beta, gamma = minkowski_three_plus_one(tuple(int(n) for n in grid_size))
    beta[0] = v*(1 - wfx)
    beta[1] = v*wfy

    return _base_metric(
        "Lentz Comoving", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v, "scale": scale},
        three_plus_one_builder(alpha, beta, gamma))


def van_den_broeck_metric(grid_size: GridSize, world_center: Vector4, v: float,
                          R1: float, sigma1: float, R2: float, sigma2: float,
                          A: float,
                          grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Van Den Broeck metric (metricGet_VanDenBroeck)."""
    t, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    v_eff = v*(1 + A)**2
    xs = v_eff*t
    r = np.sqrt((x - xs)**2 + y**2 + z**2)

    B = 1 + alcubierre_shape(r, R1, sigma1)*A
    fs = alcubierre_shape(r, R2, sigma2)*v

    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -(1 - B**2*fs**2)
    g[0, 1] = g[1, 0] = -B**2*fs
    g[1, 1] = B**2
    g[2, 2] = B**2
    g[3, 3] = B**2

    return _base_metric(
        "Van Den Broeck", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v_eff, "R1": R1, "sigma1": sigma1,
         "R2": R2, "sigma2": sigma2, "A": A}, g)


def van_den_broeck_comoving_metric(grid_size: GridSize, world_center: Vector4,
                                   v: float, R1: float, sigma1: float,
                                   R2: float, sigma2: float, A: float,
                                   grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Van Den Broeck metric in the Galilean comoving frame."""
    _require_single_time_slice(grid_size, "van_den_broeck_comoving_metric")
    _, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    r = np.sqrt(x**2 + y**2 + z**2)

    B = 1 + alcubierre_shape(r, R1, sigma1)*A
    fs = alcubierre_shape(r, R2, sigma2)*v

    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -(1 - B**2*fs**2)
    g[0, 1] = g[1, 0] = B**2*(v - fs)
    g[1, 1] = B**2
    g[2, 2] = B**2
    g[3, 3] = B**2

    return _base_metric(
        "Van Den Broeck Comoving", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v*(1 + A)**2, "R1": R1, "sigma1": sigma1,
         "R2": R2, "sigma2": sigma2, "A": A}, g)


def modified_time_metric(grid_size: GridSize, world_center: Vector4, v: float,
                         R: float, sigma: float, A: float,
                         grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Modified Time metric (metricGet_ModifiedTime, corrected).

    g_tt = -((1-fs) + fs/A)^2 + (fs v)^2,  g_tx = -v fs.
    """
    t, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    xs = v*t
    r = np.sqrt((x - xs)**2 + y**2 + z**2)
    fs = alcubierre_shape(r, R, sigma)

    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -((1 - fs) + fs/A)**2 + (fs*v)**2
    g[0, 1] = g[1, 0] = -v*fs
    for i in range(1, 4):
        g[i, i] = 1.0

    return _base_metric(
        "Modified Time", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v, "R": R, "sigma": sigma, "A": A}, g)


def modified_time_comoving_metric(grid_size: GridSize, world_center: Vector4,
                                  v: float, R: float, sigma: float, A: float,
                                  grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Modified Time metric in the Galilean comoving frame (corrected)."""
    _require_single_time_slice(grid_size, "modified_time_comoving_metric")
    _, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)
    r = np.sqrt(x**2 + y**2 + z**2)
    fs = alcubierre_shape(r, R, sigma)

    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -((1 - fs) + fs/A)**2 + (fs*v)**2
    g[0, 1] = g[1, 0] = v*(1 - fs)
    for i in range(1, 4):
        g[i, i] = 1.0

    return _base_metric(
        "Modified Time Comoving", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "velocity": v, "R": R, "sigma": sigma, "A": A}, g)


def schwarzschild_metric(grid_size: GridSize, world_center: Vector4, rs: float,
                         grid_scale: Vector4 = (1, 1, 1, 1)) -> SpacetimeTensor:
    """Schwarzschild metric in Cartesian-like coordinates.

    Port of metricGet_Schwarzschild.m: the standard Schwarzschild
    solution with areal radius r, transformed to Cartesian components.
    """
    _require_single_time_slice(grid_size, "schwarzschild_metric")
    _, x, y, z = _world_coordinates(grid_size, world_center, grid_scale)

    # Tiny offset sidesteps division by zero at the origin (as in MATLAB).
    epsilon = 1e-10
    r = np.sqrt(x**2 + y**2 + z**2) + epsilon
    factor = 1 - rs/r

    grid_shape = tuple(int(n) for n in grid_size)
    g = np.zeros((4, 4) + grid_shape)
    g[0, 0] = -factor
    g[1, 1] = (x**2/factor + y**2 + z**2)/r**2
    g[2, 2] = (x**2 + y**2/factor + z**2)/r**2
    g[3, 3] = (x**2 + y**2 + z**2/factor)/r**2
    g[1, 2] = g[2, 1] = rs/(r**3 - r**2*rs)*x*y
    g[1, 3] = g[3, 1] = rs/(r**3 - r**2*rs)*x*z
    g[2, 3] = g[3, 2] = rs/(r**3 - r**2*rs)*y*z

    return _base_metric(
        "Schwarzschild", grid_scale,
        {"grid_size": tuple(grid_size), "world_center": tuple(world_center),
         "rs": rs}, g)
